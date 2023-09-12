#include "torch_api.h"
#include "napi.h"
#include <ATen/core/TensorBody.h>
#include <ATen/ops/fill.h>
#include <ATen/ops/full.h>
#include <c10/core/TensorOptions.h>
#include <cstdint>

namespace TypeTorch {

at::Device device_of_int(int d) {
  if (d == -3)
    return at::Device(at::kVulkan);
  if (d == -2)
    return at::Device(at::kMPS);
  if (d < 0)
    return at::Device(at::kCPU);
  return at::Device(at::kCUDA, /*index=*/d);
}

std::vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len) {
  std::vector<torch::Tensor> result;
  for (int i = 0; i < len; ++i)
    result.push_back(*(vs[i]));
  return result;
}

c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs,
                                                             int len) {
  std::vector<c10::optional<torch::Tensor>> result;
  for (int i = 0; i < len; ++i) {
    result.push_back(vs[i] != nullptr ? c10::optional<torch::Tensor>(*(vs[i]))
                                      : c10::nullopt);
  }
  return c10::List<c10::optional<torch::Tensor>>(result);
}

Napi::External<torch::Tensor> at_new_tensor(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  int64_t value = info[0].As<Napi::Number>().Int64Value();
  Napi::Object options_obj = info[1].As<Napi::Object>();
  int device = options_obj.Get("device").As<Napi::Number>().Int32Value();
  int dtype = options_obj.Get("dtype").As<Napi::Number>().Int32Value();
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(static_cast<at::ScalarType>(dtype))
                                  .device(device_of_int(device));

  torch::Tensor *tensor = new torch::Tensor(at::full({}, value, options));
  return Napi::External<torch::Tensor>::New(env, tensor);
}

Napi::External<torch::Tensor>
at_new_tensor_from_array(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  // Nested javascript array
  Napi::Array array = info[0].As<Napi::Array>();

  // Get the shape of the tensor
  std::vector<int64_t> shape;
  Napi::Value first = array.Get((uint32_t)0);
  while (first.IsArray()) {
    shape.push_back(first.As<Napi::Array>().Length());
    first = first.As<Napi::Array>().Get((uint32_t)0);
  }

  // Convert the javascript array to a vector
  int64_t size = 1;
  for (int i = 0; i < shape.size(); ++i)
    size *= shape[i];
  std::vector<double> values(size);
  std::vector<int64_t> indices(shape.size(), 0);
  while (true) {
    Napi::Value value = array;
    for (int i = 0; i < shape.size(); ++i) {
      value = value.As<Napi::Array>().Get(indices[i]);
      if (value.IsUndefined()) {
        value = Napi::Number::New(env, 0);
        break;
      }
    }
    values.push_back(value.As<Napi::Number>().DoubleValue());
    int i = 0;
    while (i < shape.size() && indices[i] == shape[i] - 1) {
      indices[i] = 0;
      i++;
    }
    if (i == shape.size())
      break;
    indices[i]++;
  }

  // Get the device and dtype
  Napi::Object options_obj = info[1].As<Napi::Object>();
  int device = options_obj.Get("device").As<Napi::Number>().Int32Value();
  int dtype = options_obj.Get("dtype").As<Napi::Number>().Int32Value();

  // Create the tensor
  at::TensorOptions options = at::TensorOptions()
                                  .dtype(static_cast<at::ScalarType>(dtype))
                                  .device(device_of_int(device));
  torch::Tensor *tensor =
      new torch::Tensor(at::from_blob(values.data(), shape, options));
  return Napi::External<torch::Tensor>::New(env, tensor);
}

Napi::Value at_to_array(const Napi::CallbackInfo &info) {
  Napi::Env env = info.Env();
  Napi::HandleScope scope(env);

  torch::Tensor *tensor = info[0].As<Napi::External<torch::Tensor>>().Data();
  std::vector<int64_t> shape = tensor->sizes().vec();

  if (shape.size() == 0) {
    return Napi::Number::New(env, tensor->item<double>());
  }

  Napi::Array array = Napi::Array::New(env, shape[0]);

  std::vector<int64_t> indices(shape.size(), 0);
  while (true) {
    std::vector<at::indexing::TensorIndex> tensor_indices;
    for (int i = 0; i < shape.size(); ++i) {
      tensor_indices.push_back(indices[i]);
    }
    double value = tensor->index(tensor_indices).item<double>();

    for (int i = 0; i < shape.size() - 1; ++i) {
      if (array.Get(indices[i]).IsUndefined()) {
        array.Set(indices[i], Napi::Array::New(env, shape[i + 1]));
      }
      array = array.Get(indices[i]).As<Napi::Array>();
    }

    array.Set(indices[shape.size() - 1], Napi::Number::New(env, value));

    int i = 0;
    while (i < shape.size() && indices[i] == shape[i] - 1) {
      indices[i] = 0;
      i++;
    }
    if (i == shape.size())
      break;
    indices[i]++;
  }

  return array;
}

Napi::Object InitTorchAPI(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "at_new_tensor"),
              Napi::Function::New(env, at_new_tensor));
  exports.Set(Napi::String::New(env, "at_new_tensor_from_array"),
              Napi::Function::New(env, at_new_tensor_from_array));
  exports.Set(Napi::String::New(env, "at_to_array"),
              Napi::Function::New(env, at_to_array));
  return exports;
}

} // namespace TypeTorch