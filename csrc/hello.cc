#include <napi.h>
#include <torch/torch.h>

namespace Demo {

static Napi::External<torch::Tensor> Create(const Napi::CallbackInfo& info) {
  // Napi::Env is the opaque data structure containing the environment in which
  // the request is being run. We will need this env when we want to create any
  // new objects inside of the node.js environment
  Napi::Env env = info.Env();

  // Create a C++ level variable
  torch::Tensor* myTensor = new torch::Tensor(torch::ones({2, 2}));

  // Print the address of the C++ level variable
  printf("C++ level variable address: %p\n", myTensor);

  // Return a pointer to the C++ level variable as a Napi::External. This will
  // ensure that the pointer is passed through the JavaScript layer as-is,
  // without any copying or manipulation.
  return Napi::External<torch::Tensor>::New(env, myTensor);
}

static Napi::Value Get(const Napi::CallbackInfo& info) {
  // Napi::Env is the opaque data structure containing the environment in which
  // the request is being run. We will need this env when we want to create any
  // new objects inside of the node.js environment
  Napi::Env env = info.Env();

  // Extract the C++ level variable from the JavaScript wrapper
  Napi::External<torch::Tensor> ext =
      info[0].As<Napi::External<torch::Tensor>>();
  torch::Tensor* myTensor = ext.Data();

  // Print the address of the C++ level variable
  printf("C++ level variable address: %p\n", myTensor);

  // Convert to a nested JavaScript array
  Napi::Array arr = Napi::Array::New(env, myTensor->size(0));
  for (int i = 0; i < myTensor->size(0); i++) {
    auto arr_inner = Napi::Array::New(env, myTensor->size(1));
    for (int j = 0; j < myTensor->size(1); j++) {
      arr_inner[j] = myTensor->index({i, j}).item<float>();
    }
    arr[i] = arr_inner;
  }

  // Return the value of the C++ level variable as a Napi::Number. This will
  // ensure that the value is passed through the JavaScript layer as-is,
  // without any copying or manipulation.
  return arr;
}

static Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "create"),
              Napi::Function::New(env, Create));
  exports.Set(Napi::String::New(env, "get"), Napi::Function::New(env, Get));
  return exports;
}
NODE_API_MODULE(hello, Init)
}  // namespace Demo
