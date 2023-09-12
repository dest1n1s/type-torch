#pragma once

#include <napi.h>
#include <torch/torch.h>
#include <vector>

namespace TypeTorch {

at::Device device_of_int(int d);
std::vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len);
c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs,
                                                             int len);

Napi::External<torch::Tensor> at_new_tensor(const Napi::CallbackInfo &info);
Napi::External<torch::Tensor>
at_new_tensor_from_array(const Napi::CallbackInfo &info);
Napi::Value at_to_array(const Napi::CallbackInfo &info);

Napi::Object InitTorchAPI(Napi::Env env, Napi::Object exports);
} // namespace TypeTorch