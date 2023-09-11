#pragma once

#include <napi.h>
#include <torch/torch.h>

namespace TypeTorch {

Napi::External<torch::Tensor> Create(const Napi::CallbackInfo &info);

Napi::Value Get(const Napi::CallbackInfo &info);

Napi::Object InitHello(Napi::Env env, Napi::Object exports);

} // namespace TypeTorch