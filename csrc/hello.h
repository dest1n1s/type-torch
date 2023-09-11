#pragma once

#include <napi.h>
#include <torch/torch.h>

namespace TypeTorch {

static Napi::External<torch::Tensor> Create(const Napi::CallbackInfo &info);

static Napi::Value Get(const Napi::CallbackInfo &info);

static Napi::Object InitHello(Napi::Env env, Napi::Object exports);

} // namespace TypeTorch