#include "hello.h"
#include "torch_api_generated.h"

namespace TypeTorch {
Napi::Object InitAll(Napi::Env env, Napi::Object exports) {
  InitTypeTorchGenerated(env, exports);
  InitHello(env, exports);
  return exports;
}

NODE_API_MODULE(TypeTorch, InitAll)
} // namespace TypeTorch