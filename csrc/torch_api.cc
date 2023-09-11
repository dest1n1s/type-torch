#include "torch_api.h"
#include <c10/core/Layout.h>

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

} // namespace TypeTorch