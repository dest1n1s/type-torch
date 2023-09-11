#pragma once

#include <torch/torch.h>
#include <vector>

namespace TypeTorch {

at::Device device_of_int(int d);
std::vector<torch::Tensor> of_carray_tensor(torch::Tensor **vs, int len);
c10::List<c10::optional<torch::Tensor>> of_carray_tensor_opt(torch::Tensor **vs,
                                                             int len);

} // namespace TypeTorch