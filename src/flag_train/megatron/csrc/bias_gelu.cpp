// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include <torch/csrc/autograd/custom_function.h>
#include <torch/extension.h>

torch::Tensor bias_gelu_cuda(torch::Tensor x, torch::Tensor b);
torch::Tensor bias_gelu_grad_cuda(torch::Tensor x, torch::Tensor b,
                                  torch::Tensor g);

std::vector<torch::Tensor>
bias_gelu_fused_grad_cuda(torch::Tensor x, torch::Tensor b, torch::Tensor g);

class BiasGelu : public torch::autograd::Function<BiasGelu> {
public:
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                               torch::Tensor x, torch::Tensor b) {
    ctx->save_for_backward({x, b});
    return bias_gelu_cuda(x, b);
  }
  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grads) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),
                "bias_gelu supports first-order gradients only");
    auto saved = ctx->get_saved_variables();
    auto g = grads[0].contiguous();
    if (saved[0].numel() > (1 << 20) && saved[1].numel() % 4 == 0 &&
        saved[0].numel() / saved[1].numel() <= 16 * 65535 &&
        uintptr_t(saved[0].data_ptr()) % 16 == 0 &&
        uintptr_t(saved[1].data_ptr()) % 16 == 0 &&
        uintptr_t(g.data_ptr()) % 16 == 0 && ctx->needs_input_grad(1)) {
      auto result = bias_gelu_fused_grad_cuda(saved[0], saved[1], g);
      return {ctx->needs_input_grad(0) ? result[0] : torch::Tensor(),
              result[1]};
    }
    auto dx = bias_gelu_grad_cuda(saved[0], saved[1], g);
    return {ctx->needs_input_grad(0) ? dx : torch::Tensor(),
            ctx->needs_input_grad(1) ? dx.sum_to_size(saved[1].sizes())
                                     : torch::Tensor()};
  }
};

torch::Tensor apply(torch::Tensor x, torch::Tensor b) {
  return BiasGelu::apply(x, b);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("apply", &apply); }
