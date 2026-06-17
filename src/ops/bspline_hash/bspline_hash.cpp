#include <torch/extension.h>
#include <vector>

torch::Tensor bspline_hash_forward_cuda(
    torch::Tensor positions,
    torch::Tensor hash_table,
    int64_t base_res,
    int64_t max_res);

torch::Tensor bspline_hash_backward_hash_cuda(
    torch::Tensor grad_output,
    torch::Tensor positions,
    std::vector<int64_t> hash_table_shape,
    int64_t base_res,
    int64_t max_res);

torch::Tensor forward(
    torch::Tensor positions,
    torch::Tensor hash_table,
    int64_t base_res,
    int64_t max_res) {
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(hash_table.is_cuda(), "hash_table must be a CUDA tensor");
  TORCH_CHECK(positions.scalar_type() == torch::kFloat32, "positions must be float32");
  TORCH_CHECK(hash_table.scalar_type() == torch::kFloat32, "hash_table must be float32");
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "positions must have shape [N, 3]");
  TORCH_CHECK(hash_table.dim() == 3, "hash_table must have shape [L, T, F]");
  return bspline_hash_forward_cuda(
      positions.contiguous(), hash_table.contiguous(), base_res, max_res);
}

torch::Tensor backward_hash(
    torch::Tensor grad_output,
    torch::Tensor positions,
    std::vector<int64_t> hash_table_shape,
    int64_t base_res,
    int64_t max_res) {
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");
  TORCH_CHECK(positions.scalar_type() == torch::kFloat32, "positions must be float32");
  TORCH_CHECK(hash_table_shape.size() == 3, "hash_table_shape must have 3 entries");
  return bspline_hash_backward_hash_cuda(
      grad_output.contiguous(), positions.contiguous(), hash_table_shape, base_res, max_res);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "B-Spline hash-grid forward (CUDA)");
  m.def("backward_hash", &backward_hash, "B-Spline hash-grid hash-table backward (CUDA)");
}
