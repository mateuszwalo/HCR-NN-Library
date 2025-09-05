#include <torch/extension.h>

// kernel functions
void mean_estimation_cuda();
void propagate_expectation_cuda();
void approximate_entropy_cuda();
void approximate_mi_cuda();
void ema_update_cuda();
void conditional_estimation_cuda();
void transform_tensor_cuda();

//Python high-level binfing
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_estimation_cu", &mean_estimation_cuda, "Perform mean estimation (CUDA)");
    m.def("propagate_expectation_cu", &propagate_expectation_cuda, "Propagate estimation (CUDA)");
    m.def("approximate_entropy_cu", &approximate_entropy_cuda, "Calculate entropy (CUDA)");
    m.def("approximate_mi_cu", &approximate_mi_cuda, "Calculate mutual information (CUDA)");
    m.def("ema_update_cu", &ema_update_cuda, "Calculate ema (CUDA)");
    m.def("conditional_estimation_cu", &conditional_estimation_cuda, "Calculate conidtional estimation (CUDA)");
    m.def("base_optimization_cu", &transform_tensor_cuda, "Optimize tensor base (CUDA)");
}
