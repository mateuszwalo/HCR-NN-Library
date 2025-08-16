#include <torch/extension.h>
#include <cuda_runtime.h>

// kernel functions
void mean_estimation_cuda();
void bilinear_cuda();
void entropy_cuda();
void mi_cuda();
void ema_update_cuda();
void context_cuda();
void denominator_cuda();

//Python high-level binfing
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean_estimation_cu", &mean_estimation_cuda, "Perform mean estimation (CUDA)");
    m.def("propagation_estimation_cu", &bilinear_cuda, "Propagate estimation (CUDA)");
    m.def("entropy_cu", &entropy_cuda, "Calculate entropy (CUDA)");
    m.def("mutual_information_cu", &mi_cuda, "Calculate mutual information (CUDA)");
    m.def("ema_update_cu", &ema_update_cuda, "Calculate ema (CUDA)");
    m.def("ema_update_cu", &ema_update_cuda, "Perform conditional estimation (CUDA)");
    m.def("context_calculation_cu", &context_cuda, "Calculate context (CUDA)");
    m.def("denominator_cu", &denominator_cuda, "Calculate denominator (CUDA)");
}