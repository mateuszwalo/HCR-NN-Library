from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="mean_estimation",
    ext_modules=[
        CUDAExtension(
            name="layers_kernels",
            sources=["layers_kernel_bindings.cpp", 
                     "mean_estimation_kernel_wrapper.cu",
                     "conditional_estimation_kernel.cu",
                     "propagation_estimation_kernel.cu",
                     "dynamic_ema_kernel.cu",
                     "entropy_mi_kernel.cu",
                     ],
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
