from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="HCR_kernels",
    ext_modules=[
        CUDAExtension(
            name="HCR_kernels",
            sources=["layers_kernel_bindings.cpp", 
                     "mean_estimation_kernel_wrapper.cu",
                     "conditional_estimation_kernel.cu",
                     "propagation_estimation_kernel.cu",
                     "dynamic_ema_kernel.cu",
                     "entropy_mi_kernel.cu",
                     "base_optimization.cu"
                     ],
            extra_compile_args={'cxx': ['-g'],
                    'nvcc': ['-O2']},
            extra_link_args=['-Wl,--no-as-needed', "-s"]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
