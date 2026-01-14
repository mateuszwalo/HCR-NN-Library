@echo off
setlocal
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0
:: the code compiles the kernels into dynamic dll library
echo Building mycuda.dll...
"%CUDA_PATH%\bin\nvcc.exe" -Xcompiler="/LD" conditional_estimation_kernel.cu base_optimization_kernel.cu dynamic_ema_kernel.cu entropy_mi_kernel.cu mean_estimation_kernel.cu propagation_estimation_kernel.cu -o hcr_kernels.dll

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo Done: hcr_kernels.dll built successfully!
pause
endlocal
