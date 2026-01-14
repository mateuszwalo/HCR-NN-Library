@echo off
setlocal
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0

set SRC_DIR=%cd%
set OUT_DIR=%cd%\build\pycuda
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo.
echo ==========================================
echo Building hcr_kernels.dll for RTX 5080...
echo Architecture: sm_120 (Blackwell)
echo CUDA Path: %CUDA_PATH%
echo Output: %OUT_DIR%\hcr_kernels.dll
echo ==========================================

"%CUDA_PATH%\bin\nvcc.exe" ^
 -arch=sm_120 ^
 -Xcompiler="/LD" ^
 "%SRC_DIR%\conditional_estimation_kernel.cu" ^
 "%SRC_DIR%\base_optimization_kernel.cu" ^
 "%SRC_DIR%\dynamic_ema_kernel.cu" ^
 "%SRC_DIR%\entropy_mi_kernel.cu" ^
 "%SRC_DIR%\mean_estimation_kernel.cu" ^
 "%SRC_DIR%\propagation_estimation_kernel.cu" ^
 -o "%OUT_DIR%\hcr_kernels.dll"

if %errorlevel% neq 0 (
    echo.
    echo Build failed! Check CUDA 13.0 installation and include paths.
    pause
    exit /b 1
)

echo.
echo Done! hcr_kernels.dll built successfully.
pause
endlocal
