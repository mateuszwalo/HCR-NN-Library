@echo off
setlocal enabledelayedexpansion

:: run to build gpu library only, either as ptx or cubin

:: ==============================
:: CUDA PTX build script
:: ==============================

:: Directory setup
set SRC_DIR=%cd%
set OUT_DIR=%cd%\build\pycuda

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo.
echo ==============================
echo Building CUDA PTX files...
echo Source: %SRC_DIR%
echo Output: %OUT_DIR%
echo ==============================

:: Compile all .cu files in folder into single PTX
"%CUDA_PATH%\bin\nvcc.exe" -arch=sm_86 -ptx "%SRC_DIR%\*.cu" -o "%OUT_DIR%\hcr_kernels.ptx"

if %errorlevel% neq 0 (
    echo.
    echo Compilation failed!
    pause
    exit /b 1
)

echo.
echo Done! Generated: "%OUT_DIR%\hcr_kernels.ptx"
pause
endlocal
