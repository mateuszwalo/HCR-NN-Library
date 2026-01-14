@echo off
setlocal enabledelayedexpansion

set SRC_DIR=%cd%
set OUT_DIR=%cd%\build\pycuda
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"

echo.
echo ==========================================
echo Building CUDA PTX files for RTX 5080...
echo Architecture: sm_120 (Blackwell)
echo Source: %SRC_DIR%
echo Output: %OUT_DIR%
echo ==========================================

for %%f in ("%SRC_DIR%\*.cu") do (
    echo Compiling %%~nxf ...
    "%CUDA_PATH%\bin\nvcc.exe" -arch=sm_120 -ptx "%%f" -o "%OUT_DIR%\%%~nf.ptx"
    if !errorlevel! neq 0 (
        echo Compilation failed on %%~nxf
        pause
        exit /b 1
    )
)

echo.
echo Done! All PTX files are in "%OUT_DIR%"
pause
endlocal
