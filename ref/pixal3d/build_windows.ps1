param(
    [string]$CudaRoot = "",
    [string]$Environment = ".venv-pixal3d-cuda",
    [string]$Architecture = "86",
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$PythonExe = Join-Path $Root "$Environment\Scripts\python.exe"
$Source = Join-Path $Root "cuda\pixal3d"
$Build = Join-Path $Source "build-windows"
$Ninja = Join-Path $Root "$Environment\Scripts\ninja.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Missing $PythonExe; run ref\pixal3d\setup_windows.ps1 first"
}
if (-not $CudaRoot) {
    $CudaRoot = $env:CUDA_PATH
}
if (-not $CudaRoot) {
    $CudaRoot = Join-Path $Root ".cuda\13.3"
}
if (-not $CudaRoot -or -not (Test-Path (Join-Path $CudaRoot "bin\nvcc.exe"))) {
    throw "A complete NVIDIA CUDA Toolkit with nvcc.exe is required; pass -CudaRoot or set CUDA_PATH"
}
if (-not (Test-Path $Ninja)) { throw "Missing Ninja executable: $Ninja" }
$env:PATH = "$(Join-Path $CudaRoot 'bin\x64');$(Join-Path $CudaRoot 'bin');$env:PATH"

$VsDevCmd = Get-ChildItem "C:\Program Files\Microsoft Visual Studio\2022\*\Common7\Tools\VsDevCmd.bat" `
    -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $VsDevCmd) { throw "Visual Studio 2022 C++ tools are required" }
# Import the MSVC command-prompt environment so Ninja and nvcc can locate cl.exe,
# the Windows SDK, and linker tools without requiring CUDA's VS integration.
& cmd.exe /d /s /c "`"$($VsDevCmd.FullName)`" -no_logo -arch=x64 -host_arch=x64 && set" | `
    ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }

$Cache = Join-Path $Build "CMakeCache.txt"
if ((Test-Path $Cache) -and
    -not (Select-String -Path $Cache -Pattern '^CMAKE_GENERATOR:INTERNAL=Ninja$' -Quiet)) {
    cmake -E remove_directory $Build
    if ($LASTEXITCODE) { throw "Could not reset the Windows build directory" }
}

$arguments = @(
    "-S", $Source, "-B", $Build, "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_MAKE_PROGRAM=$Ninja",
    "-DCMAKE_CUDA_COMPILER=$(Join-Path $CudaRoot 'bin\nvcc.exe')",
    "-DCUDAToolkit_ROOT=$CudaRoot",
    "-DPython3_EXECUTABLE=$PythonExe",
    "-DCMAKE_CUDA_ARCHITECTURES=$Architecture"
)
cmake @arguments
if ($LASTEXITCODE) { throw "CMake configuration failed" }
cmake --build $Build --parallel
if ($LASTEXITCODE) { throw "Pixal3D CUDA compilation failed" }

$Library = Join-Path $Build "pixal3d_cuda.dll"
if (-not (Test-Path $Library)) {
    throw "Expected CUDA plugin was not produced: $Library"
}
& $PythonExe (Join-Path $PSScriptRoot "validate_resident.py") `
    --backend cuda --kernels mma --library $Library `
    --dll-directory (Join-Path $CudaRoot "bin\x64")
if ($LASTEXITCODE) { throw "Pixal3D resident CUDA validation failed" }
