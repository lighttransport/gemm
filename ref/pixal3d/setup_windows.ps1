param(
    [string]$Environment = ".venv-pixal3d-cuda",
    [string]$Python = "3.12",
    [switch]$SkipToolkit
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Venv = Join-Path $Root $Environment
$PythonExe = Join-Path $Venv "Scripts\python.exe"

Push-Location $Root
try {
    if (-not (Test-Path $PythonExe)) {
        uv venv --python $Python $Venv
        if ($LASTEXITCODE) { throw "uv venv failed" }
    }
    uv pip install --python $PythonExe --index-url https://download.pytorch.org/whl/cu128 `
        "torch==2.7.1" "torchvision==0.22.1"
    if ($LASTEXITCODE) { throw "PyTorch installation failed" }
    uv pip install --python $PythonExe "numpy==2.5.2" "safetensors==0.8.0" `
        "ninja==1.13.2" "packaging==26.3" "nvidia-cuda-nvcc-cu12==12.9.86"
    if ($LASTEXITCODE) { throw "Pixal3D validation dependency installation failed" }
    & $PythonExe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
    if ($LASTEXITCODE) { throw "PyTorch CUDA validation failed" }
    if (-not $SkipToolkit) {
        & (Join-Path $PSScriptRoot "setup_windows_cuda.ps1")
    }
} finally {
    Pop-Location
}
