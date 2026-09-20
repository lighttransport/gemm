param(
    [string]$Destination = ".cuda\13.3"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$CudaRoot = [IO.Path]::GetFullPath((Join-Path $Root $Destination))
$Work = Join-Path $Root "tmp\pixal3d\cuda-windows"

# CUDA 13.3 matches the maximum CUDA level reported by the tested 610.62
# Windows driver. These are NVIDIA's official redistributable toolkit archives.
$Packages = @(
    @("cuda_nvcc", "cuda_nvcc-windows-x86_64-13.3.73-archive.zip", "270214eaee58e49f8fca52a910a46afbfab227858e70897cba8afae10826280b"),
    @("cuda_cudart", "cuda_cudart-windows-x86_64-13.3.29-archive.zip", "1feb7dd266813ffe8dbc24e115183a5ac35a4795c8d34aca0df85ab616b64d9c"),
    @("cuda_crt", "cuda_crt-windows-x86_64-13.3.73-archive.zip", "9227ec7c80db10b7cb0d4ee71ed62ec7ae36e67890216413ab6f9afa35d577f0"),
    @("cuda_cccl", "cuda_cccl-windows-x86_64-13.2.86-archive.zip", "f8d75fba1f3b597ae625283fe3a03928f09223be75941a69c497e2b412b752be"),
    @("libnvvm", "libnvvm-windows-x86_64-13.3.73-archive.zip", "ca8f11d5173ac16a166be8fafefbf9676542a097de1fce61b3f17696dffc1f27"),
    @("libcublas", "libcublas-windows-x86_64-13.6.0.2-archive.zip", "62e9fa30560c8f0a28e0cdcf9d6fc1fed347bcfab8847239b9ae1fdc1d86408a")
)

New-Item -ItemType Directory -Force $CudaRoot, $Work | Out-Null
foreach ($Package in $Packages) {
    $Component, $Archive, $Sha256 = $Package
    $Url = "https://developer.download.nvidia.com/compute/cuda/redist/$Component/windows-x86_64/$Archive"
    $Zip = Join-Path $Work $Archive
    $Extract = Join-Path $Work $Component
    if (-not (Test-Path $Zip)) {
        & curl.exe -L --fail --retry 3 -o $Zip $Url
        if ($LASTEXITCODE) { throw "Download failed: $Url" }
    }
    $Actual = (Get-FileHash $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Actual -ne $Sha256) {
        throw "CUDA archive SHA-256 mismatch: $Archive ($Actual)"
    }
    Remove-Item -Recurse -Force $Extract -ErrorAction SilentlyContinue
    Expand-Archive -Path $Zip -DestinationPath $Extract -Force
    $Payload = Get-ChildItem $Extract -Directory | Select-Object -First 1
    if (-not $Payload) { throw "CUDA archive has no payload directory: $Archive" }
    Copy-Item (Join-Path $Payload.FullName "*") $CudaRoot -Recurse -Force
    Remove-Item -Recurse -Force $Extract
}

$Nvcc = Join-Path $CudaRoot "bin\nvcc.exe"
if (-not (Test-Path $Nvcc)) { throw "CUDA compiler was not installed: $Nvcc" }
& $Nvcc --version
if ($LASTEXITCODE) { throw "CUDA compiler validation failed" }
Write-Output "CUDA toolkit root: $CudaRoot"
