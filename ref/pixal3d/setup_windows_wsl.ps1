param(
    [string]$Distribution = "ubu2404"
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Drive = $Root.Substring(0, 1).ToLowerInvariant()
$Relative = $Root.Substring(2).Replace("\", "/")
$WslRoot = "/mnt/$Drive$Relative"
& wsl.exe -d $Distribution -- bash "$WslRoot/ref/pixal3d/setup_windows_wsl.sh"
if ($LASTEXITCODE) { throw "Pixal3D WSL setup failed" }
