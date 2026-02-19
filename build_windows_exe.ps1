param(
    [string]$DistPath = "dist_v4",
    [string]$WorkPath = "build_v4"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python virtual environment not found. Create .venv first."
}

$distFull = Join-Path $PSScriptRoot $DistPath
$workFull = Join-Path $PSScriptRoot $WorkPath

Write-Host "Installing runtime dependencies..."
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "pip install requirements.txt failed."
}

Write-Host "Installing build tools..."
& $python -m pip install -r requirements-build.txt
if ($LASTEXITCODE -ne 0) {
    throw "pip install requirements-build.txt failed."
}

Write-Host "Building ImageFinder.exe ..."
$pyInstallerArgs = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",
    "--name", "ImageFinder",
    "--collect-all", "rapidocr_onnxruntime",
    "--collect-all", "sentence_transformers",
    "--collect-all", "faiss",
    "--collect-all", "PySide6",
    "--paths", $PSScriptRoot,
    "--distpath", $distFull,
    "--workpath", $workFull
)

$assetsDir = Join-Path $PSScriptRoot "assets"
if (Test-Path $assetsDir) {
    $pyInstallerArgs += @("--add-data", "assets;assets")
}

$iconPath = Join-Path $PSScriptRoot "assets\app_icon.ico"
if (Test-Path $iconPath) {
    $pyInstallerArgs += @("--icon", $iconPath)
}

$pyInstallerArgs += "app_main.py"
& $python @pyInstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed."
}

Write-Host "Build complete."
Write-Host "Output folder: $distFull\ImageFinder"
