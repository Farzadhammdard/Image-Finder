param(
    [string]$DistPath = "dist_v4",
    [string]$WorkPath = "build_v4_test",
    [string]$OutputZip = "installer\output\ImageFinder-v-4.0.0-test.zip"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$sourceDir = Join-Path $PSScriptRoot (Join-Path $DistPath "ImageFinder")
$exePath = Join-Path $sourceDir "ImageFinder.exe"
if (-not (Test-Path $exePath)) {
    Write-Host "Test executable not found. Building exe first..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_windows_exe.ps1") -DistPath $DistPath -WorkPath $WorkPath
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build executable for test package."
    }
}

$zipPath = Join-Path $PSScriptRoot $OutputZip
$zipDir = Split-Path -Parent $zipPath
if (-not (Test-Path $zipDir)) {
    New-Item -ItemType Directory -Path $zipDir | Out-Null
}
if (Test-Path $zipPath) {
    Remove-Item -Force $zipPath
}

Compress-Archive -Path (Join-Path $sourceDir "*") -DestinationPath $zipPath -CompressionLevel Optimal

Write-Host "Test package ready."
Write-Host "Output file: $zipPath"
