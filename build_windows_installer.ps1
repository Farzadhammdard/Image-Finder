param(
    [string]$SourceDistDir = "dist_v3\ImageFinder"
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$defaultSourceDistDir = "dist_v3\ImageFinder"
$distExe = Join-Path $PSScriptRoot (Join-Path $SourceDistDir "ImageFinder.exe")
if (-not (Test-Path $distExe)) {
    if ($SourceDistDir -ne $defaultSourceDistDir) {
        throw "Executable not found in custom source path: $distExe"
    }

    Write-Host "Executable not found. Building exe first..."
    & powershell -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_windows_exe.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build executable."
    }
}

$isccCandidates = @(
    "$Env:ProgramFiles(x86)\Inno Setup 6\ISCC.exe",
    "$Env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$Env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)

$iscc = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $iscc) {
    throw "Inno Setup compiler (ISCC.exe) not found. Install Inno Setup 6 and rerun."
}

$issPath = Join-Path $PSScriptRoot "installer\image_finder.iss"
$sourceDirForIss = "..\" + $SourceDistDir.Replace("/", "\")

Push-Location (Join-Path $PSScriptRoot "installer")
try {
    & $iscc "/DMySourceDir=$sourceDirForIss" $issPath
    if ($LASTEXITCODE -ne 0) {
        throw "Installer compilation failed."
    }
}
finally {
    Pop-Location
}

Write-Host "Installer build complete."
Write-Host "Output folder: $PSScriptRoot\installer\output"
