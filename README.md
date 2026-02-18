# Image Finder

Image similarity search for CNC workflows (optimized for black/white pattern matching).

## Download And Install (For End Users)

1. Open your repository `Releases` page:
   `https://github.com/<YOUR_USERNAME>/<YOUR_REPO>/releases`
2. Download `ImageFinderSetup.exe` from the latest release.
3. Run installer, finish setup, and launch **Image Finder**.
4. First run only: click **Rebuild Desktop Index** once.

## Run From Source (Developer Mode)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m image_finder.cli gui
```

## Build Windows EXE

```powershell
.\build_windows_exe.ps1
```

Output:
`dist_v3\ImageFinder\ImageFinder.exe`

If folder is locked (app running), use an alternate output:

```powershell
.\build_windows_exe.ps1 -DistPath dist_v3_alt -WorkPath build_v3_alt
```

## Build Windows Installer (Setup.exe)

Prerequisite:
- Install `Inno Setup 6`

```powershell
.\build_windows_installer.ps1
```

Output:
`installer\output\ImageFinderSetup.exe`

If EXE is in another folder:

```powershell
.\build_windows_installer.ps1 -SourceDistDir "dist_v3_alt\ImageFinder"
```

## Publish On GitHub (First Time)

```powershell
git init
git add .
git commit -m "Initial release"
git branch -M main
git remote add origin https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
git push -u origin main
```

## Auto-Build Release Installer On Tag

This repo includes:
`\.github\workflows\windows-release.yml`

When you push a tag like `v3.0.0`, GitHub Actions will:
- Build `ImageFinder.exe`
- Build `ImageFinderSetup.exe`
- Attach both files to the GitHub Release automatically

Release command:

```powershell
git tag v3.0.0
git push origin v3.0.0
```

## Index Location

Default index path:
`%LOCALAPPDATA%\ImageFinder\index_data`

This keeps app data writable after normal installation in Program Files.
