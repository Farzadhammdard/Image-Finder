# Image Finder

Image and DXF similarity search for CNC workflows (optimized for black/white pattern matching).
Current release version: `v-4.0.0`

Search stack now includes:
- OpenCV handcrafted features
- OCR text reranking
- AI image embeddings (CLIP via `sentence-transformers`)
- FAISS approximate candidate search
- DXF parser/rasterizer (`ezdxf`) for shape matching
- PySide6 desktop UI

## Download And Install (For End Users)

طراحی شده توسط انجینیر احمد فرزاد همدرد

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

Optional:
- Set model name with env var `IMAGE_FINDER_EMBED_MODEL` (default: `clip-ViT-B-32`)
- First embedding-enabled run may download model weights

## Build Windows EXE

```powershell
.\build_windows_exe.ps1
```

Output:
`dist_v4\ImageFinder\ImageFinder.exe`

If folder is locked (app running), use an alternate output:

```powershell
.\build_windows_exe.ps1 -DistPath dist_v4_alt -WorkPath build_v4_alt
```

## Build Test Package (Before Installer)

Use this to test app first without installer:

```powershell
.\build_windows_test_package.ps1
```

Output:
`installer\output\ImageFinder-v-4.0.0-test.zip`

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
.\build_windows_installer.ps1 -SourceDistDir "dist_v4_alt\ImageFinder"
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

When you push a tag like `v-4.0.0`, GitHub Actions will:
- Build `ImageFinder.exe`
- Build `ImageFinderSetup.exe`
- Attach both files to the GitHub Release automatically

Release command:

```powershell
git tag v-4.0.0
git push origin v-4.0.0
```

## Index Location

Default index path:
`%LOCALAPPDATA%\ImageFinder\index_data`

This keeps app data writable after normal installation in Program Files.

Embedding files stored in the same index folder:
- `embeddings.npy`
- `embeddings_meta.json`
- `embeddings.faiss`
