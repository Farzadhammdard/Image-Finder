@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found.
  echo Run setup first: python -m venv .venv
  pause
  exit /b 1
)

set "INDEX_OUT=%LOCALAPPDATA%\ImageFinder\index_data"
echo Building index from Desktop...
echo Output: %INDEX_OUT%
".venv\Scripts\python.exe" -m image_finder.cli index --folders "C:\Users\Kabul\Desktop" --output "%INDEX_OUT%"

if errorlevel 1 (
  echo.
  echo Index build failed.
  pause
  exit /b 1
)

echo.
echo Index build completed.
pause
