@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment not found.
  echo Run setup first: python -m venv .venv
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m image_finder.cli gui --top-k 10

if errorlevel 1 (
  echo.
  echo image-finder exited with an error.
  pause
)
