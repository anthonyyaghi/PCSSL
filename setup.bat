@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  DBBD Setup ^& Installation Script
echo ============================================

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.8 or higher and ensure it is in your PATH.
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist venv (
    echo.
    echo [1/5] Creating virtual environment 'venv'...
    python -m venv venv
) else (
    echo.
    echo [1/5] Virtual environment 'venv' already exists. Skipping creation.
)

:: Define paths for the virtual environment
set VENV_PYTHON=%~dp0venv\Scripts\python.exe
set VENV_PIP=%~dp0venv\Scripts\pip.exe
set VENV_PYTEST=%~dp0venv\Scripts\pytest.exe

echo.
echo [2/5] Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip

:: Install PyTorch with specific CUDA toolkit version
echo.
echo [3/5] Installing PyTorch with CUDA 12.4...
"%VENV_PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

:: Install everything including spunet dependencies
echo.
echo [4/5] Installing DBBD package with dev and spunet dependencies...
"%VENV_PIP%" install -e ".[dev,spunet]"

echo.
echo [5/5] Running Tests to confirm everything is green...
echo ============================================
"%VENV_PYTEST%" tests/ -v

if %ERRORLEVEL% equ 0 (
    echo ============================================
    echo  Setup Complete! All tests passed and everything is GREEN. 
    echo  
    echo  To activate the environment, run:
    echo  .\venv\Scripts\activate.bat
    echo ============================================
) else (
    echo ============================================
    echo  Tests failed. Please check the output above.
    echo ============================================
    exit /b 1
)
