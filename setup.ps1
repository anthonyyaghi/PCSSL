$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host " DBBD Setup & Installation Script"
Write-Host "============================================"

# Check for Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found Python"
} catch {
    Write-Host "Python not found! Please install Python 3.8 or higher and ensure it is in your PATH."
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "`n[1/5] Creating virtual environment 'venv'..."
    python -m venv venv
} else {
    Write-Host "`n[1/5] Virtual environment 'venv' already exists. Skipping creation."
}

# Define paths for the virtual environment
$venvScripts = "$PSScriptRoot\venv\Scripts"
$venvPython = "$venvScripts\python.exe"
$venvPip = "$venvScripts\pip.exe"
$venvPytest = "$venvScripts\pytest.exe"

Write-Host "`n[2/5] Upgrading pip..."
& $venvPython -m pip install --upgrade pip

# Install PyTorch with specific CUDA toolkit version (SpUNet requires cu124 matched with torch)
Write-Host "`n[3/5] Installing PyTorch with CUDA 12.4..."
& $venvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install everything including spunet dependencies
Write-Host "`n[4/5] Installing DBBD package with dev and spunet dependencies..."
& $venvPip install -e ".[dev,spunet]"

Write-Host "`n[5/5] Running Tests to confirm everything is green..."
Write-Host "============================================"
& $venvPytest tests/ -v

if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq $null) {
    Write-Host "============================================"
    Write-Host " Setup Complete! All tests passed and everything is GREEN. "
    Write-Host " "
    Write-Host " To activate the environment, run:"
    Write-Host " .\venv\Scripts\Activate.ps1"
    Write-Host "============================================"
} else {
    Write-Host "============================================"
    Write-Host " Tests failed with exit code $LASTEXITCODE. Please check the output above."
    Write-Host "============================================"
    exit 1
}
