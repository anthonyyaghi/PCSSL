#!/bin/bash
set -e

echo "============================================"
echo " DBBD Setup & Installation Script (RunPod)"
echo "============================================"

# Use the native 'python3' (or 'python' if mapped) on the RunPod instance
PYTHON_CMD="python"
if ! command -v $PYTHON_CMD &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python native installation not found!"
        exit 1
    fi
fi

echo -e "\n[1/4] Upgrading pip natively..."
$PYTHON_CMD -m pip install --upgrade pip

echo -e "\n[2/4] Installing PyTorch with CUDA 12.4..."
# RunPods often come with PyTorch, but this ensures the required cu124 bindings for SpUNet
$PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo -e "\n[3/4] Installing DBBD package with dev and spunet dependencies..."
$PYTHON_CMD -m pip install -e ".[dev,spunet]"

echo -e "\n[4/4] Running Tests to confirm everything is green..."
echo "============================================"
# Run pytest directly through the native python module
$PYTHON_CMD -m pytest tests/ -v

echo "============================================"
echo " Setup Complete! All tests passed and everything is GREEN. "
echo "============================================"
