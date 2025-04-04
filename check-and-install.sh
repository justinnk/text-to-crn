#!/bin/bash

# Function to check command existence
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if Python 3 is available
if command_exists python3; then
    echo "✅ Python3 is installed: $(python3 --version)"
else
    echo "❌ Python3 is not installed."
    exit 1
fi

# Get Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}')
# Extract major and minor version numbers
MAJOR_MINOR=$(echo "$PYTHON_VERSION" | awk -F. '{print $1"."$2}')
# Check if the version is 3.10.x or 3.11.x
if [[ "$MAJOR_MINOR" == "3.10" || "$MAJOR_MINOR" == "3.11" ]]; then
    echo "✅ Python version is $PYTHON_VERSION."
else
    echo "❌ Python version is $PYTHON_VERSION, but 3.10 or 3.11 are required."
    exit 1
fi

# Check for NVIDIA GPU
if command_exists nvidia-smi; then
    echo "✅ NVIDIA GPU detected."
else
    echo "❌ NVIDIA GPU not detected or NVIDIA drivers not installed."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
CUDA_REQUIRED="12.6"

if [[ $(echo "$CUDA_VERSION >= $CUDA_REQUIRED" | bc -l) -eq 1 ]]; then 
    echo "✅ CUDA version is $CUDA_VERSION (>= $CUDA_REQUIRED)"
else
    echo "❌ CUDA version is $CUDA_VERSION (< $CUDA_REQUIRED). This may still be fine."
    #exit 1
fi

# Check GPU VRAM
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -nr | head -n1)
VRAM_REQUIRED=23500

if [[ $(echo "$GPU_VRAM >= $VRAM_REQUIRED" | bc -l) -eq 1 ]]; then
    echo "✅ GPU VRAM is ${GPU_VRAM}MB (>= 23500)"
else
    echo "❌ GPU VRAM is ${GPU_VRAM}MB (< 23500)"
    exit 1
fi

if [[ $(df --output=avail / | tail -1) -gt $((600 * 1024 * 1024)) ]]; then
    echo "✅ More than 600GB of space available."
else
    echo "❌ You probably don't have enough space left on your device or the command we used to check does not work in your environment. Make sure you have sufficient space and comment out this check."
    exit 1
fi

echo "✅ All checks passed! Trying to install dependencies..."

if [ -d ".venv" ]; then
  echo "Warning: .venv already exists! This script should only be run once."
else
  python3 -m venv .venv                                    
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

if [ -d "loras/best_lora" ]; then
  echo "Warning: LoRA already exists! This script should only be run once."
else
  echo "Downloading LoRA (10GiB), this may take a while..."
  curl -LO https://zenodo.org/records/15145041/files/best_lora.zip
  mv best_lora.zip loras/
  cd loras/ && unzip best_lora.zip && rm best_lora.zip
fi

echo "✅ Dependencies installed successfully. You can now start with the reproduction."