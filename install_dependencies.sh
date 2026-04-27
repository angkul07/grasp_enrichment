#!/bin/bash
# install_dependencies.sh
# Sets up the virtual environment and required libraries using uv.
# We expect virtual environment to exist at .venv.

set -e

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing base PyTorch (CUDA 12.1)..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing generic dependencies (Loguru, OpenCV, h5py, tqdm)..."
uv pip install loguru h5py opencv-python tqdm

echo "Installing HaMeR dependencies..."
uv pip install -e ./third_party/hamer[all]

echo "Installing MoGe-2 dependencies..."
uv pip install -r ./third_party/moge/requirements.txt
uv pip install -e ./third_party/moge

echo "All dependencies installed successfully! Environment is ready."
