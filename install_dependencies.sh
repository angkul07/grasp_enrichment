#!/bin/bash
# install.sh
# Fully automated environment setup for this project using uv.
# Creates Python 3.11 virtualenv, installs CUDA 12.1 PyTorch,
# installs legacy deps, HaMeR, and MoGe.

set -e

echo "Starting installation..."

# Ensure uv exists
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Recreate venv with Python 3.11 every run for consistency
echo "Recreating virtual environment with Python 3.11..."
rm -rf .venv
uv venv --python 3.11 .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading build tools..."
uv pip install pip setuptools wheel

echo "Installing base PyTorch (CUDA 12.1)..."
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "Installing generic dependencies..."
uv pip install loguru h5py opencv-python tqdm

echo "Installing legacy chumpy first..."
uv pip install chumpy --no-build-isolation

echo "Installing HaMeR..."
uv pip install -e ./third_party/hamer[all] --no-deps

echo "Installing MoGe-2 requirements..."
uv pip install -r ./third_party/moge/requirements.txt

echo "Installing MoGe-2 editable package..."
uv pip install -e ./third_party/moge

echo ""
echo "Installation complete."
echo "To activate later:"
echo "source .venv/bin/activate"
echo ""
echo "To run:"
echo "python run_stage1.py --input_dir basic_pick_place --output_dir output --limit 2"