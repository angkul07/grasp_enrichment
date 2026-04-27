#!/bin/bash
# install_dependencies.sh
# Fully automated environment setup for this project using uv.
# Creates Python 3.10 virtualenv (matching HaMeR's requirement),
# installs CUDA 12.1 PyTorch, HaMeR + ViTPose + MoGe-2, and all deps.

set -e

echo "Starting installation..."

# Ensure uv exists
if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not installed."
    echo "Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# ── 1. Virtual environment ────────────────────────────────────────────────
echo "Recreating virtual environment with Python 3.10..."
rm -rf .venv
uv venv --python 3.10 .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading build tools..."
uv pip install pip setuptools wheel

# ── 2. PyTorch (CUDA 12.1) ───────────────────────────────────────────────
echo "Installing PyTorch (CUDA 13.0)..."
uv pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu130

# ── 3. Generic project dependencies ─────────────────────────────────────
echo "Installing generic dependencies..."
uv pip install loguru h5py opencv-python tqdm matplotlib trimesh pyglet

# ── 4. HaMeR + its dependency chain ─────────────────────────────────────
echo "Installing legacy chumpy (requires --no-build-isolation)..."
uv pip install chumpy --no-build-isolation

echo "Installing HaMeR core dependencies..."
uv pip install \
  gdown yacs pyrender pytorch-lightning scikit-image \
  "smplx==0.1.28" timm einops xtcocotools pandas

echo "Installing HaMeR as editable package..."
uv pip install -e ./third_party/hamer --no-deps

echo "Installing HaMeR extras (hydra, etc.)..."
uv pip install hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils rich webdataset

# ── 5. Detectron2 (from source — required by HaMeR) ─────────────────────
echo "Installing Detectron2 from GitHub..."
uv pip install 'git+https://github.com/facebookresearch/detectron2.git'

# ── 6. ViTPose submodule + mmpose stack (hand keypoint detection) ────────
echo "Initializing ViTPose submodule..."
cd third_party/hamer
git submodule update --init --recursive
cd ../..

echo "Installing mmcv and mmpose..."
uv pip install "mmcv==1.3.9" --no-build-isolation
uv pip install -v -e ./third_party/hamer/third-party/ViTPose

# ── 7. MoGe-2 ───────────────────────────────────────────────────────────
echo "Installing MoGe-2 requirements..."
uv pip install -r ./third_party/moge/requirements.txt

echo "Installing MoGe-2 as editable package..."
uv pip install -e ./third_party/moge

# ── 8. Fetch HaMeR demo data (checkpoint + MANO mean params) ────────────
echo "Fetching HaMeR model checkpoint..."
cd third_party/hamer
bash fetch_demo_data.sh
cd ../..

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║    Installation complete!                ║"
echo "╠══════════════════════════════════════════╣"
echo "║  Activate:  source .venv/bin/activate    ║"
echo "║                                          ║"
echo "║  IMPORTANT: You must manually place      ║"
echo "║  MANO_RIGHT.pkl into:                    ║"
echo "║  third_party/hamer/_DATA/data/mano/      ║"
echo "║  (Download from https://mano.is.tue.mpg.de) ║"
echo "║                                          ║"
echo "║  Run:                                    ║"
echo "║  python run_stage1.py \\                  ║"
echo "║    --input_dir basic_pick_place \\         ║"
echo "║    --output_dir output --limit 1          ║"
echo "╚══════════════════════════════════════════╝"