#!/usr/bin/env bash
# Set up the virtual environment, dependencies and checks. Run from the repository root.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10 or newer is required"'

if ! command -v ffmpeg >/dev/null; then
  echo "WARNING: ffmpeg not found (required for H.264 MP4 and YouTube output)."
  echo "  Ubuntu/Debian: sudo apt install ffmpeg   |   macOS: brew install ffmpeg"
fi

"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

# NVIDIA GPU: install the CUDA-enabled PyTorch build first (see https://pytorch.org/get-started/locally/).
# Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-dev.txt
pip install -e . --no-deps

python - <<'EOF'
import torch, cv2
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
EOF

pytest -q
echo
echo "Ready. Try:  source .venv/bin/activate && ./scripts/demo.sh"
