#!/usr/bin/env bash
# Prepara o ambiente: venv + dependências + verificações. Corre a partir da raiz do repo.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

"$PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "precisa de Python 3.10+"'

if ! command -v ffmpeg >/dev/null; then
  echo "AVISO: ffmpeg não encontrado (necessário para mp4 H.264 e para o YouTube)."
  echo "  Ubuntu/Debian: sudo apt install ffmpeg   |   macOS: brew install ffmpeg"
fi

"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

# GPU NVIDIA: instala primeiro o PyTorch com CUDA (ver https://pytorch.org/get-started/locally/)
# ex.: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-dev.txt
pip install -e . --no-deps

python - <<'EOF'
import torch, cv2
print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__, "| CUDA disponível:", torch.cuda.is_available())
EOF

pytest -q
echo
echo "Pronto. Experimenta:  source .venv/bin/activate && ./scripts/demo.sh"
