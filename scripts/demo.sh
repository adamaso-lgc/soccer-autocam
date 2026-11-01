#!/usr/bin/env bash
# Demo sem câmara nem YOLO: gera um jogo sintético e corre o pipeline com o detetor "demo".
set -euo pipefail
mkdir -p outputs

python -m autocam.synthetic outputs/demo.mp4 --seconds 15
python -m autocam --source outputs/demo.mp4 --output outputs/demo_out.mp4 \
  --detector demo --debug-video outputs/demo_debug.mp4

echo
echo "Resultado:  outputs/demo_out.mp4   (o que iria para o YouTube)"
echo "Overlay:    outputs/demo_debug.mp4 (verde = crop, amarelo = bola, azul = jogadores)"
