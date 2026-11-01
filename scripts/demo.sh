#!/usr/bin/env bash
# Demo without a camera or YOLO: generate a synthetic match and run the "demo" detector.
set -euo pipefail
mkdir -p outputs

python -m autocam.synthetic outputs/demo.mp4 --seconds 15
python -m autocam --source outputs/demo.mp4 --output outputs/demo_out.mp4 \
  --detector demo --debug-video outputs/demo_debug.mp4

echo
echo "Output:     outputs/demo_out.mp4   (virtual-camera feed)"
echo "Debug view: outputs/demo_debug.mp4 (green = crop, yellow = ball, blue = players)"
