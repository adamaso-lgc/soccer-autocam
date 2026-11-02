# Soccer AutoCam

Software PTZ for a fixed, wide-angle soccer camera. The pipeline detects the ball and players, tracks the ball, moves a virtual camera smoothly, crops to a 16:9 output, and writes an MP4 or streams to YouTube over RTMP.

```text
video source -> detector -> ball tracker -> virtual camera -> crop/resize -> H.264 output
```

The intended input is a high, fixed camera that sees most or all of the pitch. This project is an early prototype: the tracking and framing pipeline has automated coverage, but the YOLO settings have not yet been validated against a representative set of real matches. Treat real-time and broadcast quality as goals to measure, not established capabilities.

## Project status

| Area | Status |
|---|---|
| Tracking, virtual camera, MP4/RTMP output, CLI | Implemented; covered by unit and synthetic end-to-end tests |
| YOLO full-frame, tile and local-window inference | Implemented; tested with a fake model, not yet benchmarked on real match footage |
| Match-footage evaluation set and quality metrics | Needed |
| Fine-tuned soccer ball detector | Future work; requires dataset/license review |
| Multi-camera stitching and physical PTZ | Out of current scope |

The `demo` detector is only for the generated synthetic video. It is not a soccer detector.

Live and `--realtime` inputs use a bounded latest-frame buffer and fixed-rate output. If inference is slower than the requested output FPS, the last crop is repeated. This preserves stream timing but cannot make slow inference look smooth; the deployment computer must process new frames quickly enough for acceptable camera movement.

## Requirements

- Python 3.10 or newer
- FFmpeg for H.264 MP4 output and RTMP streaming
- PyTorch/Ultralytics supported device for YOLO inference (CUDA, Apple MPS, or CPU)
- Fixed, high-resolution source recommended; higher input resolution helps preserve small ball detail

## Setup

```bash
./scripts/setup.sh
source .venv/bin/activate
```

For NVIDIA, install the matching CUDA build of PyTorch before installing the project dependencies; see the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). Keep model weights and stream keys out of version control.

## Synthetic demo

```bash
./scripts/demo.sh
```

This creates a synthetic match, including brief ball occlusions, then writes:

- `outputs/demo_out.mp4`: virtual-camera output
- `outputs/demo_debug.mp4`: input view with the crop, detections and tracking state

This checks software wiring and framing behavior. It does not measure detector quality on real matches.

## Process a real video

```bash
python -m autocam --source data/clip.mp4 --output outputs/clip_out.mp4 \
  --debug-video outputs/clip_debug.mp4 --device mps \
  --tile-size 1280 --ball-conf 0.05 --detect-every 2 --max-frames 300
```

The default `yolov8m.pt` uses the COCO `sports ball` class as a baseline. Small balls can be missed even at high source resolution. Tiles increase effective detail at additional compute cost. Lower confidence thresholds may recover missed balls but also add false positives; ROI filtering and temporal tracking can reject some of them.

For a field polygon, select its boundary in a representative frame:

```bash
python scripts/pick_roi.py data/clip.mp4 --time 5
```

Pass the emitted `--roi` string to `python -m autocam`. The ROI coordinates are normalized to the full source frame. Review the debug video to confirm the polygon follows the playable area and that useful detections remain inside it.

## Camera and YouTube streaming

RTSP input:

```bash
python -m autocam --source "rtsp://user:password@192.168.1.50:554/stream1" \
  --output outputs/live.mp4 --debug-video outputs/live_debug.mp4
```

YouTube RTMP output (use a stream key from YouTube Studio; do not commit it):

```bash
python -m autocam --source "rtsp://camera/stream" \
  --output "rtmp://a.rtmp.youtube.com/live2/STREAM_KEY" --bitrate 6000k \
  --audio-source "rtsp://camera/audio-stream"
```

For a file-based streaming rehearsal, add `--realtime`; slower inference drops old input frames to keep the output timeline aligned with file playback time. Without `--audio-source`, RTMP output contains silent AAC audio. Pass an audio-bearing file or stream to `--audio-source` to include AAC audio. This opens a separate audio connection, so check that the camera/server permits it and verify audio/video sync during rehearsal. Check upload capacity and the stream dashboard before a match.

Live sources are retried up to five times by default, with a one-second delay. Change this with `--reconnect-attempts` and `--reconnect-delay`. If the stream cannot be reopened, the process exits with an error instead of quietly producing a truncated program feed. A resolution change after reconnect requires restarting the process.

The CLI reports processing FPS separately from output FPS. A live run can maintain a 25 FPS encoded feed by repeating frames while inference is slower, but the camera view will only update at the measured processing rate. Confirm both numbers on the actual deployment computer before streaming a match.

## Useful tuning options

| Option | Purpose |
|---|---|
| `--tile-size` | Run ball inference over overlapping image tiles |
| `--window-size` | Search near the predicted ball location while tracking |
| `--full-search-every` | Periodic full search while the ball is tracked |
| `--detect-every` | Run detection once every N source frames |
| `--ball-conf` | Ball confidence threshold |
| `--roi` | Ignore detections outside the pitch polygon |
| `--static-after` | Reject stationary ball-like detections after a time interval |
| `--crop-ball`, `--crop-wide` | Crop width when the ball is tracked or unavailable |
| `--smooth-pan` | Camera pan response time |
| `--debug-video` | Write a diagnostic overlay video |
| `--audio-source` | Optional audio input for FFmpeg muxing |
| `--reconnect-attempts` | Bounded retry count for live source recovery |

See `python -m autocam --help` for all options. Core defaults are in `autocam/config.py`.

## Repository layout

```text
autocam/       detector, filters, tracking, framing, I/O and pipeline
scripts/       setup, demo and ROI selection helpers
tests/         unit and synthetic end-to-end tests
data/          local test footage (not necessarily tracked by Git)
outputs/       generated videos and diagnostics
```

## Development plan

The work should proceed in evidence-driven stages. See [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for the detailed workflow and [docs/DEPLOYMENT_CHECKLIST.md](docs/DEPLOYMENT_CHECKLIST.md) for the release gate before live use.

1. Establish repeatable baselines on the supplied clip and a longer representative match.
2. Measure ball detection and framing quality, not just processing FPS.
3. Fix detector/tracker failure modes against annotated failure examples.
4. Benchmark target hardware and only then choose model/export optimizations.
5. Validate RTSP recovery, end-to-end latency and an unlisted YouTube stream before a public match.
