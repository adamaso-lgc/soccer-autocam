# Development Plan and Handoff

## Goal

Build a reliable soccer auto-camera for a fixed, high-resolution camera. The software should produce a well-framed 16:9 program feed and, after validation, stream it live to YouTube. A successful system must keep the ball and relevant play visible, move naturally, recover after missed detections, and stay within the latency and compute limits of the deployment machine.

## Current baseline

- Python package and CLI: `autocam/`
- Detector: Ultralytics YOLO, with full-frame inference, overlapping tiles, and a local search window around the predicted ball position
- Tracking/framing: confidence-based ball association, coast prediction, player-centroid fallback, smoothed digital pan and zoom
- I/O: OpenCV capture and FFmpeg H.264/RTMP output
- Test assets mentioned by the project owner: `data/clip.mp4` and a longer source video under `data/`
- Existing check: 22 tests passed with `pytest -q` before this handoff
- Latest recorded sample run: 300 frames at 7.0 FPS on Apple MPS; 150 detector calls, 468 raw ball detections, 343 rejected outside the ROI, 53 calls with no accepted ball. The figures describe one run and do not establish accuracy.
- Current environment: the installed PyTorch reports no MPS or CUDA device. A 60-frame CPU run took 15.5 seconds (3.9 processed FPS; 437 ms average detection), with 69 raw ball detections, 29 outside the ROI and one detector call with no accepted ball. The MP4 was valid H.264 at 1920x1080/25 FPS; this is offline throughput, not live suitability.
- Paced YOLO replay of the 90-second clip on the same CPU ended after 90.5 seconds with 56 frames processed (0.6 processing FPS; 3,159 ms average detection), 46 frames in `lost` state, and 2,150 output frames at 24.9 FPS. The output cadence worked, but tracking quality/refresh rate failed the proposed live gate.
- Live/output handling now retries a disconnected source a bounded number of times and repeats the newest crop at the configured output cadence. This preserves output timing but does not increase the camera update rate.
- The longer source contains AAC audio. RTMP output remains silent unless `--audio-source` is supplied; explicit audio muxing and synchronization still need testing.

The sample run used a hand-selected ROI. Its geometry, detector output and final crop must be inspected together before tuning thresholds. In particular, many raw detections outside the ROI may be expected, or may indicate that a class/model setting is too permissive.

## Operating principles

1. Keep source video, trained weights, generated media and stream keys out of Git unless a deliberate data policy says otherwise.
2. Preserve a reproducible command, machine/device details, model name/hash, source resolution/FPS, ROI, and output files for every benchmark.
3. Evaluate on complete sequences and varied play phases. A short clip is useful for iteration but can hide reacquisition, boundary and camera-motion problems.
4. Separate detector quality, tracking quality, framing quality, throughput and end-to-end latency. FPS alone is not a quality measure.
5. Use a small annotated sample first. Review licensing and intended use before adopting an external dataset or weights.
6. Keep changes narrow and add focused regression coverage when changing behavior. Do not claim real-match quality until measured on held-out footage.

## Milestones

### 1. Reproduce and inspect the baseline

- Record `git status`, Python/PyTorch/Ultralytics/OpenCV versions, device, input metadata and model weights.
- Re-run the existing test suite and the exact short-clip command from the project owner.
- Inspect the source, ROI overlay, debug video and final output at normal speed and frame by frame around missed detections and fast passes.
- Save a short run report with the command, counters, processing FPS and notable timestamps.

**Exit condition:** another developer can reproduce the same run and locate the key failure frames.

### 2. Build an evaluation set

- Select representative segments: wide play, transitions, near-touchline play, set pieces, occlusion, ball near image boundaries, and stationary ball.
- Annotate ball center/box and optionally player locations at a measured sampling rate. Document coordinate convention and uncertain/occluded labels.
- Keep tuning and held-out clips separate.
- Track ball detection precision/recall at a stated matching radius, track continuity/reacquisition, ball-in-output-frame ratio, crop motion/smoothness, processing FPS and latency.

**Exit condition:** a repeatable evaluation command/report compares model or configuration changes on the same clips.

### 3. Improve detection and tracking

- Categorize failures from debug footage: missed ball, false ball, duplicate tile detections, ROI rejection, local-window miss, stale player detections, and identity jump.
- Check whether a local window is large enough for ball motion between detection frames; tune its size and periodic full-search interval from measured fast-pass cases.
- Compare COCO baseline with soccer-specific weights only after checking dataset annotation quality, class mapping, licenses and held-out performance.
- Review stationary filtering against legitimate long set pieces so that a stationary ball is not suppressed in a way that harms coverage.
- Tune confidence, ROI and tracker gates against both false positives and missed detections.

**Exit condition:** documented improvement on held-out clips with no unacceptable regression in framing or throughput.

### 4. Improve production readiness

- Profile inference, crop rendering and FFmpeg separately on the intended machine.
- Decide the acceptable source/output resolution, frame rate and bitrate from hardware and network measurements.
- Exercise RTSP disconnect/reconnect, stalled input, FFmpeg failure, graceful shutdown, and stream-key handling.
- Measure capture-to-output latency and camera update rate under sustained load; verify repeated frames do not create unacceptable stalls.
- Test audio muxing and sync with the actual camera or mixer. Default RTMP audio is silent.
- Run an unlisted live test and review both the received stream and local diagnostics.

**Exit condition:** a documented hardware profile sustains the target stream and recovers from expected failures.

### 5. Optional expansions

Consider TensorRT/ONNX, multi-camera stitching, or physical PTZ only when evaluation shows a concrete need and a target deployment is defined. These add calibration, compatibility and operational complexity.

## Handoff checklist for an AI coding agent

At the start of a task:

1. Read this file, `README.md`, and applicable `AGENTS.md` instructions.
2. Inspect `git status` and preserve unrelated user changes.
3. Identify the affected module and current tests before editing.
4. State assumptions in the task summary and keep private video/credentials out of logs and commits.

Before handing work back:

1. Explain what changed and why, with file links and any known limitations.
2. Run only the checks requested by the user or task instructions; report exact results.
3. For algorithm changes, include before/after evaluation evidence or say what remains unmeasured.
4. Do not call the pipeline production-ready based only on synthetic tests.

## Suggested next task

The next release gate is deployment validation: select the target computer and camera, measure the camera update rate on the full panorama, and verify audio/video sync with an unlisted stream. The current CPU environment is far below the 25 FPS input rate and cannot qualify as the live deployment host.
