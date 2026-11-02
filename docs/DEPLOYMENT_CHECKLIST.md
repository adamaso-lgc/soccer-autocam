# Deployment Checklist

This checklist is the release gate for using Soccer AutoCam on a real match. Passing the unit tests or producing an MP4 does not qualify a computer for live use.

## 1. Prepare the deployment computer

- Install the project and FFmpeg using `README.md`.
- Confirm the selected PyTorch device is available. Run a short detector benchmark on the exact model, source resolution and options intended for the match.
- Ensure the camera can provide a stable fixed wide shot at the expected resolution and frame rate.
- Confirm the network has stable upload headroom above the selected video bitrate plus audio and protocol overhead.
- Keep RTSP credentials and the YouTube stream key in a local, access-restricted environment file. Do not commit it or paste credentials into shared logs.

## 2. Validate framing and tracking offline

- Select the pitch ROI from the deployment camera view, not from a different camera position.
- Process a representative segment containing fast passes, play near each touchline, set pieces, occlusion, and a stationary ball.
- Review the source and debug output side by side. Confirm the ball and relevant play remain inside the green crop and that the crop does not oscillate or jump.
- Record the exact model/weight hash, CLI command, source resolution/FPS, ROI, detector counters, processing FPS, and output video.
- Review false positives outside and inside the ROI. A high ball counter does not mean the ball was correctly detected.

## 3. Proposed performance and quality gates

Agree deployment-specific thresholds before a match. Suggested initial gates for a 25 FPS source:

- At least 12.5 processed frames per second when using `--detect-every 2`, with no growing capture-to-output delay.
- Output cadence remains close to the configured 25 FPS, and repeated-frame holds are brief enough that camera movement looks acceptable.
- On an annotated held-out sample, the ball remains inside the program crop for at least 95% of visible-ball frames; manually inspect all long losses and boundary cases.
- No sustained `lost` state during normal play; recovery from occlusion or a missed detection happens within the agreed recovery window.
- Audio/video sync remains within 500 ms over a sustained rehearsal.

These are proposed engineering gates, not claims about current performance. Adjust them to the production format and validate them with the broadcaster/operator.

## 4. Run an unlisted stream rehearsal

- Start an unlisted YouTube stream, never the public match stream as the first live test.
- Use the same camera, audio route, computer, bitrate, network and detector settings planned for production.
- Check the received stream from a separate device for picture, framing, motion cadence, audio, sync, bitrate and dropped frames.
- Interrupt and restore the camera/network once. Confirm the reconnect message, recovery behavior and output stream state.
- Keep a local debug recording and note timestamps for any lost tracking or stream interruption.
- Stop the rehearsal and review the recording before enabling a public stream.

## 5. Example command

Use a local shell environment file that is excluded from Git to set `YOUTUBE_STREAM_KEY`, camera credentials and `PITCH_ROI`. Restrict its file permissions and load it with `set -a; source .env; set +a` in the shell before running the command. This example assumes the RTSP endpoint carries audio and can accept a second connection; otherwise provide the audio mixer/file endpoint separately.

```bash
python -m autocam \
  --source "$CAMERA_RTSP_URL" \
  --output "rtmp://a.rtmp.youtube.com/live2/$YOUTUBE_STREAM_KEY" \
  --audio-source "$CAMERA_AUDIO_URL" \
  --device cuda:0 \
  --tile-size 1280 --ball-conf 0.05 --detect-every 2 \
  --roi "$PITCH_ROI" \
  --debug-video outputs/live_debug.mp4
```

Replace `cuda:0` with a verified accelerator for the deployment computer. The `--audio-source` input is opened separately from the video capture, so confirm camera connection limits and synchronization. If audio is unavailable, RTMP still gets a silent AAC track.

## 6. Stop conditions

Do not start a public match stream if any of these apply:

- The full-resolution rehearsal does not meet the agreed tracking/update-rate threshold.
- The camera ROI or crop loses play at field boundaries.
- The stream cannot recover from the planned network/camera interruption.
- Audio is missing, badly delayed, or out of sync when commentary/crowd audio is required.
- The YouTube receiving device shows persistent dropped frames, unstable bitrate, or excessive delay.

If a stop condition occurs, retain the run report/debug video and fix the highest-impact measured failure before the next rehearsal.
