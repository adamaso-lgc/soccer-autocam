"""CLI entry point: python -m autocam --source match.mp4 --output out.mp4."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import FramingConfig, PipelineConfig, TrackerConfig
from .detectors import DemoColorDetector, YoloDetector
from .filters import parse_roi
from .pipeline import run


def _size(s: str) -> tuple[int, int]:
    try:
        w, h = (int(part) for part in s.lower().split("x"))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("Size must use WIDTHxHEIGHT, for example 1920x1080") from exc
    if w <= 0 or h <= 0 or w % 2 or h % 2:
        raise argparse.ArgumentTypeError("Width and height must be positive even numbers for H.264")
    return w, h


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be at least 1")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be 0 or greater")
    return number


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than 0")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be 0 or greater")
    return number


def _fraction(value: str) -> float:
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("Value must be greater than 0 and at most 1")
    return number


def _bitrate(value: str) -> str:
    normalized = value.lower()
    if not normalized.endswith("k") or not normalized[:-1].isdigit() or int(normalized[:-1]) <= 0:
        raise argparse.ArgumentTypeError("Bitrate must be a positive value in kbit/s, for example 6000k")
    return normalized


def _confidence(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Confidence must be between 0 and 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autocam", description="Digital camera that follows the ball.")
    p.add_argument("--source", required=True, help="file, rtsp://... URL, or webcam index (0)")
    p.add_argument("--output", default="output.mp4", help=".mp4 file or rtmp://a.rtmp.youtube.com/live2/STREAM_KEY")
    p.add_argument("--detector", choices=["yolo", "demo"], default="yolo",
                   help="'demo' only works with synthetic video")
    p.add_argument("--model", default="yolov8m.pt", help="YOLO weights (COCO or soccer-finetuned)")
    p.add_argument("--imgsz", type=_positive_int, default=1280)
    p.add_argument("--device", default=None, help="cuda:0, cpu, mps, etc.")
    p.add_argument("--window-size", type=_nonnegative_int, default=640,
                   help="While tracking, search near the ball in a window of this size (px); 0 disables it")
    p.add_argument("--tile-size", type=_nonnegative_int, default=0, help="Tiled ball inference (e.g. 1280); 0 disables it")
    p.add_argument("--person-every", type=_positive_int, default=5, help="Refresh player detections once every N detector calls")
    p.add_argument("--full-search-every", type=_positive_float, default=2.0,
                   help="Seconds between full searches (tiles) while tracking the ball")
    p.add_argument("--ball-conf", type=_confidence, default=0.10)
    p.add_argument("--person-conf", type=_confidence, default=0.35)
    p.add_argument("--detect-every", type=_positive_int, default=1, help="Run detection once every N frames")
    p.add_argument("--out-size", type=_size, default=(1920, 1080), metavar="WxH")
    p.add_argument("--fps", type=_positive_float, default=None)
    p.add_argument("--bitrate", type=_bitrate, default=None, help="e.g. 6000k (RTMP defaults to 6000k)")
    p.add_argument("--audio-source", default=None,
                   help="Optional file or stream with audio to mux into the output (FFmpeg required)")
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--max-frames", type=_positive_int, default=None)
    p.add_argument("--realtime", action="store_true", help="Limit to source FPS (for testing RTMP with files)")
    p.add_argument("--debug-video", default=None, help="Write a diagnostic video with detections and crop overlay")
    p.add_argument("--roi", default=None,
                   help="Pitch polygon in normalized 0-1 coordinates: 'x,y;x,y;...' (see scripts/pick_roi.py)")
    p.add_argument("--static-after", type=_nonnegative_float, default=3.0,
                   help="Ignore stationary ball-like detections after N seconds (0 disables it)")
    p.add_argument("--reconnect-attempts", type=_nonnegative_int, default=5,
                   help="Reconnect attempts for a live source before stopping (0 disables retries)")
    p.add_argument("--reconnect-delay", type=_nonnegative_float, default=1.0,
                   help="Seconds between live source reconnect attempts")
    p.add_argument("--crop-ball", type=_fraction, default=FramingConfig.crop_width_ball)
    p.add_argument("--crop-wide", type=_fraction, default=FramingConfig.crop_width_wide)
    p.add_argument("--smooth-pan", type=_positive_float, default=FramingConfig.smooth_time_pan)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    source_path = Path(a.source).expanduser()
    output_is_url = a.output.lower().startswith(("rtmp://", "rtmps://"))
    output_path = Path(a.output).expanduser() if not output_is_url else None
    debug_path = Path(a.debug_video).expanduser() if a.debug_video else None
    if output_path is not None and debug_path is not None and output_path.resolve() == debug_path.resolve():
        raise SystemExit("Output and debug video paths must be different.")
    if source_path.exists() and not output_is_url:
        if source_path.resolve() == output_path.resolve():
            raise SystemExit("Output path must not overwrite the source video.")
        if debug_path and source_path.resolve() == debug_path.resolve():
            raise SystemExit("Debug video path must not overwrite the source video.")
    if a.crop_wide < a.crop_ball:
        raise SystemExit("--crop-wide must be greater than or equal to --crop-ball.")
        if a.debug_video and output_path.resolve() == Path(a.debug_video).expanduser().resolve():
            raise SystemExit("Output and debug video paths must be different.")
    framing = FramingConfig(out_width=a.out_size[0], out_height=a.out_size[1],
                            crop_width_ball=a.crop_ball, crop_width_wide=a.crop_wide,
                            smooth_time_pan=a.smooth_pan)
    cfg = PipelineConfig(source=a.source, dest=a.output, fps=a.fps, bitrate=a.bitrate,
                         audio_source=a.audio_source, preset=a.preset,
                         detect_every=a.detect_every, max_frames=a.max_frames, realtime=a.realtime,
                         debug_video=a.debug_video, framing=framing, tracker=TrackerConfig(),
                         roi=parse_roi(a.roi) if a.roi else None, static_after_s=a.static_after,
                         person_every=a.person_every, full_search_every_s=a.full_search_every,
                         reconnect_attempts=a.reconnect_attempts, reconnect_delay_s=a.reconnect_delay)
    if a.detector == "demo":
        detector = DemoColorDetector()
    else:
        detector = YoloDetector(model=a.model, imgsz=a.imgsz, ball_conf=a.ball_conf,
                                person_conf=a.person_conf, device=a.device, tile_size=a.tile_size,
                                window_size=a.window_size)
    res = run(cfg, detector)
    print(f"{res.frames} processed frames in {res.seconds:.1f}s ({res.fps:.1f} processing FPS); "
          f"{res.output_frames} output frames at {res.output_fps:.1f} FPS, "
          f"average detection {res.detect_ms:.0f} ms, source reconnects: {res.source_reconnects}")
    print(f"ball states: {res.status_counts}")
    n = res.window_calls + res.full_calls
    print(f"diagnostics: {n} detector calls ({res.window_calls} local, {res.full_calls} full); "
          f"raw ball detections: {res.ball_raw}, outside ROI: {res.ball_roi_rejected}, "
          f"stationary: {res.ball_static_rejected}; calls without accepted ball: {res.detect_no_ball}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
