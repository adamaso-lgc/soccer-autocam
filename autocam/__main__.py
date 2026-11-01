"""CLI entry point: python -m autocam --source match.mp4 --output out.mp4."""
from __future__ import annotations

import argparse
import sys

from .config import FramingConfig, PipelineConfig, TrackerConfig
from .detectors import DemoColorDetector, YoloDetector
from .filters import parse_roi
from .pipeline import run


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autocam", description="Digital camera that follows the ball.")
    p.add_argument("--source", required=True, help="file, rtsp://... URL, or webcam index (0)")
    p.add_argument("--output", default="output.mp4", help=".mp4 file or rtmp://a.rtmp.youtube.com/live2/STREAM_KEY")
    p.add_argument("--detector", choices=["yolo", "demo"], default="yolo",
                   help="'demo' only works with synthetic video")
    p.add_argument("--model", default="yolov8m.pt", help="YOLO weights (COCO or soccer-finetuned)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None, help="cuda:0, cpu, mps, etc.")
    p.add_argument("--window-size", type=int, default=640,
                   help="While tracking, search near the ball in a window of this size (px); 0 disables it")
    p.add_argument("--tile-size", type=int, default=0, help="Tiled ball inference (e.g. 1280); 0 disables it")
    p.add_argument("--person-every", type=int, default=5, help="Refresh player detections once every N detector calls")
    p.add_argument("--full-search-every", type=float, default=2.0,
                   help="Seconds between full searches (tiles) while tracking the ball")
    p.add_argument("--ball-conf", type=float, default=0.10)
    p.add_argument("--person-conf", type=float, default=0.35)
    p.add_argument("--detect-every", type=int, default=1, help="Run detection once every N frames")
    p.add_argument("--out-size", type=_size, default=(1920, 1080), metavar="WxH")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--bitrate", default=None, help="e.g. 6000k (RTMP defaults to 6000k)")
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--realtime", action="store_true", help="Limit to source FPS (for testing RTMP with files)")
    p.add_argument("--debug-video", default=None, help="Write a diagnostic video with detections and crop overlay")
    p.add_argument("--roi", default=None,
                   help="Pitch polygon in normalized 0-1 coordinates: 'x,y;x,y;...' (see scripts/pick_roi.py)")
    p.add_argument("--static-after", type=float, default=3.0,
                   help="Ignore stationary ball-like detections after N seconds (0 disables it)")
    p.add_argument("--crop-ball", type=float, default=FramingConfig.crop_width_ball)
    p.add_argument("--crop-wide", type=float, default=FramingConfig.crop_width_wide)
    p.add_argument("--smooth-pan", type=float, default=FramingConfig.smooth_time_pan)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    framing = FramingConfig(out_width=a.out_size[0], out_height=a.out_size[1],
                            crop_width_ball=a.crop_ball, crop_width_wide=a.crop_wide,
                            smooth_time_pan=a.smooth_pan)
    cfg = PipelineConfig(source=a.source, dest=a.output, fps=a.fps, bitrate=a.bitrate, preset=a.preset,
                         detect_every=a.detect_every, max_frames=a.max_frames, realtime=a.realtime,
                         debug_video=a.debug_video, framing=framing, tracker=TrackerConfig(),
                         roi=parse_roi(a.roi) if a.roi else None, static_after_s=a.static_after,
                         person_every=a.person_every, full_search_every_s=a.full_search_every)
    if a.detector == "demo":
        detector = DemoColorDetector()
    else:
        detector = YoloDetector(model=a.model, imgsz=a.imgsz, ball_conf=a.ball_conf,
                                person_conf=a.person_conf, device=a.device, tile_size=a.tile_size,
                                window_size=a.window_size)
    res = run(cfg, detector)
    print(f"{res.frames} frames in {res.seconds:.1f}s ({res.fps:.1f} FPS), "
          f"average detection {res.detect_ms:.0f} ms, ball states: {res.status_counts}")
    n = res.window_calls + res.full_calls
    print(f"diagnostics: {n} detector calls ({res.window_calls} local, {res.full_calls} full); "
          f"raw ball detections: {res.ball_raw}, outside ROI: {res.ball_roi_rejected}, "
          f"stationary: {res.ball_static_rejected}; calls without accepted ball: {res.detect_no_ball}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
