"""Main source -> detection -> tracking -> virtual camera -> output loop."""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PipelineConfig
from .detectors import Detector, Detection
from .filters import RoiFilter, StaticFilter
from .framing import BallTracker, CropRect, VirtualCamera, render_crop
from .io import RealtimeSink, Sink, VideoSource, make_sink


@dataclass
class RunResult:
    frames: int = 0
    seconds: float = 0.0
    detect_ms: float = 0.0  # Average time per detector call.
    status_counts: dict = field(default_factory=dict)
    log: list = field(default_factory=list)  # Per-frame log when collect=True.
    # Diagnostic counters (counts per detector call, not per frame).
    window_calls: int = 0  # Fast local-window searches.
    full_calls: int = 0  # Full-frame searches.
    ball_raw: int = 0  # Raw ball detections returned by the model.
    ball_roi_rejected: int = 0  # Ball detections rejected outside the ROI.
    ball_static_rejected: int = 0  # Ball detections rejected as stationary.
    detect_no_ball: int = 0  # Calls with no accepted ball detection.
    output_frames: int = 0
    output_fps: float = 0.0
    source_reconnects: int = 0

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds > 0 else 0.0


def _draw_debug(frame, size, persons, balls, ball_pos, rect: CropRect, status: str,
                roi_poly=None, rejected=()):
    """Draw the overlay on a frame resized exactly to ``size`` = (width, height)."""
    dw, dh = size
    sx, sy = dw / frame.shape[1], dh / frame.shape[0]
    img = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    if roi_poly is not None:
        pts = (roi_poly * np.array([sx, sy], dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, (255, 0, 255), 1)
    for d in rejected:  # Filtered detections are shown as small red circles.
        cx, cy = d.center
        cv2.circle(img, (int(cx * sx), int(cy * sy)), 5, (0, 0, 255), 1)
    for d in persons:
        cv2.rectangle(img, (int(d.x1 * sx), int(d.y1 * sy)), (int(d.x2 * sx), int(d.y2 * sy)), (255, 200, 0), 1)
    for d in balls:
        cx, cy = d.center
        cv2.circle(img, (int(cx * sx), int(cy * sy)), 8, (0, 255, 255), 2)
    if ball_pos is not None:
        cv2.drawMarker(img, (int(ball_pos[0] * sx), int(ball_pos[1] * sy)), (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.rectangle(img, (int(rect.x0 * sx), int(rect.y0 * sy)),
                  (int((rect.x0 + rect.w) * sx), int((rect.y0 + rect.h) * sy)), (0, 255, 0), 2)
    cv2.putText(img, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def run(cfg: PipelineConfig, detector: Detector, collect: bool = False) -> RunResult:
    src = VideoSource(cfg.source, cfg.reconnect_attempts, cfg.reconnect_delay_s, cfg.realtime)
    fps = cfg.fps or src.fps or 25.0
    dt = 1.0 / fps
    fc = cfg.framing
    W, H = src.width, src.height
    if W <= 0 or H <= 0:
        src.close()
        raise RuntimeError("Could not determine the source resolution.")

    tracker = BallTracker(W, H, cfg.tracker)
    camera = VirtualCamera(W, H, fc, cfg.tracker)
    sink: Sink | None = None
    dbg: Sink | None = None
    realtime_sink: RealtimeSink | None = None
    dbg = None
    dbg_size = (0, 0)
    roi = RoiFilter(cfg.roi, W, H) if cfg.roi else None
    static = StaticFilter(W, cfg.static_after_s) if cfg.static_after_s > 0 else None

    result = RunResult()
    counts: Counter = Counter()
    players: np.ndarray | None = None
    persons: list[Detection] = []
    balls: list[Detection] = []
    rejected: list[Detection] = []
    last_ball = None
    det_time, det_calls = 0.0, 0
    t_start = time.perf_counter()
    next_t = t_start
    media_time = 0.0
    previous_source_time: float | None = None
    last_full_search_time = float("-inf")

    try:
        sink = make_sink(cfg.dest, fc.out_width, fc.out_height, fps, cfg.bitrate,
                         cfg.preset, cfg.audio_source)
        if cfg.debug_video:
            k = min(1.0, 1280 / W)
            dbg_size = (max(2, int(W * k) // 2 * 2), max(2, int(H * k) // 2 * 2))  # H.264 requires even dimensions.
            dbg = make_sink(cfg.debug_video, dbg_size[0], dbg_size[1], fps)
        if src.paced:
            realtime_sink = RealtimeSink(sink, fps, dbg)

        idx = 0
        while cfg.max_frames is None or idx < cfg.max_frames:
            frame = src.read()
            if frame is None:
                break

            step_dt = dt
            if src.paced and src.last_frame_time is not None:
                if previous_source_time is not None:
                    step_dt = max(1e-3, src.last_frame_time - previous_source_time)
                previous_source_time = src.last_frame_time
            media_time += step_dt

            detected = idx % max(1, cfg.detect_every) == 0
            if detected:
                # Search near the tracked ball; periodically perform a full search.
                hint = None
                if (last_ball is not None and last_ball.status == "tracked" and last_ball.pos is not None
                        and media_time - last_full_search_time < cfg.full_search_every_s):
                    hint = (last_ball.pos[0] + last_ball.vel[0] * step_dt,
                            last_ball.pos[1] + last_ball.vel[1] * step_dt)
                if hint is None:
                    last_full_search_time = media_time
                want_persons = det_calls % max(1, cfg.person_every) == 0 or players is None

                t0 = time.perf_counter()
                dets = detector.detect(frame, hint=hint, persons=want_persons)
                det_time += time.perf_counter() - t0
                det_calls += 1
                result.window_calls += hint is not None
                result.full_calls += hint is None
                result.ball_raw += sum(1 for d in dets if d.label == "ball")
                rejected = []
                if roi is not None:
                    dets, out = roi.split(dets)
                    rejected += out
                    result.ball_roi_rejected += sum(1 for d in out if d.label == "ball")
                balls = [d for d in dets if d.label == "ball"]
                if static is not None:
                    balls, still = static.split(balls, media_time)
                    rejected += still
                    result.ball_static_rejected += len(still)
                result.detect_no_ball += not balls
                if want_persons:  # Otherwise retain players from the last refresh.
                    persons = [d for d in dets if d.label == "person"]
                    players = np.array([d.foot for d in persons], dtype=np.float32) if persons else None

            ball = tracker.update(balls if detected else [], step_dt, detected)
            last_ball = ball
            rect = camera.update(ball, players, step_dt)
            output_frame = render_crop(frame, rect, fc.out_width, fc.out_height)
            debug_frame = None
            if dbg is not None:
                debug_frame = _draw_debug(frame, dbg_size, persons, balls if detected else [], ball.pos, rect,
                                          ball.status, roi.poly if roi else None, rejected if detected else [])
            if realtime_sink is not None:
                realtime_sink.update(output_frame, debug_frame)
            else:
                sink.write(output_frame)
                result.output_frames += 1
                if dbg is not None and debug_frame is not None:
                    dbg.write(debug_frame)

            counts[ball.status] += 1
            if collect:
                result.log.append({"frame": idx, "cx": rect.cx, "cy": rect.cy, "w": rect.w, "h": rect.h,
                                   "ball": ball.pos, "status": ball.status})
            idx += 1

    finally:
        src.close()
        if realtime_sink is not None:
            realtime_sink.close()
            result.output_frames = realtime_sink.frames_written
        else:
            if dbg is not None:
                dbg.close()
            if sink is not None:
                sink.close()

    result.frames = idx
    result.source_reconnects = src.reconnect_count
    result.seconds = time.perf_counter() - t_start
    result.output_fps = realtime_sink.measured_fps if realtime_sink is not None else fps
    result.detect_ms = 1000 * det_time / det_calls if det_calls else 0.0
    result.status_counts = dict(counts)
    return result
