"""Pipeline configuration with practical default values."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FramingConfig:
    # Output video resolution.
    out_width: int = 1920
    out_height: int = 1080

    # Crop width as a fraction of the source frame width.
    crop_width_ball: float = 0.30  # Tracked ball: tighter zoom.
    crop_width_wide: float = 0.50  # Lost ball: wider shot.
    min_crop_px: int = 960  # Avoid upscaling from a smaller crop.

    # Camera smoothing response time in seconds.
    smooth_time_pan: float = 0.55
    smooth_time_zoom: float = 1.2
    max_pan_speed: float = 0.30  # Source-frame widths per second.

    dead_zone: float = 0.03  # Fraction of crop width where the camera stays still.
    ball_weight: float = 0.70  # Blend between ball position and player centroid.
    lead_seconds: float = 0.25  # Look-ahead for ball motion.


@dataclass
class TrackerConfig:
    gate: float = 0.12  # Association radius as a fraction of frame width.
    max_coast_s: float = 1.2  # Predict the ball for this long without a detection.
    min_start_conf: float = 0.15  # Minimum confidence to start a track.
    reacquire_conf: float = 0.50  # Confidence needed to jump to another candidate.
    vel_alpha: float = 0.5  # Smoothing factor for estimated velocity.


@dataclass
class PipelineConfig:
    source: str = ""
    dest: str = "output.mp4"  # MP4 path or rtmp://... URL.
    fps: float | None = None  # None uses the source FPS.
    bitrate: str | None = None  # Example: "6000k"; RTMP defaults to 6000k.
    preset: str = "veryfast"
    detect_every: int = 1  # Run detection once every N frames.
    max_frames: int | None = None
    realtime: bool = False  # Limit processing to source FPS for file-based stream tests.
    debug_video: str | None = None  # Diagnostic video with detections and crop overlay.
    roi: list[tuple[float, float]] | None = None  # Pitch polygon in normalized frame coordinates.
    static_after_s: float = 3.0  # Reject stationary ball-like detections after N seconds; 0 disables.
    person_every: int = 5  # Refresh player detections once every N detector calls.
    full_search_every_s: float = 2.0  # Full tile search interval while the ball is tracked.
    framing: FramingConfig = field(default_factory=FramingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
