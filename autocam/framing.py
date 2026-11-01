"""Ball tracking and smoothly moving digital PTZ camera."""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .config import FramingConfig, TrackerConfig
from .detectors import Detection


# ------------------------------------------------------------- Ball tracker
@dataclass
class BallState:
    pos: tuple[float, float] | None
    vel: tuple[float, float]  # Pixels per second.
    status: str  # "tracked", "coasting", or "lost".
    coast_time: float  # Seconds since the last accepted detection.


class BallTracker:
    """Associate ball detections over time and predict through short losses."""

    def __init__(self, frame_w: int, frame_h: int, cfg: TrackerConfig | None = None) -> None:
        self.W, self.H = frame_w, frame_h
        self.cfg = cfg or TrackerConfig()
        self._pos: tuple[float, float] | None = None
        self._last_det: tuple[float, float] | None = None
        self._vel = (0.0, 0.0)
        self._coast = 0.0

    def _state(self) -> BallState:
        if self._pos is None:
            return BallState(None, (0.0, 0.0), "lost", self._coast)
        status = "tracked" if self._coast <= 0.3 else "coasting"
        return BallState(self._pos, self._vel, status, self._coast)

    def update(self, balls: list[Detection], dt: float, detected: bool = True) -> BallState:
        cfg = self.cfg
        pred = None
        if self._pos is not None:
            pred = (self._pos[0] + self._vel[0] * dt, self._pos[1] + self._vel[1] * dt)
            self._coast += dt

        chosen: Detection | None = None
        if detected and balls:
            if self._pos is None:
                best = max(balls, key=lambda d: d.conf)
                if best.conf >= cfg.min_start_conf:
                    chosen = best
            else:
                gate = cfg.gate * self.W * (1 + 2 * self._coast)
                near = [d for d in balls if math.dist(d.center, pred) <= gate]
                if near:
                    chosen = min(near, key=lambda d: math.dist(d.center, pred) / max(d.conf, 1e-3))
                else:
                    best = max(balls, key=lambda d: d.conf)
                    if best.conf >= cfg.reacquire_conf and self._coast > 0.3:
                        chosen = best

        if chosen is not None:
            c = chosen.center
            if self._last_det is not None and self._coast > 0:
                inst = ((c[0] - self._last_det[0]) / self._coast, (c[1] - self._last_det[1]) / self._coast)
                a = cfg.vel_alpha
                self._vel = (a * inst[0] + (1 - a) * self._vel[0], a * inst[1] + (1 - a) * self._vel[1])
                vmax = 3.0 * self.W  # Pixels per second; cap implausible velocities.
                speed = math.hypot(*self._vel)
                if speed > vmax:
                    self._vel = (self._vel[0] * vmax / speed, self._vel[1] * vmax / speed)
            self._pos = c
            self._last_det = c
            self._coast = 0.0
        elif self._pos is not None:
            if self._coast > cfg.max_coast_s:
                self._pos, self._last_det, self._vel = None, None, (0.0, 0.0)
            else:
                damp = math.exp(-2.0 * dt)
                self._vel = (self._vel[0] * damp, self._vel[1] * damp)
                self._pos = (min(max(pred[0], 0.0), self.W), min(max(pred[1], 0.0), self.H))
        return self._state()


# ----------------------------------------------------------- Smoothing util
def smooth_damp(current: float, target: float, vel: float, smooth_time: float,
                max_speed: float, dt: float) -> tuple[float, float]:
    """Critically damped spring with a speed limit (SmoothDamp-style)."""
    smooth_time = max(1e-4, smooth_time)
    omega = 2.0 / smooth_time
    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = current - target
    max_change = max_speed * smooth_time
    change = max(-max_change, min(max_change, change))
    tmp_target = current - change
    temp = (vel + omega * change) * dt
    vel = (vel - omega * temp) * exp
    out = tmp_target + (change + temp) * exp
    if (target - current > 0.0) == (out > target):  # Prevent overshoot.
        out = target
        vel = (out - target) / dt if dt > 0 else 0.0
    return out, vel


# ------------------------------------------------------------ Virtual camera
@dataclass
class CropRect:
    cx: float
    cy: float
    w: float
    h: float

    @property
    def x0(self) -> float:
        return self.cx - self.w / 2

    @property
    def y0(self) -> float:
        return self.cy - self.h / 2

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x0 + margin <= x <= self.x0 + self.w - margin
                and self.y0 + margin <= y <= self.y0 + self.h - margin)


class VirtualCamera:
    def __init__(self, frame_w: int, frame_h: int, cfg: FramingConfig | None = None,
                 tracker_cfg: TrackerConfig | None = None) -> None:
        self.cfg = cfg or FramingConfig()
        self.tcfg = tracker_cfg or TrackerConfig()
        self.W, self.H = frame_w, frame_h
        self.aspect = self.cfg.out_height / self.cfg.out_width  # Height / width.
        self.max_w = min(float(frame_w), frame_h / self.aspect)
        self.min_w = min(self.max_w, float(self.cfg.min_crop_px))
        self.w = self._cw(self.cfg.crop_width_wide * frame_w)
        self.cx, self.cy = frame_w / 2, frame_h / 2
        self.vx = self.vy = self.vw = 0.0

    def _cw(self, w: float) -> float:
        return max(self.min_w, min(self.max_w, w))

    def _clamp_center(self, cx: float, cy: float, w: float) -> tuple[float, float]:
        hw, hh = w / 2, w * self.aspect / 2
        return (min(max(cx, hw), self.W - hw), min(max(cy, hh), self.H - hh))

    @staticmethod
    def _dead(cur: float, tgt: float, dz: float) -> float:
        d = tgt - cur
        if abs(d) <= dz:
            return cur
        return tgt - math.copysign(dz, d)

    def _target(self, ball: BallState, players: np.ndarray | None) -> tuple[float, float, float]:
        cfg = self.cfg
        w_ball = cfg.crop_width_ball * self.W
        w_wide = cfg.crop_width_wide * self.W
        centroid = None
        if players is not None and len(players) >= 3:
            centroid = (float(np.median(players[:, 0])), float(np.median(players[:, 1])))

        if ball.pos is not None:
            bx, by = ball.pos
            tx = bx + ball.vel[0] * cfg.lead_seconds
            ty = by + ball.vel[1] * cfg.lead_seconds
            if centroid is not None:
                bw = cfg.ball_weight if ball.status == "tracked" else cfg.ball_weight * 0.6
                tx = bw * tx + (1 - bw) * centroid[0]
                ty = bw * ty + (1 - bw) * centroid[1]
            k = min(1.0, ball.coast_time / max(self.tcfg.max_coast_s, 1e-3))
            tw = self._cw(w_ball + (w_wide - w_ball) * k)
            # Keep the ball within the central region of the crop.
            lim = 0.30 * tw
            tx = bx + max(-lim, min(lim, tx - bx))
            ty = by + max(-lim * self.aspect, min(lim * self.aspect, ty - by))
            return tx, ty, tw
        if centroid is not None:
            return centroid[0], centroid[1], self._cw(w_wide)
        return self.cx, self.cy, self._cw(w_wide)

    def update(self, ball: BallState, players: np.ndarray | None, dt: float) -> CropRect:
        cfg = self.cfg
        if dt > 0:
            tx, ty, tw = self._target(ball, players)
            dz = cfg.dead_zone * self.w
            tx, ty = self._dead(self.cx, tx, dz), self._dead(self.cy, ty, dz)
            tx, ty = self._clamp_center(tx, ty, tw)
            vmax = cfg.max_pan_speed * self.W
            self.cx, self.vx = smooth_damp(self.cx, tx, self.vx, cfg.smooth_time_pan, vmax, dt)
            self.cy, self.vy = smooth_damp(self.cy, ty, self.vy, cfg.smooth_time_pan, vmax, dt)
            self.w, self.vw = smooth_damp(self.w, tw, self.vw, cfg.smooth_time_zoom, 0.5 * self.W, dt)
            self.w = self._cw(self.w)
            self.cx, self.cy = self._clamp_center(self.cx, self.cy, self.w)
        return CropRect(self.cx, self.cy, self.w, self.w * self.aspect)


def render_crop(frame: np.ndarray, rect: CropRect, out_w: int, out_h: int) -> np.ndarray:
    """Crop and resize at sub-pixel precision for smooth pans."""
    sx, sy = out_w / rect.w, out_h / rect.h
    m = np.array([[sx, 0, -rect.x0 * sx], [0, sy, -rect.y0 * sy]], dtype=np.float32)
    return cv2.warpAffine(frame, m, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
