"""False-positive filters applied to detections before tracking.

* ``RoiFilter`` - accepts detections inside the pitch polygon only.
* ``StaticFilter`` - rejects ball-like detections that remain in one place for
  several seconds (spare balls, logos, signs, etc.).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .detectors import Detection


def parse_roi(text: str) -> list[tuple[float, float]]:
    """Parse normalized frame coordinates: '0.02,0.2;0.98,0.2;0.99,0.95'."""
    pts = []
    for part in text.replace(" ", "").split(";"):
        if not part:
            continue
        x, y = part.split(",")
        pts.append((float(x), float(y)))
    if len(pts) < 3:
        raise ValueError("ROI needs at least 3 points (format: 'x,y;x,y;x,y' in normalized 0-1 coordinates).")
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in pts):
        raise ValueError("ROI points must be fractions between 0 and 1 (for example, 0.25,0.5).")
    return pts


class RoiFilter:
    def __init__(self, polygon_norm: list[tuple[float, float]], width: int, height: int) -> None:
        self.poly = np.array([[x * width, y * height] for x, y in polygon_norm], dtype=np.float32)

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.poly.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def split(self, dets: list[Detection]) -> tuple[list[Detection], list[Detection]]:
        """Return (accepted, rejected): ball center or person foot must be inside."""
        keep, drop = [], []
        for d in dets:
            x, y = d.center if d.label == "ball" else d.foot
            (keep if self.contains(x, y) else drop).append(d)
        return keep, drop


class StaticFilter:
    """Reject detections that remain (nearly) still for ``after_s`` seconds.

    A real ball that stays still longer than this (for example, before a free
    kick) is ignored until it moves again; the tracker follows players meanwhile.
    """

    def __init__(self, width: int, after_s: float = 3.0, radius_frac: float = 0.006,
                 forget_s: float = 1.0) -> None:
        self.after_s = after_s
        self.radius = radius_frac * width
        self.forget_s = forget_s
        self._anchors: list[list[float]] = []  # [x, y, first_seen, last_seen]

    def split(self, balls: list[Detection], t: float) -> tuple[list[Detection], list[Detection]]:
        if self.after_s <= 0:
            return list(balls), []
        self._anchors = [a for a in self._anchors if t - a[3] <= self.forget_s]
        keep, drop = [], []
        for d in balls:
            cx, cy = d.center
            anchor = next((a for a in self._anchors if math.hypot(a[0] - cx, a[1] - cy) <= self.radius), None)
            if anchor is None:
                self._anchors.append([cx, cy, t, t])
                keep.append(d)
                continue
            anchor[3] = t
            (drop if t - anchor[2] >= self.after_s else keep).append(d)
        return keep, drop
