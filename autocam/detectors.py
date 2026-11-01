"""Ball and player detectors.

* ``YoloDetector`` - Ultralytics YOLO detector with optional tiled inference
  to find small balls in high-resolution frames.
* ``DemoColorDetector`` - color-based detector for synthetic demo footage only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    label: str  # "ball" or "person".
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def foot(self) -> tuple[float, float]:
        """Ground point (center of the bottom edge), useful for pitch position."""
        return ((self.x1 + self.x2) / 2, self.y2)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, hint: tuple[float, float] | None = None,
               persons: bool = True) -> list[Detection]:
        """A hint is the likely ball position for a fast local search.

        If ``persons`` is false, the caller does not need player detections
        for this frame.
        """
        ...


# --------------------------------------------------------------------- YOLO
def _tile_origins(h: int, w: int, size: int, overlap: float) -> list[tuple[int, int]]:
    step = max(1, int(size * (1 - overlap)))

    def starts(total: int) -> list[int]:
        if total <= size:
            return [0]
        s = list(range(0, total - size + 1, step))
        if s[-1] != total - size:
            s.append(total - size)
        return s

    return [(x, y) for y in starts(h) for x in starts(w)]


def _iou(a: Detection, b: Detection) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / ua if ua > 0 else 0.0


def _nms(dets: list[Detection], thr: float = 0.3) -> list[Detection]:
    keep: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.conf, reverse=True):
        if all(_iou(d, k) < thr for k in keep):
            keep.append(d)
    return keep


class YoloDetector:
    """Wrapper for Ultralytics YOLO.

    ``model`` may be a COCO pretrained model ("sports ball" class) or a
    soccer-finetuned model. Accepted class names are configured by
    ``ball_names`` and ``person_names``.
    """

    def __init__(
        self,
        model: str = "yolov8m.pt",
        imgsz: int = 1280,
        ball_conf: float = 0.10,
        person_conf: float = 0.35,
        device: str | None = None,
        tile_size: int = 0,
        tile_overlap: float = 0.2,
        window_size: int = 640,
        ball_names: Iterable[str] = ("sports ball", "ball", "soccer ball", "football"),
        person_names: Iterable[str] = ("person", "player", "goalkeeper", "referee"),
        model_obj=None,
    ) -> None:
        if model_obj is None:
            try:
                from ultralytics import YOLO
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "Ultralytics is not installed. Run: pip install -r requirements.txt"
                ) from e
            model_obj = YOLO(model)
        self.model = model_obj
        self.imgsz = imgsz
        self.ball_conf = ball_conf
        self.person_conf = person_conf
        self.device = device
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.window_size = window_size  # Square search window around the hint; 0 disables it.
        self.ball_names = {n.lower() for n in ball_names}
        self.person_names = {n.lower() for n in person_names}

    def _predict(self, images: list[np.ndarray], conf: float, imgsz: int):
        return self.model.predict(
            images, imgsz=imgsz, conf=conf, device=self.device, verbose=False
        )

    def _extract(self, result, offset: tuple[int, int]) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        ox, oy = offset
        out: list[Detection] = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            name = str(result.names[int(k)]).lower()
            if name in self.ball_names:
                label, thr = "ball", self.ball_conf
            elif name in self.person_names:
                label, thr = "person", self.person_conf
            else:
                continue
            if c < thr:
                continue
            out.append(
                Detection(label, float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy), float(c))
            )
        return out

    def _detect_window(self, frame: np.ndarray, hint: tuple[float, float], persons: bool) -> list[Detection]:
        """Fast search: look for the ball near ``hint`` and players only when requested."""
        h, w = frame.shape[:2]
        ww, wh = min(self.window_size, w), min(self.window_size, h)
        x0 = int(min(max(hint[0] - ww / 2, 0), w - ww))
        y0 = int(min(max(hint[1] - wh / 2, 0), h - wh))
        out: list[Detection] = []
        if persons:
            res = self._predict([frame], self.person_conf, self.imgsz)[0]
            out += [d for d in self._extract(res, (0, 0)) if d.label == "person"]
        res = self._predict([frame[y0 : y0 + wh, x0 : x0 + ww]], self.ball_conf, self.window_size)[0]
        out += [d for d in self._extract(res, (x0, y0)) if d.label == "ball"]
        return out

    def detect(self, frame: np.ndarray, hint: tuple[float, float] | None = None,
               persons: bool = True) -> list[Detection]:
        if hint is not None and self.window_size > 0:
            return self._detect_window(frame, hint, persons)
        h, w = frame.shape[:2]
        res = self._predict([frame], min(self.ball_conf, self.person_conf), self.imgsz)[0]
        dets = self._extract(res, (0, 0))

        if self.tile_size > 0:
            origins = _tile_origins(h, w, self.tile_size, self.tile_overlap)
            tile_balls: list[Detection] = []
            for i in range(0, len(origins), 8):  # Process tiles in batches of eight.
                chunk = origins[i : i + 8]
                crops = [frame[y : y + self.tile_size, x : x + self.tile_size] for x, y in chunk]
                results = self._predict(crops, self.ball_conf, self.tile_size)
                for r, org in zip(results, chunk):
                    tile_balls += [d for d in self._extract(r, org) if d.label == "ball"]
            balls = _nms([d for d in dets if d.label == "ball"] + tile_balls)
            dets = [d for d in dets if d.label != "ball"] + balls
        return dets


# --------------------------------------------------------------- Color demo
class DemoColorDetector:
    """Color detector for synthetic video (white ball, red/blue players)."""

    def detect(self, frame: np.ndarray, hint: tuple[float, float] | None = None,
               persons: bool = True) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dets: list[Detection] = []

        # Ball: small and nearly white.
        white = cv2.inRange(hsv, (0, 0, 240), (179, 30, 255))
        n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if 8 <= area <= 400 and 0.6 <= w / max(h, 1) <= 1.6:
                fill = area / float(w * h)
                dets.append(Detection("ball", x, y, x + w, y + h, float(min(1.0, fill))))

        # Players: highly saturated red (H<10 or >170) or blue (H 100-130).
        red = cv2.inRange(hsv, (0, 150, 80), (10, 255, 255)) | cv2.inRange(hsv, (170, 150, 80), (179, 255, 255))
        blue = cv2.inRange(hsv, (100, 150, 80), (130, 255, 255))
        mask = red | blue
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area >= 40:
                dets.append(Detection("person", x, y, x + w, y + h, 0.9))
        return dets
