"""Filtros de falsos positivos aplicados às deteções antes do tracker.

* ``RoiFilter``    - só aceita deteções dentro do polígono do campo (descarta bolas
                     suplentes, marcas no relvado, público, banco...).
* ``StaticFilter`` - descarta "bolas" que ficam no mesmo sítio durante vários
                     segundos (bolas paradas fora de jogo, logótipos, sinais).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .detectors import Detection


def parse_roi(text: str) -> list[tuple[float, float]]:
    """'0.02,0.20;0.98,0.20;0.99,0.95;0.01,0.95' -> [(x, y), ...] em frações (0-1) do frame."""
    pts = []
    for part in text.replace(" ", "").split(";"):
        if not part:
            continue
        x, y = part.split(",")
        pts.append((float(x), float(y)))
    if len(pts) < 3:
        raise ValueError("O ROI precisa de pelo menos 3 pontos (formato: 'x,y;x,y;x,y' em frações 0-1).")
    if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in pts):
        raise ValueError("Os pontos do ROI têm de ser frações entre 0 e 1 (ex: 0.25,0.5).")
    return pts


class RoiFilter:
    def __init__(self, polygon_norm: list[tuple[float, float]], width: int, height: int) -> None:
        self.poly = np.array([[x * width, y * height] for x, y in polygon_norm], dtype=np.float32)

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.poly.reshape(-1, 1, 2), (float(x), float(y)), False) >= 0

    def split(self, dets: list[Detection]) -> tuple[list[Detection], list[Detection]]:
        """(aceites, rejeitadas). Bola: centro dentro. Pessoa: pés (base da caixa) dentro."""
        keep, drop = [], []
        for d in dets:
            x, y = d.center if d.label == "ball" else d.foot
            (keep if self.contains(x, y) else drop).append(d)
        return keep, drop


class StaticFilter:
    """Rejeita bolas que ficam (quase) paradas durante ``after_s`` segundos.

    Uma bola de jogo que fique parada mais tempo que isso (ex: antes de um livre) deixa de
    ser aceite até se voltar a mexer; o tracker segue os jogadores entretanto.
    """

    def __init__(self, width: int, after_s: float = 3.0, radius_frac: float = 0.006,
                 forget_s: float = 1.0) -> None:
        self.after_s = after_s
        self.radius = radius_frac * width
        self.forget_s = forget_s
        self._anchors: list[list[float]] = []  # [x, y, t_primeira, t_ultima]

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
