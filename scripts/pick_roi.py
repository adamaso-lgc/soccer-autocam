#!/usr/bin/env python
"""Ajuda a definir o polígono do campo (--roi).

Modo clique (abre uma janela):
    python scripts/pick_roi.py data/clip.mp4 --time 5
    Clica nos cantos do campo (linhas laterais e de fundo, em volta da zona de jogo).
    u = desfazer último ponto | Enter = terminar e imprimir | q/Esc = cancelar

Modo grelha (sem janela; grava uma imagem com coordenadas para leres à mão):
    python scripts/pick_roi.py data/clip.mp4 --time 5 --grid outputs/grid.png
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def grab(path: str, t: float) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Não consegui abrir {path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Não consegui ler um frame nesse instante.")
    return frame


def save_grid(frame: np.ndarray, out: str) -> None:
    h, w = frame.shape[:2]
    img = frame.copy()
    for i in range(1, 20):  # linhas a cada 0.05
        f = i / 20
        major = i % 2 == 0
        col = (0, 255, 255) if major else (180, 180, 180)
        cv2.line(img, (int(f * w), 0), (int(f * w), h), col, 1)
        cv2.line(img, (0, int(f * h)), (w, int(f * h)), col, 1)
        if major:
            cv2.putText(img, f"{f:.1f}", (int(f * w) + 3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(img, f"{f:.1f}", (3, int(f * h) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(out, img)
    print(f"Grelha gravada em {out}. Lê as frações (x,y) dos cantos do campo e monta:")
    print("  --roi 'x1,y1;x2,y2;x3,y3;x4,y4'")


def pick(frame: np.ndarray) -> list[tuple[float, float]]:
    h, w = frame.shape[:2]
    scale = min(1.0, 1600 / w)
    view = cv2.resize(frame, None, fx=scale, fy=scale) if scale != 1 else frame.copy()
    pts: list[tuple[int, int]] = []

    def redraw() -> None:
        img = view.copy()
        for p in pts:
            cv2.circle(img, p, 4, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32).reshape(-1, 1, 2)], len(pts) > 2, (255, 0, 255), 2)
        cv2.imshow("ROI", img)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            redraw()

    cv2.namedWindow("ROI")
    cv2.setMouseCallback("ROI", on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (13, 10):
            break
        if k == ord("u") and pts:
            pts.pop()
            redraw()
        if k in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit("Cancelado.")
    cv2.destroyAllWindows()
    vw, vh = view.shape[1], view.shape[0]
    return [(x / vw, y / vh) for x, y in pts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--time", type=float, default=5.0, help="segundo do vídeo a usar")
    ap.add_argument("--grid", default=None, help="em vez de janela, grava uma imagem com grelha")
    a = ap.parse_args()
    frame = grab(a.video, a.time)
    if a.grid:
        save_grid(frame, a.grid)
        return
    pts = pick(frame)
    if len(pts) < 3:
        sys.exit("Precisas de pelo menos 3 pontos.")
    s = ";".join(f"{x:.3f},{y:.3f}" for x, y in pts)
    print("\nUsa isto no autocam:\n")
    print(f'  --roi "{s}"')


if __name__ == "__main__":
    main()
