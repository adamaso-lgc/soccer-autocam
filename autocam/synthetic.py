"""Generate a synthetic match (overhead pitch, ball, and 22 players) for camera-free testing.

    python -m autocam.synthetic demo.mp4 --seconds 20

Also write demo.truth.json with the ground-truth ball position for each frame.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

FIELD_BGR = (50, 130, 55)
LINE_BGR = (195, 225, 195)  # Bright, but below the ball's "white" threshold.
BALL_BGR = (255, 255, 255)
TEAM_A = (30, 30, 220)  # Red (BGR).
TEAM_B = (220, 60, 30)  # Blue (BGR).


def _formation(w: int, h: int) -> np.ndarray:
    xs = np.array([0.08, 0.22, 0.34, 0.45, 0.55, 0.66, 0.78, 0.92])
    pts = []
    for team in (0, 1):
        for i in range(11):
            col = [0.05, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.45, 0.45][i]
            row = [0.5, 0.15, 0.4, 0.6, 0.85, 0.15, 0.4, 0.6, 0.85, 0.35, 0.65][i]
            x = col if team == 0 else 1 - col
            pts.append((x * w * 0.9 + w * 0.05, row * h * 0.8 + h * 0.1))
    return np.array(pts, dtype=np.float32)


def generate(path: str, width: int = 2560, height: int = 720, seconds: float = 20.0,
             fps: int = 25, seed: int = 7, occlusions: int = 3) -> list[dict]:
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    home = _formation(width, height)
    phase = rng.uniform(0, 2 * np.pi, size=(22, 2))

    # Ball trajectory: passes between players.
    waypoints: list[tuple[int, np.ndarray]] = []
    t = 0
    cur = int(rng.integers(0, 22))
    while t < n + fps * 2:
        waypoints.append((t, home[cur].copy()))
        t += int(rng.uniform(0.9, 1.8) * fps)
        cur = int(rng.integers(0, 22))
    ball = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        for (t0, p0), (t1, p1) in zip(waypoints, waypoints[1:]):
            if t0 <= i < t1:
                u = (i - t0) / (t1 - t0)
                u = u * u * (3 - 2 * u)  # Ease-in/ease-out interpolation.
                ball[i] = p0 + (p1 - p0) * u
                break

    hidden = np.zeros(n, dtype=bool)  # The ball "disappears" during selected intervals.
    for _ in range(occlusions):
        s = int(rng.uniform(0.15, 0.85) * n)
        hidden[s : s + int(0.8 * fps)] = True

    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not write {path}")
    truth = []
    for i in range(n):
        img = np.full((height, width, 3), FIELD_BGR, dtype=np.uint8)
        cv2.rectangle(img, (int(width * 0.03), int(height * 0.06)), (int(width * 0.97), int(height * 0.94)), LINE_BGR, 2)
        cv2.line(img, (width // 2, int(height * 0.06)), (width // 2, int(height * 0.94)), LINE_BGR, 2)
        cv2.circle(img, (width // 2, height // 2), int(height * 0.14), LINE_BGR, 2)
        sec = i / fps
        pull = 0.25 * (ball[i][None, :] - home)
        for k in range(22):
            wob = np.array([np.sin(sec * 0.9 + phase[k, 0]), np.cos(sec * 0.7 + phase[k, 1])]) * 18
            pos = home[k] + np.clip(pull[k], -200, 200) + wob
            cv2.circle(img, (int(pos[0]), int(pos[1])), 10, TEAM_A if k < 11 else TEAM_B, -1)
        if not hidden[i]:
            cv2.circle(img, (int(ball[i][0]), int(ball[i][1])), 5, BALL_BGR, -1)
        writer.write(img)
        truth.append({"frame": i, "ball": [float(ball[i][0]), float(ball[i][1])], "visible": bool(not hidden[i])})
    writer.release()
    return truth


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("output")
    ap.add_argument("--width", type=int, default=2560)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    truth = generate(a.output, a.width, a.height, a.seconds, a.fps, a.seed)
    out = a.output.rsplit(".", 1)[0] + ".truth.json"
    with open(out, "w") as f:
        json.dump(truth, f)
    print(f"{len(truth)} frames -> {a.output} (+ {out})")


if __name__ == "__main__":
    main()
