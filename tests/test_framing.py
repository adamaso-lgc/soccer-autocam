import math

from autocam.config import FramingConfig
from autocam.detectors import Detection
from autocam.framing import BallState, BallTracker, VirtualCamera, smooth_damp

W, H, DT = 2560, 720, 1 / 25


def ball_det(x, y, conf=0.9):
    return Detection("ball", x - 4, y - 4, x + 4, y + 4, conf)


def test_smooth_damp_converges_without_overshoot():
    x, v = 0.0, 0.0
    for _ in range(200):
        x, v = smooth_damp(x, 100.0, v, 0.5, 1e9, DT)
        assert x <= 100.0 + 1e-6
    assert abs(x - 100.0) < 0.5


def test_smooth_damp_respects_max_speed():
    x, v = 0.0, 0.0
    prev = x
    for _ in range(50):
        x, v = smooth_damp(x, 10_000.0, v, 0.5, 200.0, DT)
        assert (x - prev) / DT <= 200.0 * 1.05
        prev = x


def test_tracker_follows_then_coasts_then_loses():
    t = BallTracker(W, H)
    s = None
    for i in range(10):
        s = t.update([ball_det(500 + 10 * i, 300)], DT)
    assert s.status == "tracked" and s.vel[0] > 100  # ~250 px/s

    # With no detections, predict (coast) and then lose the track.
    statuses = []
    for _ in range(60):
        statuses.append(t.update([], DT).status)
    assert "coasting" in statuses
    assert statuses[-1] == "lost"


def test_tracker_ignores_far_false_positive_with_low_conf():
    t = BallTracker(W, H)
    for _ in range(5):
        t.update([ball_det(1000, 300)], DT)
    s = t.update([ball_det(1005, 300), ball_det(2300, 100, conf=0.3)], DT)
    assert abs(s.pos[0] - 1005) < 5


def test_camera_follows_ball_and_stays_in_bounds():
    cam = VirtualCamera(W, H, FramingConfig())
    ball = BallState((2400.0, 650.0), (0.0, 0.0), "tracked", 0.0)
    rect = None
    for _ in range(300):
        rect = cam.update(ball, None, DT)
        assert rect.x0 >= -1e-6 and rect.y0 >= -1e-6
        assert rect.x0 + rect.w <= W + 1e-6 and rect.y0 + rect.h <= H + 1e-6
    assert rect.contains(2400, 650)


def test_camera_speed_is_limited():
    cfg = FramingConfig()
    cam = VirtualCamera(W, H, cfg)
    ball = BallState((2400.0, 360.0), (0.0, 0.0), "tracked", 0.0)
    prev = cam.cx
    for _ in range(100):
        rect = cam.update(ball, None, DT)
        assert abs(rect.cx - prev) / DT <= cfg.max_pan_speed * W * 1.1
        prev = rect.cx


def test_dead_zone_prevents_jitter():
    cam = VirtualCamera(W, H, FramingConfig())
    ball = BallState((cam.cx + 5, cam.cy + 3), (0.0, 0.0), "tracked", 0.0)
    start = (cam.cx, cam.cy)
    for _ in range(50):
        cam.update(ball, None, DT)
    assert math.dist(start, (cam.cx, cam.cy)) < 1.0


def test_lost_ball_falls_back_to_players_centroid():
    import numpy as np

    cam = VirtualCamera(W, H, FramingConfig())
    players = np.array([[1800, 300], [1900, 350], [2000, 400], [1850, 320]], dtype=np.float32)
    lost = BallState(None, (0.0, 0.0), "lost", 5.0)
    for _ in range(200):
        rect = cam.update(lost, players, DT)
    assert abs(rect.cx - 1875) < 100
