import pytest

from autocam.detectors import Detection
from autocam.filters import RoiFilter, StaticFilter, parse_roi

W, H = 4096, 1080


def ball(x, y, conf=0.8):
    return Detection("ball", x - 4, y - 4, x + 4, y + 4, conf)


def person(x, foot_y):
    return Detection("person", x - 10, foot_y - 40, x + 10, foot_y, 0.9)


def test_parse_roi_ok_and_errors():
    assert parse_roi("0.1,0.2; 0.9,0.2;0.9,0.9") == [(0.1, 0.2), (0.9, 0.2), (0.9, 0.9)]
    with pytest.raises(ValueError):
        parse_roi("0.1,0.2;0.9,0.2")  # só 2 pontos
    with pytest.raises(ValueError):
        parse_roi("0.1,0.2;0.9,0.2;900,500")  # pixels em vez de frações


def test_roi_rejects_ball_outside_and_uses_feet_for_people():
    roi = RoiFilter(parse_roi("0.05,0.2;0.95,0.2;0.95,0.8;0.05,0.8"), W, H)
    inside_ball, outside_ball = ball(2000, 500), ball(2000, 1000)  # 1000 > 0.8*1080=864
    keep, drop = roi.split([inside_ball, outside_ball])
    assert keep == [inside_ball] and drop == [outside_ball]

    # a cabeça está dentro mas os pés estão fora -> rejeitada (ex: pessoa fora da linha)
    p_out = person(2000, 900)
    p_in = person(2000, 700)
    keep, drop = roi.split([p_in, p_out])
    assert keep == [p_in] and drop == [p_out]


def test_static_filter_drops_ball_that_never_moves():
    f = StaticFilter(W, after_s=3.0)
    dt = 1 / 25
    dropped_at = None
    for i in range(25 * 6):
        keep, drop = f.split([ball(1500, 900)], i * dt)
        if drop and dropped_at is None:
            dropped_at = i * dt
    assert dropped_at is not None and 2.9 <= dropped_at <= 3.2
    assert not keep  # continua rejeitada


def test_static_filter_keeps_moving_ball_and_ignores_static_one():
    f = StaticFilter(W, after_s=3.0)
    dt = 1 / 25
    for i in range(25 * 8):
        moving = ball(500 + 10 * i % 3000, 400)  # move-se 10 px por frame
        keep, _ = f.split([moving, ball(1500, 900, conf=0.9)], i * dt)
    assert any(abs(d.center[0] - 1500) > 100 for d in keep)  # a bola real passa
    assert all(abs(d.center[0] - 1500) > 100 or abs(d.center[1] - 900) > 100 for d in keep)  # a parada não


def test_static_filter_disabled():
    f = StaticFilter(W, after_s=0)
    keep, drop = f.split([ball(100, 100)], 100.0)
    assert len(keep) == 1 and not drop
