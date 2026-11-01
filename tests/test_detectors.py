import numpy as np

from autocam.detectors import YoloDetector, _tile_origins


class FakeTensor:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeBoxes:
    def __init__(self, rows):  # rows: (x1,y1,x2,y2,conf,cls)
        r = np.array(rows, dtype=np.float32).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = FakeTensor(r[:, :4]), FakeTensor(r[:, 4]), FakeTensor(r[:, 5])
        self._n = len(r)

    def __len__(self):
        return self._n


class FakeResult:
    names = {0: "person", 1: "sports ball", 2: "car"}

    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """First call uses the full frame; subsequent calls use tiles."""

    def __init__(self, full_rows, tile_rows_by_call):
        self.full_rows, self.tile_rows = full_rows, tile_rows_by_call
        self.calls = []

    def predict(self, images, imgsz, conf, device, verbose):
        self.calls.append((len(images), imgsz, conf))
        if len(self.calls) == 1:
            return [FakeResult(self.full_rows)]
        return [FakeResult(self.tile_rows.get(i, [])) for i in range(len(images))]


def test_tile_origins_cover_frame():
    origins = _tile_origins(1000, 2500, 1000, 0.2)
    assert max(x for x, _ in origins) == 1500  # Last tile reaches the edge.
    assert all(0 <= x <= 1500 and y == 0 for x, y in origins)


def test_class_and_confidence_filtering():
    rows = [(10, 10, 30, 80, 0.9, 0),   # pessoa ok
            (50, 50, 60, 60, 0.05, 1),  # Ball below threshold.
            (70, 70, 80, 80, 0.4, 1),   # Accepted ball.
            (0, 0, 100, 100, 0.99, 2)]  # Ignore car.
    det = YoloDetector(model_obj=FakeModel(rows, {}))
    out = det.detect(np.zeros((200, 200, 3), np.uint8))
    assert sorted(d.label for d in out) == ["ball", "person"]


def test_tiles_add_balls_with_correct_offsets_and_dedupe():
    full = [(10, 10, 30, 80, 0.9, 0)]
    # Tile 1 (origin x=800) finds a ball at (100..110, 50..60).
    model = FakeModel(full, {1: [(100, 50, 110, 60, 0.5, 1)]})
    det = YoloDetector(model_obj=model, tile_size=1000, tile_overlap=0.2)
    out = det.detect(np.zeros((1000, 2500, 3), np.uint8))
    balls = [d for d in out if d.label == "ball"]
    assert len(balls) == 1
    assert (balls[0].x1, balls[0].y1) == (900.0, 50.0)  # 800 + 100
    assert any(d.label == "person" for d in out)
    assert model.calls[1][1] == 1000  # imgsz das tiles = tile_size


class ShapeRoutedModel:
    """Respond based on image size: full frame -> player; window -> ball."""

    def __init__(self, full_w):
        self.full_w, self.calls = full_w, []

    def predict(self, images, imgsz, conf, device, verbose):
        img = images[0]
        self.calls.append((img.shape[1], img.shape[0], imgsz))
        if img.shape[1] == self.full_w:
            return [FakeResult([(10, 10, 30, 80, 0.9, 0)])]
        return [FakeResult([(50, 60, 58, 68, 0.6, 1)])]  # Ball inside the window.


def test_window_detection_uses_one_small_call_and_correct_offsets():
    model = ShapeRoutedModel(full_w=4096)
    det = YoloDetector(model_obj=model, window_size=640)
    frame = np.zeros((1080, 4096, 3), np.uint8)

    out = det.detect(frame, hint=(1000.0, 500.0), persons=False)
    assert model.calls == [(640, 640, 640)]  # Window only, at native resolution.
    (ball,) = out
    # Window origin is (680, 180): 1000-320, 500-320.
    assert (ball.x1, ball.y1) == (730.0, 240.0)

    model.calls.clear()
    out = det.detect(frame, hint=(1000.0, 500.0), persons=True)
    assert len(model.calls) == 2 and {d.label for d in out} == {"ball", "person"}


def test_window_is_clamped_at_frame_edges():
    model = ShapeRoutedModel(full_w=4096)
    det = YoloDetector(model_obj=model, window_size=640)
    frame = np.zeros((1080, 4096, 3), np.uint8)
    (ball,) = det.detect(frame, hint=(4090.0, 5.0), persons=False)
    assert (ball.x1, ball.y1) == (4096 - 640 + 50.0, 60.0)  # Origin (3456, 0).


def test_no_hint_or_window_disabled_falls_back_to_full_search():
    model = FakeModel([(10, 10, 30, 80, 0.9, 0)], {})
    det = YoloDetector(model_obj=model, window_size=0)
    det.detect(np.zeros((200, 200, 3), np.uint8), hint=(100.0, 100.0))
    assert len(model.calls) == 1 and model.calls[0][1] == 1280  # One full-frame pass (imgsz=1280).
