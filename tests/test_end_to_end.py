"""Pipeline completo com o vídeo sintético (sem YOLO, sem GPU)."""
import cv2

from autocam.config import PipelineConfig
from autocam.detectors import DemoColorDetector
from autocam.pipeline import run
from autocam.synthetic import generate


def test_pipeline_follows_synthetic_ball(tmp_path):
    src = str(tmp_path / "game.mp4")
    out = str(tmp_path / "out.mp4")
    truth = generate(src, width=1920, height=540, seconds=6, fps=25, seed=3, occlusions=1)

    res = run(PipelineConfig(source=src, dest=out), DemoColorDetector(), collect=True)

    assert res.frames == len(truth)

    cap = cv2.VideoCapture(out)
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 1920
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 1080
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == res.frames
    cap.release()

    visible = [(l, t) for l, t in zip(res.log, truth) if t["visible"]]
    inside = sum(
        abs(t["ball"][0] - l["cx"]) <= l["w"] / 2 and abs(t["ball"][1] - l["cy"]) <= l["h"] / 2
        for l, t in visible
    )
    assert inside / len(visible) >= 0.85

    # movimento suave: nenhum salto brusco entre frames consecutivos
    steps = [abs(b["cx"] - a["cx"]) for a, b in zip(res.log, res.log[1:])]
    assert max(steps) < 0.03 * 1920


def test_debug_video_is_written_for_awkward_sizes(tmp_path):
    """Regressão: tamanhos que arredondam de forma diferente não podem dar um ficheiro de debug vazio."""
    src = str(tmp_path / "game.mp4")
    dbg = str(tmp_path / "dbg.mp4")
    generate(src, width=3001, height=1003, seconds=2, fps=25, seed=1, occlusions=0)

    res = run(PipelineConfig(source=src, dest=str(tmp_path / "out.mp4"), debug_video=dbg), DemoColorDetector())

    cap = cv2.VideoCapture(dbg)
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == res.frames
    ok, img = cap.read()
    assert ok and img.shape[1] % 2 == 0 and img.shape[0] % 2 == 0
    cap.release()


class RecordingDetector:
    def __init__(self):
        self.inner = DemoColorDetector()
        self.calls = []

    def detect(self, frame, hint=None, persons=True):
        self.calls.append((hint, persons))
        return self.inner.detect(frame)


def test_pipeline_uses_hint_and_skips_persons_between_refreshes(tmp_path):
    src = str(tmp_path / "game.mp4")
    generate(src, width=1920, height=540, seconds=6, fps=25, seed=3, occlusions=0)
    det = RecordingDetector()
    cfg = PipelineConfig(source=src, dest=str(tmp_path / "out.mp4"), person_every=5, full_search_every_s=1.0)

    res = run(cfg, det)

    assert res.window_calls > 0 and res.full_calls > 0
    assert res.window_calls + res.full_calls == len(det.calls)
    hints = [h for h, _ in det.calls]
    assert hints[0] is None and any(h is not None for h in hints[5:])  # arranca com procura completa
    assert sum(1 for h in hints if h is None) >= len(hints) // 25  # procuras completas periódicas
    persons_flags = [p for _, p in det.calls]
    assert persons_flags[0] is True and persons_flags.count(False) > persons_flags.count(True)
