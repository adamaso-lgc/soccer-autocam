"""CLI: python -m autocam --source jogo.mp4 --output saida.mp4"""
from __future__ import annotations

import argparse
import sys

from .config import FramingConfig, PipelineConfig, TrackerConfig
from .detectors import DemoColorDetector, YoloDetector
from .filters import parse_roi
from .pipeline import run


def _size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autocam", description="Câmara virtual que segue a bola.")
    p.add_argument("--source", required=True, help="ficheiro, rtsp://..., ou índice da webcam (0)")
    p.add_argument("--output", default="output.mp4", help="ficheiro .mp4 ou rtmp://a.rtmp.youtube.com/live2/CHAVE")
    p.add_argument("--detector", choices=["yolo", "demo"], default="yolo",
                   help="'demo' só funciona no vídeo sintético")
    p.add_argument("--model", default="yolov8m.pt", help="pesos YOLO (COCO ou o teu modelo afinado)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default=None, help="cuda:0, cpu, mps...")
    p.add_argument("--tile-size", type=int, default=0, help="inferência em tiles para a bola (ex: 1280); 0 = desligado")
    p.add_argument("--window-size", type=int, default=640,
                   help="com a bola seguida, procurá-la só numa janela deste tamanho (px); 0 = desligado")
    p.add_argument("--person-every", type=int, default=5, help="detetar jogadores de N em N deteções")
    p.add_argument("--full-search-every", type=float, default=2.0,
                   help="segundos entre procuras completas (tiles) enquanto a bola é seguida")
    p.add_argument("--ball-conf", type=float, default=0.10)
    p.add_argument("--person-conf", type=float, default=0.35)
    p.add_argument("--detect-every", type=int, default=1, help="correr deteção de N em N frames")
    p.add_argument("--out-size", type=_size, default=(1920, 1080), metavar="WxH")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--bitrate", default=None, help="ex: 6000k (RTMP usa 6000k por defeito)")
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--realtime", action="store_true", help="limitar ao fps (para testar RTMP com ficheiros)")
    p.add_argument("--debug-video", default=None, help="guardar vídeo com overlay das deteções e do crop")
    p.add_argument("--roi", default=None,
                   help="polígono do campo em frações 0-1: 'x,y;x,y;x,y;...' (ver scripts/pick_roi.py)")
    p.add_argument("--static-after", type=float, default=3.0,
                   help="ignora 'bolas' paradas há mais de N segundos (0 = desligado)")
    p.add_argument("--crop-ball", type=float, default=FramingConfig.crop_width_ball)
    p.add_argument("--crop-wide", type=float, default=FramingConfig.crop_width_wide)
    p.add_argument("--smooth-pan", type=float, default=FramingConfig.smooth_time_pan)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    framing = FramingConfig(out_width=a.out_size[0], out_height=a.out_size[1],
                            crop_width_ball=a.crop_ball, crop_width_wide=a.crop_wide,
                            smooth_time_pan=a.smooth_pan)
    cfg = PipelineConfig(source=a.source, dest=a.output, fps=a.fps, bitrate=a.bitrate, preset=a.preset,
                         detect_every=a.detect_every, max_frames=a.max_frames, realtime=a.realtime,
                         debug_video=a.debug_video, framing=framing, tracker=TrackerConfig(),
                         roi=parse_roi(a.roi) if a.roi else None, static_after_s=a.static_after,
                         person_every=a.person_every, full_search_every_s=a.full_search_every)
    if a.detector == "demo":
        detector = DemoColorDetector()
    else:
        detector = YoloDetector(model=a.model, imgsz=a.imgsz, ball_conf=a.ball_conf,
                                person_conf=a.person_conf, device=a.device, tile_size=a.tile_size,
                                window_size=a.window_size)
    res = run(cfg, detector)
    print(f"{res.frames} frames em {res.seconds:.1f}s ({res.fps:.1f} fps), "
          f"deteção média {res.detect_ms:.0f} ms, estados da bola: {res.status_counts}")
    n = res.window_calls + res.full_calls
    print(f"diagnóstico: {n} deteções ({res.window_calls} em janela, {res.full_calls} completas); "
          f"bolas vistas pelo modelo: {res.ball_raw}, fora do ROI: {res.ball_roi_rejected}, "
          f"paradas: {res.ball_static_rejected}; deteções sem bola aceite: {res.detect_no_ball}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
