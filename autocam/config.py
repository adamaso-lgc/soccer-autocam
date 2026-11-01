"""Configuração do pipeline (todos os valores têm defaults razoáveis)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FramingConfig:
    # Resolução do vídeo final
    out_width: int = 1920
    out_height: int = 1080

    # Largura do crop como fração da largura do frame original
    crop_width_ball: float = 0.30  # bola localizada -> zoom mais apertado
    crop_width_wide: float = 0.50  # bola perdida -> plano mais aberto
    min_crop_px: int = 960  # nunca fazer zoom abaixo disto (qualidade)

    # Suavização (segundos que a câmara demora a "chegar" ao alvo)
    smooth_time_pan: float = 0.55
    smooth_time_zoom: float = 1.2
    max_pan_speed: float = 0.30  # larguras de frame por segundo

    dead_zone: float = 0.03  # fração da largura do crop onde a câmara não mexe
    ball_weight: float = 0.70  # peso da bola vs centro dos jogadores
    lead_seconds: float = 0.25  # antecipa o movimento da bola


@dataclass
class TrackerConfig:
    gate: float = 0.12  # raio de associação (fração da largura do frame)
    max_coast_s: float = 1.2  # segundos a prever a bola sem a detetar
    min_start_conf: float = 0.15  # confiança mínima para iniciar um track
    reacquire_conf: float = 0.50  # confiança para saltar para outra bola
    vel_alpha: float = 0.5  # suavização da velocidade estimada


@dataclass
class PipelineConfig:
    source: str = ""
    dest: str = "output.mp4"  # ficheiro .mp4 ou rtmp://...
    fps: float | None = None  # None -> usa o fps da fonte
    bitrate: str | None = None  # ex: "6000k" (obrigatório para RTMP, default 6000k)
    preset: str = "veryfast"
    detect_every: int = 1  # correr deteção de N em N frames
    max_frames: int | None = None
    realtime: bool = False  # limita a velocidade ao fps (testar RTMP com ficheiros)
    debug_video: str | None = None  # vídeo com overlay (deteções + crop)
    roi: list[tuple[float, float]] | None = None  # polígono do campo, frações 0-1 do frame
    static_after_s: float = 3.0  # ignora bolas paradas há mais de N s (0 = desligado)
    person_every: int = 5  # detetar jogadores de N em N deteções (o resto reutiliza)
    full_search_every_s: float = 2.0  # com bola seguida, procura completa (tiles) a cada N s
    framing: FramingConfig = field(default_factory=FramingConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
