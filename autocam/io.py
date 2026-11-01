"""Entrada (ficheiro / RTSP / webcam) e saída (mp4 / RTMP YouTube)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from typing import Protocol

import cv2
import numpy as np

LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://")


class VideoSource:
    """Lê frames de ficheiro, stream (RTSP/HTTP) ou webcam.

    Em fontes "live" corre uma thread que guarda só o frame mais recente: se a
    inferência for mais lenta que o stream, descartam-se frames em vez de acumular
    latência.
    """

    def __init__(self, source: str) -> None:
        self.live = source.isdigit() or source.lower().startswith(LIVE_PREFIXES)
        if source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Não consegui abrir a fonte de vídeo: {source}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 0.0
        self._latest: np.ndarray | None = None
        self._cond = threading.Condition()
        self._stop = False
        self._eof = False
        self._thread: threading.Thread | None = None
        if self.live:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self) -> None:
        while not self._stop:
            ok, frame = self.cap.read()
            with self._cond:
                if not ok:
                    self._eof = True
                    self._cond.notify_all()
                    return
                self._latest = frame
                self._cond.notify_all()

    def read(self) -> np.ndarray | None:
        if not self.live:
            ok, frame = self.cap.read()
            return frame if ok else None
        with self._cond:
            while self._latest is None and not self._eof:
                self._cond.wait(timeout=5.0)
                if self._latest is None and not self._eof:
                    return None  # sem frames durante 5s
            frame, self._latest = self._latest, None
            return frame

    def close(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.cap.release()


class Sink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


class FfmpegSink:
    """Envia frames BGR para o ffmpeg (H.264) -> ficheiro mp4 ou RTMP (YouTube)."""

    def __init__(self, dest: str, width: int, height: int, fps: float,
                 bitrate: str | None = None, preset: str = "veryfast") -> None:
        self.dest = dest
        rtmp = dest.lower().startswith(("rtmp://", "rtmps://"))
        fps_i = max(1, int(round(fps)))
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-"]
        if rtmp:
            br = bitrate or "6000k"
            kbps = int(br.lower().rstrip("k"))
            cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p",
                    "-b:v", br, "-maxrate", br, "-bufsize", f"{2 * kbps}k",
                    "-g", str(fps_i * 2), "-keyint_min", str(fps_i * 2), "-sc_threshold", "0",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                    "-shortest", "-f", "flv", dest]
        else:
            codec = ["-b:v", bitrate] if bitrate else ["-crf", "20"]
            cmd += ["-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", *codec,
                    "-movflags", "+faststart", dest]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err)

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"ffmpeg terminou inesperadamente: {self._stderr()}") from e

    def _stderr(self) -> str:
        self._err.seek(0)
        return self._err.read().decode(errors="replace").strip()

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        if self.proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg falhou (código {self.proc.returncode}): {self._stderr()}")
        self._err.close()


class OpenCvSink:
    """Fallback sem ffmpeg (só ficheiros)."""

    def __init__(self, dest: str, width: int, height: int, fps: float) -> None:
        self.writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Não consegui criar o ficheiro de saída: {dest}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def make_sink(dest: str, width: int, height: int, fps: float,
              bitrate: str | None = None, preset: str = "veryfast") -> Sink:
    is_rtmp = dest.lower().startswith(("rtmp://", "rtmps://"))
    if shutil.which("ffmpeg"):
        return FfmpegSink(dest, width, height, fps, bitrate, preset)
    if is_rtmp:
        raise RuntimeError("ffmpeg não encontrado: é necessário para enviar para RTMP/YouTube.")
    return OpenCvSink(dest, width, height, fps)
