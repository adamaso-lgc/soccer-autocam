"""Video input (file / RTSP / webcam) and output (MP4 / YouTube RTMP)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Protocol

import cv2
import numpy as np

LIVE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://")


class VideoSource:
    """Read frames from a file, stream (RTSP/HTTP), or webcam.

    Live sources use a reader thread that retains only the newest frame. If
    inference is slower than the stream, frames are dropped instead of latency
    accumulating.
    """

    def __init__(self, source: str, reconnect_attempts: int = 5,
                 reconnect_delay_s: float = 1.0, realtime: bool = False) -> None:
        self.source = source
        self.reconnect_attempts = max(0, reconnect_attempts)
        self.reconnect_delay_s = max(0.0, reconnect_delay_s)
        self.live = source.isdigit() or source.lower().startswith(LIVE_PREFIXES)
        self.paced_file = realtime and not self.live
        self.paced = self.live or realtime
        if source.lower().startswith("rtsp://"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        self.reconnect_count = 0
        if self.live:
            for _ in range(self.reconnect_attempts):
                if self.cap.isOpened():
                    break
                self.cap.release()
                if self.reconnect_delay_s:
                    time.sleep(self.reconnect_delay_s)
                self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
                self.reconnect_count += 1
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source. Check the path, URL, credentials and camera availability.")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else 0.0
        self._latest: np.ndarray | None = None
        self._cond = threading.Condition()
        self._stop = False
        self._eof = False
        self._error: str | None = None
        self._latest_time: float | None = None
        self.last_frame_time: float | None = None
        self._thread: threading.Thread | None = None
        if self.paced:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()

    def _reader(self) -> None:
        frame_index = 0
        next_read = time.monotonic()
        source_interval = 1.0 / (self.fps if self.fps > 1 else 25.0)
        while not self._stop:
            if self.paced_file:
                delay = next_read - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            ok, frame = self.cap.read()
            captured_at = frame_index * source_interval if self.paced_file else time.monotonic()
            with self._cond:
                if not ok:
                    if not self.live or self._stop:
                        self._eof = True
                        self._cond.notify_all()
                        return
                else:
                    self._latest = frame
                    self._latest_time = captured_at
                    frame_index += 1
                    if self.paced_file:
                        next_read += source_interval
                    self._cond.notify_all()
                    continue

            # A live stream may drop temporarily. Reopen it a bounded number of
            # times so the process fails clearly instead of silently ending output.
            self.cap.release()
            recovered = False
            for _ in range(self.reconnect_attempts):
                if self._stop:
                    break
                if self.reconnect_delay_s:
                    time.sleep(self.reconnect_delay_s)
                reopened = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
                if not reopened.isOpened():
                    reopened.release()
                    continue
                width = int(reopened.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(reopened.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if (width, height) != (self.width, self.height):
                    reopened.release()
                    with self._cond:
                        self._error = (
                            f"Live source resolution changed from {self.width}x{self.height} "
                            f"to {width}x{height}; restart with the new resolution."
                        )
                        self._eof = True
                        self._cond.notify_all()
                    return
                self.cap = reopened
                self.reconnect_count += 1
                recovered = True
                break
            if not recovered:
                with self._cond:
                    if not self._stop:
                        self._error = (
                            f"Live source disconnected and could not be reopened after "
                            f"{self.reconnect_attempts} attempts. Check network, credentials and camera availability."
                        )
                    self._eof = True
                    self._cond.notify_all()
                return

    def read(self) -> np.ndarray | None:
        if not self.paced:
            ok, frame = self.cap.read()
            return frame if ok else None
        with self._cond:
            while self._latest is None and not self._eof:
                self._cond.wait(timeout=1.0)
            if self._error:
                raise RuntimeError(self._error)
            frame, self._latest = self._latest, None
            self.last_frame_time, self._latest_time = self._latest_time, None
            return frame

    def close(self) -> None:
        self._stop = True
        self.cap.release()
        if self._thread is not None:
            self._thread.join(timeout=5)


class Sink(Protocol):
    def write(self, frame: np.ndarray) -> None: ...
    def close(self) -> None: ...


class RealtimeSink:
    """Write the newest processed frame at a fixed cadence for live sources.

    Inference may run slower than the camera. Repeating the latest frame keeps
    output timestamps and stream duration aligned with wall time instead of
    accelerating the footage or emitting irregular frame intervals.
    """

    def __init__(self, sink: Sink, fps: float, debug_sink: Sink | None = None) -> None:
        self.sink = sink
        self.debug_sink = debug_sink
        self.interval = 1.0 / max(1.0, fps)
        self._condition = threading.Condition()
        self._latest: np.ndarray | None = None
        self._latest_debug: np.ndarray | None = None
        self._stopping = False
        self._error: BaseException | None = None
        self.frames_written = 0
        self.first_write_time: float | None = None
        self.last_write_time: float | None = None
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def update(self, frame: np.ndarray, debug_frame: np.ndarray | None = None) -> None:
        with self._condition:
            if self._error:
                raise RuntimeError("Live output writer failed") from self._error
            self._latest = frame
            self._latest_debug = debug_frame
            self._condition.notify_all()

    def _write_loop(self) -> None:
        next_tick = time.monotonic()
        while True:
            with self._condition:
                while self._latest is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._condition.wait(timeout=delay)
                    if self._stopping:
                        return
                    if time.monotonic() < next_tick:
                        continue
                frame, debug_frame = self._latest, self._latest_debug
            try:
                self.sink.write(frame)
                if self.debug_sink is not None and debug_frame is not None:
                    self.debug_sink.write(debug_frame)
                self.frames_written += 1
                written_at = time.monotonic()
                if self.first_write_time is None:
                    self.first_write_time = written_at
                self.last_write_time = written_at
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._stopping = True
                    self._condition.notify_all()
                return
            next_tick += self.interval
            now = time.monotonic()
            if next_tick <= now:
                missed_ticks = int((now - next_tick) / self.interval) + 1
                next_tick += missed_ticks * self.interval

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=10)
        if self._thread.is_alive():
            raise RuntimeError("Live output writer did not stop within 10 seconds")
        error = self._error
        close_error: BaseException | None = None
        for output in (self.debug_sink, self.sink):
            if output is not None:
                try:
                    output.close()
                except BaseException as exc:
                    close_error = close_error or exc
        if error:
            raise RuntimeError("Live output writer failed") from error
        if close_error:
            raise close_error

    @property
    def measured_fps(self) -> float:
        if self.frames_written <= 1 or self.first_write_time is None or self.last_write_time is None:
            return 0.0
        active_duration = self.last_write_time - self.first_write_time + self.interval
        return self.frames_written / active_duration if active_duration > 0 else 0.0


class FfmpegSink:
    """Send BGR frames to FFmpeg (H.264) -> MP4 file or YouTube RTMP."""

    def __init__(self, dest: str, width: int, height: int, fps: float,
                 bitrate: str | None = None, preset: str = "veryfast",
                 audio_source: str | None = None) -> None:
        self.dest = dest
        rtmp = dest.lower().startswith(("rtmp://", "rtmps://"))
        fps_i = max(1, int(round(fps)))
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "-"]
        if audio_source:
            cmd += ["-thread_queue_size", "512", "-i", audio_source]
        if rtmp:
            br = bitrate or "6000k"
            kbps = int(br.lower().rstrip("k"))
            if not audio_source:
                cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
            cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p",
                    "-b:v", br, "-maxrate", br, "-bufsize", f"{2 * kbps}k",
                    "-g", str(fps_i * 2), "-keyint_min", str(fps_i * 2), "-sc_threshold", "0",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                    "-shortest", "-f", "flv", dest]
        else:
            codec = ["-b:v", bitrate] if bitrate else ["-crf", "20"]
            cmd += ["-map", "0:v:0", "-c:v", "libx264", "-preset", preset, "-pix_fmt", "yuv420p", *codec]
            if audio_source:
                cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-shortest"]
            cmd += ["-movflags", "+faststart", dest]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err)

    def write(self, frame: np.ndarray) -> None:
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"FFmpeg exited unexpectedly: {self._stderr()}") from e

    def _stderr(self) -> str:
        self._err.seek(0)
        message = self._err.read().decode(errors="replace").strip()
        for secret in (self.dest, self.audio_source):
            if secret:
                message = message.replace(secret, "<redacted URL>")
        return message

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait()
        if self.proc.returncode not in (0, None):
            raise RuntimeError(f"FFmpeg failed (exit code {self.proc.returncode}): {self._stderr()}")
        self._err.close()


class OpenCvSink:
    """Fallback without FFmpeg (files only)."""

    def __init__(self, dest: str, width: int, height: int, fps: float) -> None:
        self.writer = cv2.VideoWriter(dest, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create output file: {dest}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()


def make_sink(dest: str, width: int, height: int, fps: float,
              bitrate: str | None = None, preset: str = "veryfast",
              audio_source: str | None = None) -> Sink:
    is_rtmp = dest.lower().startswith(("rtmp://", "rtmps://"))
    if shutil.which("ffmpeg"):
        return FfmpegSink(dest, width, height, fps, bitrate, preset, audio_source)
    if is_rtmp:
        raise RuntimeError("FFmpeg not found: it is required for RTMP/YouTube output.")
    if audio_source:
        raise RuntimeError("FFmpeg is required to mux an audio source into the output video.")
    return OpenCvSink(dest, width, height, fps)
