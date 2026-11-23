"""Timeline audio waveform extraction and optional QtMultimedia audio playback."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_audio_waveform(video_path: str, num_bins: int = 400) -> list[float]:
    """Extract normalized mono audio peak amplitudes `[0..1]` for timeline display.

    Returns an empty list if `video_path` has no audio, doesn't exist, or FFmpeg fails.
    """
    if not video_path or not os.path.isfile(video_path):
        return []
    bins = max(16, min(2000, int(num_bins or 400)))
    try:
        from backend.ffmpeg_tools.discovery import require_ffmpeg_install
        ffmpeg = require_ffmpeg_install(require_probe=False).ffmpeg_path
        if not ffmpeg:
            return []
        cmd = [
            ffmpeg,
            "-v", "error",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "8000",
            "-f", "s16le",
            "pipe:1",
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            timeout=12,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if res.returncode != 0 or not res.stdout:
            return []
        raw = res.stdout
        if len(raw) < 4:
            return []
        usable = len(raw) - (len(raw) % 2)
        samples = np.frombuffer(raw[:usable], dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return []
        if samples.size < bins:
            bins = samples.size
        chunks = np.array_split(np.abs(samples), bins)
        peaks = np.array([float(c.max()) if c.size > 0 else 0.0 for c in chunks], dtype=np.float32)
        max_p = float(peaks.max())
        if max_p > 1e-5:
            peaks = np.clip(peaks / max_p, 0.0, 1.0)
        return [round(float(v), 4) for v in peaks]
    except Exception as exc:
        logger.debug("extract_audio_waveform skipped for %s: %s", video_path, exc)
        return []


class TimelineAudioPlayer:
    """Optional QtMultimedia audio player synchronized to timeline scrubbing/playback."""

    def __init__(self, parent=None) -> None:
        self._player = None
        self._audio_out = None
        self._available = False
        self._muted = False
        self._source_path: Optional[str] = None
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            self._audio_out = QAudioOutput(parent)
            self._player = QMediaPlayer(parent)
            self._player.setAudioOutput(self._audio_out)
            self._audio_out.setVolume(0.8)
            self._available = True
        except Exception as exc:
            logger.debug("QtMultimedia unavailable for timeline audio playback: %s", exc)
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def is_muted(self) -> bool:
        return self._muted

    def load(self, video_path: str | None) -> None:
        self._source_path = video_path
        if not self._available or self._player is None:
            return
        try:
            from PySide6.QtCore import QUrl
            self._player.stop()
            if video_path and os.path.isfile(video_path):
                self._player.setSource(QUrl.fromLocalFile(video_path))
            else:
                self._player.setSource(QUrl())
        except Exception:
            pass

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        if self._audio_out is not None:
            try:
                self._audio_out.setMuted(self._muted)
            except Exception:
                pass

    def play_from(self, seconds: float) -> None:
        if not self._available or self._player is None or not self._source_path:
            return
        try:
            self._player.setPosition(max(0, int(round(seconds * 1000.0))))
            self._player.play()
        except Exception:
            pass

    def pause(self) -> None:
        if self._player is not None:
            try:
                self._player.pause()
            except Exception:
                pass

    def stop(self) -> None:
        if self._player is not None:
            try:
                self._player.stop()
            except Exception:
                pass
