"""Video probing via ffprobe."""
from __future__ import annotations

import json
import logging
import os
import subprocess

from .discovery import require_ffmpeg_install

logger = logging.getLogger(__name__)


def probe_video(path: str) -> dict:
    """Probe a video file for metadata.

    Returns dict with keys: fps (float), width (int), height (int),
    frame_count (int), codec (str), duration (float), pix_fmt (str),
    color_range (str), color_space (str), color_primaries (str),
    color_transfer (str), chroma_location (str), bits_per_raw_sample (int).
    Raises RuntimeError if ffprobe fails.
    """
    validation = require_ffmpeg_install(require_probe=True)
    ffprobe = validation.ffprobe_path
    if not ffprobe:
        raise RuntimeError("FFprobe not found")

    cmd = [
        ffprobe,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        path,
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=30,
        encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr[:500]}")

    data = json.loads(result.stdout)

    # Find first video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise RuntimeError(f"No video stream found in {path}")

    def _parse_rate(rate_str: str) -> float:
        if not rate_str or rate_str.upper() == "N/A":
            return 0.0
        try:
            if "/" in rate_str:
                num_s, den_s = rate_str.split("/", 1)
                num_v, den_v = float(num_s), float(den_s)
                return (num_v / den_v) if den_v != 0 else 0.0
            return float(rate_str)
        except (ValueError, TypeError):
            return 0.0

    def _parse_float(val, default: float = 0.0) -> float:
        if val is None or str(val).strip().upper() in ("", "N/A"):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _parse_int(val, default: int = 0) -> int:
        if val is None or str(val).strip().upper() in ("", "N/A"):
            return default
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return default

    # Prefer avg_frame_rate (handles VFR clips without container timebase explosion like 90000/1)
    fps = _parse_rate(str(video_stream.get("avg_frame_rate", "")))
    if fps <= 0.0 or fps > 240.0:
        r_fps = _parse_rate(str(video_stream.get("r_frame_rate", "")))
        if 0.0 < r_fps <= 240.0:
            fps = r_fps
    if fps <= 0.0:
        fps = 24.0

    # Frame count: prefer nb_frames, fall back to duration * fps
    frame_count = _parse_int(video_stream.get("nb_frames"), 0)

    duration = _parse_float(video_stream.get("duration"), 0.0) or _parse_float(
        data.get("format", {}).get("duration"), 0.0
    )
    if frame_count <= 0 and duration > 0:
        frame_count = int(duration * fps)

    return {
        "fps": round(fps, 4),
        "width": _parse_int(video_stream.get("width"), 0),
        "height": _parse_int(video_stream.get("height"), 0),
        "frame_count": frame_count,
        "codec": video_stream.get("codec_name", "unknown"),
        "duration": duration,
        # Color metadata for building explicit conversion filters
        "pix_fmt": video_stream.get("pix_fmt", ""),
        "color_space": video_stream.get("color_space", ""),
        "color_primaries": video_stream.get("color_primaries", ""),
        "color_transfer": video_stream.get("color_transfer", ""),
        "color_range": video_stream.get("color_range", ""),
        "chroma_location": video_stream.get("chroma_location", ""),
        "bits_per_raw_sample": _parse_int(video_stream.get("bits_per_raw_sample"), 0),
    }


def probe_audio_and_timecode(path: str) -> dict:
    """Probe optional audio stream and reel timecode metadata via ffprobe.

    Returns ``{"has_audio": bool, "audio_codec": str, "timecode": str | None}``.
    Never raises on missing ffprobe or unreadable files.
    """
    info: dict = {"has_audio": False, "audio_codec": "", "timecode": None}
    if not path or not os.path.isfile(path):
        return info
    try:
        validation = require_ffmpeg_install(require_probe=True)
        ffprobe = validation.ffprobe_path
        if not ffprobe:
            return info
        cmd = [
            ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        if res.returncode != 0 or not res.stdout:
            return info
        data = json.loads(res.stdout)
        for s in data.get("streams", []):
            if s.get("codec_type") == "audio" and not info["has_audio"]:
                info["has_audio"] = True
                info["audio_codec"] = str(s.get("codec_name") or "")
        # Search format tags then stream tags for timecode
        fmt_tags = (data.get("format") or {}).get("tags") or {}
        for k, v in fmt_tags.items():
            if str(k).lower() == "timecode" and str(v).strip():
                info["timecode"] = str(v).strip()
                break
        if not info["timecode"]:
            for s in data.get("streams", []):
                tags = s.get("tags") or {}
                for k, v in tags.items():
                    if str(k).lower() == "timecode" and str(v).strip():
                        info["timecode"] = str(v).strip()
                        break
                if info["timecode"]:
                    break
    except Exception as exc:
        logger.debug("probe_audio_and_timecode failed for %s: %s", path, exc)
    return info
