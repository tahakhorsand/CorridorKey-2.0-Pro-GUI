"""Video stitching from image sequences via FFmpeg."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
from typing import Callable, Optional

from .discovery import require_ffmpeg_install
from .probe import probe_audio_and_timecode

logger = logging.getLogger(__name__)


def build_stitch_command(
    ffmpeg: str,
    in_dir: str,
    out_path: str,
    *,
    fps: float = 24.0,
    pattern: str = "frame_%06d.png",
    codec: str = "libx264",
    crf: int = 18,
    start_number: int = 0,
    source_video_path: Optional[str] = None,
    start_frame_offset: int = 0,
    timecode: Optional[str] = None,
    preserve_audio: bool = True,
) -> list[str]:
    """Build the FFmpeg command for stitching image sequences with optional audio & timecode."""
    ext = os.path.splitext(out_path)[1].lower()
    is_webm = ext == ".webm"
    req_codec = (codec or "libx264").lower()
    is_prores_422hq = req_codec in ("prores_422hq", "prores_hq")
    is_prores_4444 = (
        req_codec in ("prores_ks", "prores_4444")
        or (ext == ".mov" and not is_prores_422hq)
    )
    is_exr = pattern.lower().endswith(".exr")

    if is_prores_422hq:
        ff_codec = "prores_ks"
        pix_fmt = "yuv422p10le"
    elif is_prores_4444:
        ff_codec = "prores_ks"
        pix_fmt = "yuva444p10le"
    elif is_webm:
        ff_codec = "libvpx-vp9"
        pix_fmt = "yuva420p"
    else:
        ff_codec = codec if req_codec not in ("h264",) else "libx264"
        pix_fmt = "yuv420p"

    has_audio = False
    resolved_tc = timecode
    if source_video_path and os.path.isfile(source_video_path):
        probed = probe_audio_and_timecode(source_video_path)
        has_audio = bool(preserve_audio and probed.get("has_audio", False))
        if not resolved_tc:
            resolved_tc = probed.get("timecode")

    seq_input = os.path.join(in_dir, pattern).replace("\\", "/")
    cmd: list[str] = [ffmpeg, "-framerate", str(fps), "-start_number", str(start_number)]
    if is_exr:
        cmd.extend(["-apply_trc", "iec61966_2_1"])
    cmd.extend(["-i", seq_input])

    if has_audio and source_video_path:
        ss_sec = max(0.0, float(max(0, int(start_frame_offset))) / max(1e-3, float(fps)))
        if ss_sec > 0.0:
            cmd.extend(["-ss", f"{ss_sec:.6f}"])
        cmd.extend(["-i", source_video_path, "-map", "0:v:0", "-map", "1:a:0?"])

    cmd.extend(["-c:v", ff_codec])
    if is_prores_422hq:
        cmd.extend(["-profile:v", "3", "-qscale:v", "9", "-vendor", "apl0"])
    elif is_prores_4444:
        cmd.extend(["-profile:v", "4444", "-qscale:v", "9", "-vendor", "apl0"])
    else:
        cmd.extend(["-crf", str(crf)])
    cmd.extend(["-pix_fmt", pix_fmt])
    if is_webm:
        cmd.extend(["-b:v", "0", "-auto-alt-ref", "0", "-deadline", "good", "-row-mt", "1"])

    if has_audio:
        audio_codec = "libopus" if is_webm else "aac"
        cmd.extend(["-c:a", audio_codec, "-shortest"])
    else:
        cmd.append("-an")

    if resolved_tc:
        cmd.extend(["-timecode", str(resolved_tc)])

    cmd.extend([out_path, "-y"])
    return cmd


def stitch_video(
    in_dir: str,
    out_path: str,
    fps: float = 24.0,
    pattern: str = "frame_%06d.png",
    codec: str = "libx264",
    crf: int = 18,
    start_number: int = 0,
    on_progress: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    *,
    source_video_path: Optional[str] = None,
    start_frame_offset: int = 0,
    timecode: Optional[str] = None,
    preserve_audio: bool = True,
) -> None:
    """Stitch image sequence back into a video file with optional audio passthrough and timecode.

    Raises:
        RuntimeError if ffmpeg is not found or stitching fails.
    """
    validation = require_ffmpeg_install(require_probe=False)
    ffmpeg = validation.ffmpeg_path
    if not ffmpeg:
        raise RuntimeError("FFmpeg not found")

    total_frames = len([f for f in os.listdir(in_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.exr'))])

    cmd = build_stitch_command(
        ffmpeg,
        in_dir,
        out_path,
        fps=fps,
        pattern=pattern,
        codec=codec,
        crf=crf,
        start_number=start_number,
        source_video_path=source_video_path,
        start_frame_offset=start_frame_offset,
        timecode=timecode,
        preserve_audio=preserve_audio,
    )

    logger.info(f"Stitching video: {' '.join(cmd)}")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        text=True,
        encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )

    frame_re = re.compile(r"frame=\s*(\d+)")
    stderr_lines: list[str] = []
    cancelled = False

    try:
        for line in proc.stderr:
            stderr_lines.append(line.rstrip())

            if cancel_event and cancel_event.is_set():
                cancelled = True
                try:
                    if proc.stdin:
                        proc.stdin.write("q\n")
                        proc.stdin.flush()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                logger.info("Stitching cancelled")
                return

            match = frame_re.search(line)
            if match:
                current = int(match.group(1))
                if on_progress and total_frames > 0:
                    on_progress(current, total_frames)

        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        if not cancelled and not (cancel_event and cancel_event.is_set()):
            raise RuntimeError("FFmpeg stitching timed out")
    finally:
        for stream in (proc.stdin, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

    if proc.returncode != 0 and not cancelled and not (cancel_event and cancel_event.is_set()):
        # Last few stderr lines usually contain the actual error
        tail = "\n".join(stderr_lines[-10:]) if stderr_lines else "(no output)"
        logger.error(f"FFmpeg stitching failed (exit {proc.returncode}):\n{tail}")
        raise RuntimeError(f"FFmpeg stitching failed:\n{tail}")

    logger.info(f"Video stitched: {out_path}")
