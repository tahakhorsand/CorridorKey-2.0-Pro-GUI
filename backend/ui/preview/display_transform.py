"""Display transform module for preview rendering.



Handles conversion of VFX image data to display-ready 8-bit sRGB QImage.

Supports EXR (float16/32, 1ch/3ch/4ch), PNG, and video frames.



Codex critical finding: naive clip-to-8bit looks wrong for:

- Linear EXR (needs linear→sRGB gamma)

- Matte (1ch → grayscale visualization)

- Processed (straight RGBA → composite with alpha for display)

- Negative/HDR values (needs clamping before conversion)

"""

from __future__ import annotations



import os

import logging

import threading

from collections import OrderedDict

from functools import lru_cache



os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")



import cv2

import numpy as np

from PySide6.QtGui import QImage



from .frame_index import ViewMode

from backend.frame_io import imread_unicode, open_video



logger = logging.getLogger(__name__)



# LRU cache for decoded QImages (expanded to 128 frames for smooth scrubbing & playback)

# Lock protects concurrent access from thread pool workers

_QIMAGE_CACHE: OrderedDict[str, QImage] = OrderedDict()

_CACHE_LOCK = threading.Lock()

_CACHE_MAX = 320



# Cached open VideoCapture instances to eliminate repeated open/close overhead per frame

_CAP_CACHE: OrderedDict[str, cv2.VideoCapture] = OrderedDict()

_CAP_LOCK = threading.Lock()





def _cache_key(
    path: str,
    mode: ViewMode,
    input_exr_is_linear: bool = False,
    backdrop: str = "checker",
) -> str:

    """Cache key combining path, mode, and compositing backdrop."""

    b = (backdrop or "checker").lower() if mode in (ViewMode.COMP, ViewMode.PROCESSED) else "default"

    return f"{mode.value}:{int(input_exr_is_linear)}:{b}:{path}"


BACKDROP_MODES: tuple[str, ...] = ("checker", "black", "white", "gray", "magenta", "custom")


_CUSTOM_PLATE_LOCK = threading.Lock()
_CUSTOM_PLATE_RGB_F32: np.ndarray | None = None
_CUSTOM_PLATE_PATH: str | None = None
_CUSTOM_PLATE_VIDEO_PATH: str | None = None
_CUSTOM_PLATE_VIDEO_COUNT: int = 0
_CUSTOM_PLATE_FRAME_CACHE: OrderedDict[int, np.ndarray] = OrderedDict()


def set_custom_backdrop_plate(source: str | np.ndarray | None) -> bool:
    """Load and cache a custom background plate (image file, video sequence, or RGB array)."""
    global _CUSTOM_PLATE_RGB_F32, _CUSTOM_PLATE_PATH, _CUSTOM_PLATE_VIDEO_PATH, _CUSTOM_PLATE_VIDEO_COUNT
    if source is None:
        with _CUSTOM_PLATE_LOCK:
            _CUSTOM_PLATE_RGB_F32 = None
            _CUSTOM_PLATE_PATH = None
            _CUSTOM_PLATE_VIDEO_PATH = None
            _CUSTOM_PLATE_VIDEO_COUNT = 0
            _CUSTOM_PLATE_FRAME_CACHE.clear()
        clear_cache()
        return True

    plate_f32: np.ndarray | None = None
    path_str: str | None = None
    video_path_str: str | None = None
    video_count = 0
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[..., :3]
            if rgb.dtype == np.uint8:
                plate_f32 = rgb.astype(np.float32) * (1.0 / 255.0)
            elif rgb.dtype == np.uint16:
                plate_f32 = rgb.astype(np.float32) * (1.0 / 65535.0)
            else:
                plate_f32 = np.clip(rgb.astype(np.float32, copy=False), 0.0, 1.0)
            path_str = "<array>"
    elif isinstance(source, str) and os.path.isfile(source):
        path_str = source
        ext = os.path.splitext(source)[1].lower()
        if ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".mxf"):
            cap = open_video(source)
            if cap is not None:
                try:
                    video_count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
                    ok, bgr = cap.read()
                    if ok and bgr is not None:
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        plate_f32 = rgb.astype(np.float32) * (1.0 / 255.0)
                        video_path_str = source
                finally:
                    cap.release()
        else:
            img = imread_unicode(source, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 2:
                    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] >= 4:
                    bgr = img[:, :, :3]
                else:
                    bgr = img
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if rgb.dtype == np.uint8:
                    plate_f32 = rgb.astype(np.float32) * (1.0 / 255.0)
                elif rgb.dtype == np.uint16:
                    plate_f32 = rgb.astype(np.float32) * (1.0 / 65535.0)
                else:
                    plate_f32 = np.clip(rgb.astype(np.float32), 0.0, 1.0)

    if plate_f32 is None:
        return False

    with _CUSTOM_PLATE_LOCK:
        _CUSTOM_PLATE_RGB_F32 = np.ascontiguousarray(np.clip(plate_f32, 0.0, 1.0))
        _CUSTOM_PLATE_PATH = path_str
        _CUSTOM_PLATE_VIDEO_PATH = video_path_str
        _CUSTOM_PLATE_VIDEO_COUNT = video_count
        _CUSTOM_PLATE_FRAME_CACHE.clear()
        if video_path_str is not None:
            _CUSTOM_PLATE_FRAME_CACHE[0] = _CUSTOM_PLATE_RGB_F32
    clear_cache()
    return True


def _get_custom_plate_at_frame(frame_index: int) -> np.ndarray | None:
    with _CUSTOM_PLATE_LOCK:
        vpath = _CUSTOM_PLATE_VIDEO_PATH
        vcount = _CUSTOM_PLATE_VIDEO_COUNT
        fallback = _CUSTOM_PLATE_RGB_F32
        if not vpath or vcount <= 1:
            return fallback
        idx = int(max(0, frame_index)) % vcount
        cached = _CUSTOM_PLATE_FRAME_CACHE.get(idx)
        if cached is not None:
            _CUSTOM_PLATE_FRAME_CACHE.move_to_end(idx)
            return cached

    # Decode frame outside _CUSTOM_PLATE_LOCK to avoid blocking reader threads
    decoded: np.ndarray | None = None
    cap = open_video(vpath)
    if cap is not None:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                decoded = np.ascontiguousarray(np.clip(rgb.astype(np.float32) * (1.0 / 255.0), 0.0, 1.0))
        finally:
            cap.release()

    if decoded is not None:
        with _CUSTOM_PLATE_LOCK:
            _CUSTOM_PLATE_FRAME_CACHE[idx] = decoded
            _CUSTOM_PLATE_FRAME_CACHE.move_to_end(idx)
            while len(_CUSTOM_PLATE_FRAME_CACHE) > 32:
                _CUSTOM_PLATE_FRAME_CACHE.popitem(last=False)
        return decoded
    return fallback


def get_custom_backdrop_path() -> str | None:
    with _CUSTOM_PLATE_LOCK:
        return _CUSTOM_PLATE_PATH


def build_backdrop_rgb(height: int, width: int, backdrop: str = "checker", *, frame_index: int = 0) -> np.ndarray:

    """Return [H, W, 3] float32 sRGB backdrop image in [0, 1]."""

    b = (backdrop or "checker").lower()

    if b == "custom":
        plate = _get_custom_plate_at_frame(frame_index)
        if plate is not None and plate.shape[0] > 0 and plate.shape[1] > 0:
            if plate.shape[:2] == (height, width):
                return plate
            return cv2.resize(plate, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        b = "checker"

    if b == "black":

        return np.zeros((height, width, 3), dtype=np.float32)

    if b == "white":

        return np.ones((height, width, 3), dtype=np.float32)

    if b == "gray":

        return np.full((height, width, 3), 0.5, dtype=np.float32)

    if b == "magenta":

        bg = np.empty((height, width, 3), dtype=np.float32)

        bg[:, :, 0] = 1.0

        bg[:, :, 1] = 0.0

        bg[:, :, 2] = 1.0

        return bg

    tile = 16

    y_idx = (np.arange(height, dtype=np.int32) // tile)[:, None]

    x_idx = (np.arange(width, dtype=np.int32) // tile)[None, :]

    dark = ((y_idx + x_idx) & 1) == 0

    bg = np.where(dark[:, :, None], np.float32(0.24), np.float32(0.36))

    return np.broadcast_to(bg, (height, width, 3)).astype(np.float32)


def composite_over_backdrop(
    fg_srgb: np.ndarray,
    alpha_2d: np.ndarray,
    backdrop: str = "checker",
    *,
    frame_index: int = 0,
) -> np.ndarray:

    """Composite sRGB float foreground [H, W, 3] over selected backdrop using alpha [H, W]."""

    fg = np.clip(fg_srgb.astype(np.float32), 0.0, 1.0)

    a = alpha_2d.astype(np.float32)

    if a.ndim == 2:

        a = a[:, :, None]

    elif a.ndim == 3 and a.shape[2] > 1:

        a = a[:, :, :1]

    a = np.clip(a, 0.0, 1.0)

    h, w = fg.shape[:2]

    bg = build_backdrop_rgb(h, w, backdrop, frame_index=frame_index)

    comp = fg * a + bg * (1.0 - a)

    return (np.clip(comp, 0.0, 1.0) * 255.0).astype(np.uint8)


def apply_inspection_filter(
    qimg: QImage,
    mode: str = "rgb",
    alpha_arr=None,
) -> QImage:
    """Apply viewer-only channel/status/rubylith inspection and optional Display LUT."""
    if qimg is None or qimg.isNull():
        return qimg
    fmt_img = qimg if qimg.format() == QImage.Format_RGB888 else qimg.convertToFormat(QImage.Format_RGB888)
    w, h = fmt_img.width(), fmt_img.height()
    bpl = fmt_img.bytesPerLine()
    ptr = fmt_img.bits()
    buf = np.frombuffer(ptr, dtype=np.uint8, count=h * bpl).reshape((h, bpl))
    rgb = np.ascontiguousarray(buf[:, : w * 3].reshape((h, w, 3)))

    from backend.color_management import apply_display_lut_rgb
    rgb = apply_display_lut_rgb(rgb)

    m = (mode or "rgb").strip().lower()
    if m == "rgb":
        out = rgb
    elif m == "r":
        ch = rgb[:, :, 0]
        out = np.stack([ch, ch, ch], axis=2)
    elif m == "g":
        ch = rgb[:, :, 1]
        out = np.stack([ch, ch, ch], axis=2)
    elif m == "b":
        ch = rgb[:, :, 2]
        out = np.stack([ch, ch, ch], axis=2)
    elif m in ("status", "rubylith"):
        if alpha_arr is not None:
            a = np.asarray(alpha_arr, dtype=np.float32)
            if a.ndim == 3:
                a = a[:, :, 0]
            if a.dtype.kind in ("u", "i"):
                a = a / 255.0
            if a.shape[:2] != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
            a = np.clip(a, 0.0, 1.0)
        else:
            # Fallback to normalized grayscale intensity when alpha array not cached
            a = np.mean(rgb.astype(np.float32) / 255.0, axis=2)
        if m == "status":
            out = np.zeros((h, w, 3), dtype=np.uint8)
            out[a >= 0.985] = [255, 255, 255]
            semi = (a > 0.015) & (a < 0.985)
            out[semi] = [255, 80, 60]
        else:
            inv = (1.0 - a)[:, :, None] * 0.60
            ruby = np.array([210.0, 20.0, 20.0], dtype=np.float32)
            comp = rgb.astype(np.float32) * (1.0 - inv) + ruby * inv
            out = np.clip(comp, 0.0, 255.0).astype(np.uint8)
    else:
        out = rgb

    out_c = np.ascontiguousarray(out)
    return QImage(out_c.data, w, h, w * 3, QImage.Format_RGB888).copy()





def clear_cache() -> None:

    """Clear the frame cache and release open video handles (call on clip switch)."""

    with _CACHE_LOCK:

        _QIMAGE_CACHE.clear()

    with _CAP_LOCK:

        for cap in _CAP_CACHE.values():

            try:

                cap.release()

            except Exception:

                pass

        _CAP_CACHE.clear()





def _get_video_cap(video_path: str) -> cv2.VideoCapture | None:

    """Retrieve or create an open VideoCapture instance with LRU eviction."""

    if video_path in _CAP_CACHE:

        _CAP_CACHE.move_to_end(video_path)

        cap = _CAP_CACHE[video_path]

        if cap.isOpened():

            return cap

    cap = open_video(video_path)

    if not cap.isOpened():

        return None

    _CAP_CACHE[video_path] = cap

    while len(_CAP_CACHE) > 4:

        _, old_cap = _CAP_CACHE.popitem(last=False)

        try:

            old_cap.release()

        except Exception:

            pass

    return cap





def decode_frame(

    path: str,

    mode: ViewMode,

    *,

    input_exr_is_linear: bool = False,

    backdrop: str = "checker",

) -> QImage | None:

    """Decode a frame file to a display-ready QImage.



    Applies mode-specific display transforms:

    - COMP: 8-bit sRGB PNG/EXR, or dynamically composited over custom backdrop

    - INPUT: Uses the current source interpretation for EXR, PNG, JPG, etc.

    - ALPHA: Matte visualization from AlphaHint image frames

    - FG: Linear float EXR → sRGB gamma

    - MATTE: 1-channel float → grayscale visualization

    - PROCESSED: Straight RGBA float → composite over selected backdrop for display



    Returns None if the file can't be read.

    """

    key = _cache_key(path, mode, input_exr_is_linear=input_exr_is_linear, backdrop=backdrop)

    with _CACHE_LOCK:

        if key in _QIMAGE_CACHE:

            _QIMAGE_CACHE.move_to_end(key)

            return _QIMAGE_CACHE[key]



    qimg = _do_decode(path, mode, input_exr_is_linear=input_exr_is_linear, backdrop=backdrop)

    if qimg is not None:

        with _CACHE_LOCK:

            _QIMAGE_CACHE[key] = qimg

            while len(_QIMAGE_CACHE) > _CACHE_MAX:

                _QIMAGE_CACHE.popitem(last=False)



    return qimg





def _try_decode_comp_with_backdrop(path: str, backdrop: str) -> QImage | None:

    """Composite FG + Matte (or Processed) over a non-checker backdrop for COMP view."""

    comp_dir = os.path.dirname(path)

    out_root = os.path.dirname(comp_dir)

    stem = os.path.splitext(os.path.basename(path))[0]

    fg_path = next(
        (os.path.join(out_root, "FG", f"{stem}.{ext}") for ext in ("exr", "png", "tif", "tiff")
         if os.path.isfile(os.path.join(out_root, "FG", f"{stem}.{ext}"))),
        None,
    )

    matte_path = next(
        (os.path.join(out_root, "Matte", f"{stem}.{ext}") for ext in ("exr", "png", "tif", "tiff")
         if os.path.isfile(os.path.join(out_root, "Matte", f"{stem}.{ext}"))),
        None,
    )

    if fg_path and matte_path:

        fg_raw = imread_unicode(fg_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        matte_raw = imread_unicode(matte_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        if fg_raw is not None and matte_raw is not None:

            fg_scale = 65535.0 if fg_raw.dtype == np.uint16 else (255.0 if fg_raw.dtype == np.uint8 else 1.0)

            fg_f = fg_raw.astype(np.float32) / fg_scale

            if fg_f.ndim == 2:

                fg_rgb = np.stack([fg_f, fg_f, fg_f], axis=2)

            elif fg_f.shape[2] == 4:

                fg_rgb = cv2.cvtColor(fg_f, cv2.COLOR_BGRA2RGB)

            else:

                fg_rgb = cv2.cvtColor(fg_f, cv2.COLOR_BGR2RGB)

            if fg_path.lower().endswith(".exr"):

                fg_rgb = _linear_to_srgb(np.clip(fg_rgb, 0.0, 1.0))

            else:

                fg_rgb = np.clip(fg_rgb, 0.0, 1.0)

            m_scale = 65535.0 if matte_raw.dtype == np.uint16 else (255.0 if matte_raw.dtype == np.uint8 else 1.0)

            alpha_f = matte_raw.astype(np.float32) / m_scale

            if alpha_f.ndim == 3:

                alpha_f = alpha_f[:, :, 3] if alpha_f.shape[2] == 4 else alpha_f[:, :, 0]

            if alpha_f.shape[:2] != fg_rgb.shape[:2]:

                alpha_f = cv2.resize(alpha_f, (fg_rgb.shape[1], fg_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            import re as _re
            m_fr = _re.search(r"(\d+)$", stem)
            f_idx = int(m_fr.group(1)) if m_fr else 0
            return _numpy_to_qimage(composite_over_backdrop(fg_rgb, alpha_f, backdrop, frame_index=f_idx))

    proc_path = next(
        (os.path.join(out_root, "Processed", f"{stem}.{ext}") for ext in ("exr", "png")
         if os.path.isfile(os.path.join(out_root, "Processed", f"{stem}.{ext}"))),
        None,
    )

    if proc_path:

        if proc_path.lower().endswith(".exr"):

            return _decode_exr(proc_path, ViewMode.PROCESSED, backdrop=backdrop)

        return _decode_ldr(proc_path, ViewMode.PROCESSED, backdrop=backdrop)

    return None


def _do_decode(
    path: str,
    mode: ViewMode,
    *,
    input_exr_is_linear: bool = False,
    backdrop: str = "checker",
) -> QImage | None:

    """Actual decode + transform logic."""

    if not os.path.isfile(path):

        return None



    is_exr = path.lower().endswith('.exr')



    try:

        if mode == ViewMode.COMP and (backdrop or "checker").lower() != "checker":

            custom = _try_decode_comp_with_backdrop(path, backdrop)

            if custom is not None:

                return custom

        if is_exr:

            return _decode_exr(path, mode, input_exr_is_linear=input_exr_is_linear, backdrop=backdrop)

        else:

            return _decode_ldr(path, mode, input_exr_is_linear=input_exr_is_linear, backdrop=backdrop)

    except Exception as e:

        logger.warning(f"Failed to decode frame {path}: {e}")

        return None





def _decode_ldr(

    path: str,

    mode: ViewMode,

    *,

    input_exr_is_linear: bool = False,

    backdrop: str = "checker",

) -> QImage | None:

    """Decode an 8-bit/16-bit image (PNG/JPG/etc) to QImage."""

    if mode == ViewMode.PROCESSED:

        raw = imread_unicode(path, cv2.IMREAD_UNCHANGED)

        if raw is not None and raw.ndim == 3 and raw.shape[2] == 4:

            scale = 65535.0 if raw.dtype == np.uint16 else (255.0 if raw.dtype == np.uint8 else 1.0)

            bgra = raw.astype(np.float32) / scale

            fg_rgb = cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2RGB)

            alpha = bgra[:, :, 3]

            return _numpy_to_qimage(composite_over_backdrop(fg_rgb, alpha, backdrop))

    img = imread_unicode(path, cv2.IMREAD_COLOR)

    if img is None:

        return None

    if mode == ViewMode.INPUT and input_exr_is_linear:

        return _transform_linear_rgb(

            img.astype(np.float32) / 255.0,

            ViewMode.INPUT,

            input_exr_is_linear=True,

        )

    # BGR → RGB

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return _numpy_to_qimage(rgb)





def _decode_exr(
    path: str,
    mode: ViewMode,
    *,
    input_exr_is_linear: bool = False,
    backdrop: str = "checker",
) -> QImage | None:

    """Decode an EXR file with mode-specific display transform."""

    img = imread_unicode(path, cv2.IMREAD_UNCHANGED)

    if img is None:

        return None



    # Determine channel layout

    if img.ndim == 2:

        # Single channel (typical for Matte)

        return _transform_matte(img)

    elif img.ndim == 3:

        channels = img.shape[2]

        if channels == 1:

            return _transform_matte(img[:, :, 0])

        elif channels == 2:

            # Rare: treat first channel as matte

            return _transform_matte(img[:, :, 0])

        elif channels == 3:

            if mode in (ViewMode.ALPHA, ViewMode.MASK, ViewMode.MATTE):

                return _transform_matte(img[:, :, 0])

            # BGR float — FG or Input

            return _transform_linear_rgb(img, mode, input_exr_is_linear=input_exr_is_linear)

        elif channels == 4:

            if mode in (ViewMode.ALPHA, ViewMode.MASK, ViewMode.MATTE):

                return _transform_matte(img[:, :, 0])

            # BGRA float — Processed (straight RGBA)

            if mode == ViewMode.PROCESSED:

                return _transform_processed_rgba(img, backdrop=backdrop)

            else:

                # Strip alpha, treat as RGB

                return _transform_linear_rgb(

                    img[:, :, :3], mode, input_exr_is_linear=input_exr_is_linear,

                )

        else:

            # Unexpected channel count, take first 3

            return _transform_linear_rgb(

                img[:, :, :3], mode, input_exr_is_linear=input_exr_is_linear,

            )

    return None





def _transform_matte(data: np.ndarray) -> QImage:

    """Single-channel float → grayscale visualization.



    Clamp to [0,1], apply slight gamma for better visualization of

    subtle alpha edges, convert to 8-bit grayscale displayed as RGB.

    """

    clamped = np.clip(data.astype(np.float32), 0.0, 1.0)

    # Light gamma lift for matte visualization (makes edges visible)

    display = np.power(clamped, 0.85)

    gray8 = (display * 255.0).astype(np.uint8)

    # Stack to 3-channel for consistent QImage format

    rgb = np.stack([gray8, gray8, gray8], axis=2)

    return _numpy_to_qimage(rgb)





def _transform_linear_rgb(

    bgr: np.ndarray,

    mode: ViewMode,

    *,

    input_exr_is_linear: bool = False,

) -> QImage:

    """Float BGR → sRGB 8-bit RGB.



    INPUT mode: extracted video EXRs are already display-encoded float, while

    standalone EXR plates are true linear-light and need gamma.

    FG mode: Inference output is linear float — apply linear→sRGB gamma.

    """

    # Clamp negatives

    clamped = np.clip(bgr.astype(np.float32), 0.0, None)



    if mode == ViewMode.INPUT and not input_exr_is_linear:

        # Extracted frames are display-encoded float (FFmpeg writes video

        # values as-is into EXR). Just clamp to [0,1] — no gamma.

        display = np.clip(clamped, 0.0, 1.0)

    else:

        # Linear EXR input and inference outputs both need gamma for display.

        # Clamp to [0,1] — stray super-whites from float rounding should not

        # trigger Reinhard tone mapping on the entire image.

        clamped = np.clip(clamped, 0.0, 1.0)

        display = _linear_to_srgb(clamped)



    # BGR → RGB

    rgb = cv2.cvtColor((display * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)

    return _numpy_to_qimage(rgb)





def _transform_processed_rgba(bgra: np.ndarray, backdrop: str = "checker") -> QImage:

    """Straight linear BGRA float → composite over selected backdrop for display."""

    bgra_f = bgra.astype(np.float32)

    bgr = np.clip(bgra_f[:, :, :3], 0.0, None)

    alpha = np.clip(bgra_f[:, :, 3], 0.0, 1.0)

    if (backdrop or "checker").lower() == "black":

        composite = np.clip(bgr * alpha[:, :, None], 0.0, 1.0)

        srgb = _linear_to_srgb(composite)

        rgb = cv2.cvtColor((srgb * 255.0).astype(np.uint8), cv2.COLOR_BGR2RGB)

        return _numpy_to_qimage(rgb)

    fg_rgb_srgb = _linear_to_srgb(np.clip(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), 0.0, 1.0))

    return _numpy_to_qimage(composite_over_backdrop(fg_rgb_srgb, alpha, backdrop))





def processed_rgba_to_qimage(rgba: np.ndarray, backdrop: str = "checker") -> QImage:

    """Render in-memory straight linear RGBA preview composited over selected backdrop."""

    rgba_f = rgba.astype(np.float32)

    rgb = np.clip(rgba_f[:, :, :3], 0.0, None)

    alpha = np.clip(rgba_f[:, :, 3], 0.0, 1.0)

    if (backdrop or "checker").lower() == "black":

        composite = np.clip(rgb * alpha[:, :, None], 0.0, 1.0)

        srgb = _linear_to_srgb(composite)

        return _numpy_to_qimage((srgb * 255.0).astype(np.uint8))

    fg_srgb = _linear_to_srgb(np.clip(rgb, 0.0, 1.0))

    return _numpy_to_qimage(composite_over_backdrop(fg_srgb, alpha, backdrop))





def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:

    """Apply sRGB gamma curve to linear float data."""

    linear = np.clip(linear, 0.0, 1.0)

    mask = linear <= 0.0031308

    return np.where(mask, linear * 12.92, 1.055 * np.power(linear, 1.0 / 2.4) - 0.055)





def _numpy_to_qimage(rgb: np.ndarray) -> QImage:

    """Convert an RGB uint8 numpy array to QImage.



    Makes a copy of the data so the numpy array can be freed.

    """

    h, w, ch = rgb.shape

    assert ch == 3, f"Expected 3 channels, got {ch}"

    # Ensure contiguous

    rgb = np.ascontiguousarray(rgb)

    bytes_per_line = w * 3

    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

    # QImage doesn't copy by default — force a deep copy

    return qimg.copy()





def decode_video_frame(

    video_path: str,

    frame_index: int,

    *,

    input_exr_is_linear: bool = False,

) -> QImage | None:

    """Decode a single frame from a video file.



    Uses cv2.VideoCapture with seek. Expensive for random access.

    Results are cached in the LRU cache.

    """

    key = _cache_key(

        f"video:{video_path}:{frame_index}",

        ViewMode.INPUT,

        input_exr_is_linear=input_exr_is_linear,

    )

    with _CACHE_LOCK:

        if key in _QIMAGE_CACHE:

            _QIMAGE_CACHE.move_to_end(key)

            return _QIMAGE_CACHE[key]



    try:

        with _CAP_LOCK:

            cap = _get_video_cap(video_path)

            if cap is None or not cap.isOpened():

                return None

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            ret, frame = cap.read()

        if not ret or frame is None:

            return None

        if input_exr_is_linear:

            qimg = _transform_linear_rgb(

                frame.astype(np.float32) / 255.0,

                ViewMode.INPUT,

                input_exr_is_linear=True,

            )

        else:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            qimg = _numpy_to_qimage(rgb)



        with _CACHE_LOCK:

            _QIMAGE_CACHE[key] = qimg

            while len(_QIMAGE_CACHE) > _CACHE_MAX:

                _QIMAGE_CACHE.popitem(last=False)



        return qimg

    except Exception as e:

        logger.warning(f"Failed to decode video frame {frame_index} from {video_path}: {e}")

        return None



def get_cached_qimage(
    path: str,
    mode: ViewMode,
    input_exr_is_linear: bool = False,
    backdrop: str = "checker",
) -> QImage | None:
    """Synchronous cache probe for already decoded images (0ms latency)."""
    key = _cache_key(path, mode, input_exr_is_linear=input_exr_is_linear, backdrop=backdrop)
    with _CACHE_LOCK:
        if key in _QIMAGE_CACHE:
            _QIMAGE_CACHE.move_to_end(key)
            return _QIMAGE_CACHE[key]
    return None


def get_cached_video_qimage(video_path: str, frame_index: int, input_exr_is_linear: bool = False) -> QImage | None:
    """Synchronous cache probe for already decoded video frames (0ms latency)."""
    key = _cache_key(
        f"video:{video_path}:{frame_index}",
        ViewMode.INPUT,
        input_exr_is_linear=input_exr_is_linear,
    )
    with _CACHE_LOCK:
        if key in _QIMAGE_CACHE:
            _QIMAGE_CACHE.move_to_end(key)
            return _QIMAGE_CACHE[key]
    return None
