"""Professional Color Management: Cinema Log Transforms & 1D/3D .cube Display LUTs."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

LOG_PROFILES: tuple[str, ...] = ("none", "slog3", "logc3", "log3g10", "acescg")

# Bradford-adapted ACES AP1 (ACEScg) <-> Linear sRGB/Rec.709 3x3 conversion matrices
_ACESCG_TO_REC709 = np.array(
    [
        [1.70505099, -0.62179212, -0.08325887],
        [-0.13025642, 1.14080474, -0.01054832],
        [-0.02400336, -0.12896898, 1.15297234],
    ],
    dtype=np.float32,
)
_REC709_TO_ACESCG = np.linalg.inv(_ACESCG_TO_REC709).astype(np.float32)


def log_to_linear(rgb: np.ndarray, profile: str) -> np.ndarray:
    """Convert cinema Log / wide-gamut float RGB to linear Rec.709.

    When `profile` is `'none'` (or empty/unrecognized), returns input unchanged.
    """
    prof = (profile or "none").strip().lower()
    if prof in ("", "none"):
        return np.asarray(rgb, dtype=np.float32)

    x = np.asarray(rgb, dtype=np.float32)
    if prof == "slog3":
        cut = 171.2102946929 / 1023.0
        lin = np.where(
            x >= cut,
            np.power(10.0, (x * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
            (x * 1023.0 - 95.0) * 0.01125000 / (171.2102946929 - 95.0),
        )
        return np.maximum(lin, 0.0).astype(np.float32, copy=False)

    if prof == "logc3":
        # ARRI LogC3 EI 800 AWG v3
        cut = 0.010591 * 5.367655 + 0.092809  # ~0.149658
        e = 5.367655
        f = 0.092809
        c = 0.247190
        d = 0.385537
        a = 5.555556
        b = 0.052272
        lin = np.where(
            x > cut,
            (np.power(10.0, (x - d) / c) - b) / a,
            (x - f) / e,
        )
        return np.maximum(lin, 0.0).astype(np.float32, copy=False)

    if prof == "log3g10":
        # RED IPP2 Log3G10
        a = 0.224282
        b = 155.975327
        c = 0.01
        g = 15.1927
        lin = np.where(
            x >= 0.0,
            (np.power(10.0, x / a) - 1.0) / b - c,
            x / g - c,
        )
        return np.maximum(lin, 0.0).astype(np.float32, copy=False)

    if prof == "acescg":
        flat = x.reshape(-1, 3) @ _ACESCG_TO_REC709.T
        return flat.reshape(x.shape).astype(np.float32, copy=False)

    return x.astype(np.float32, copy=False)


def linear_to_log(rgb: np.ndarray, profile: str) -> np.ndarray:
    """Inverse transform from linear Rec.709 float RGB back to named Log/gamut space."""
    prof = (profile or "none").strip().lower()
    if prof in ("", "none"):
        return np.asarray(rgb, dtype=np.float32)

    lin = np.asarray(rgb, dtype=np.float32)
    if prof == "slog3":
        out = np.where(
            lin >= 0.01125000,
            (420.0 + np.log10(np.maximum((lin + 0.01) / (0.18 + 0.01), 1e-6)) * 261.5) / 1023.0,
            (lin * (171.2102946929 - 95.0) / 0.01125000 + 95.0) / 1023.0,
        )
        return out.astype(np.float32, copy=False)

    if prof == "logc3":
        a, b, c, d, e, f = 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
        out = np.where(
            lin > 0.010591,
            c * np.log10(np.maximum(a * lin + b, 1e-6)) + d,
            e * lin + f,
        )
        return out.astype(np.float32, copy=False)

    if prof == "log3g10":
        a, b, c, g = 0.224282, 155.975327, 0.01, 15.1927
        shifted = lin + c
        out = np.where(
            shifted >= 0.0,
            a * np.log10(np.maximum(shifted * b + 1.0, 1e-6)),
            shifted * g,
        )
        return out.astype(np.float32, copy=False)

    if prof == "acescg":
        flat = lin.reshape(-1, 3) @ _REC709_TO_ACESCG.T
        return flat.reshape(lin.shape).astype(np.float32, copy=False)

    return lin.astype(np.float32, copy=False)


class CubeLUT:
    """Parser and fast vectorized applicator for 1D and 3D `.cube` LUT files."""

    def __init__(
        self,
        table: np.ndarray,
        is_3d: bool,
        domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        domain_max: tuple[float, float, float] = (1.0, 1.0, 1.0),
        title: str = "",
    ) -> None:
        self.table = np.asarray(table, dtype=np.float32)
        self.is_3d = bool(is_3d)
        self.domain_min = np.asarray(domain_min, dtype=np.float32)
        self.domain_max = np.asarray(domain_max, dtype=np.float32)
        self.title = title

    @classmethod
    def from_file(cls, path: str) -> "CubeLUT":
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return cls.from_lines(text.splitlines())

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "CubeLUT":
        size_3d = 0
        size_1d = 0
        dmin = (0.0, 0.0, 0.0)
        dmax = (1.0, 1.0, 1.0)
        title = ""
        rows: list[tuple[float, float, float]] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            head = parts[0].upper()
            if head == "TITLE":
                title = line[5:].strip().strip('"')
                continue
            if head == "LUT_3D_SIZE" and len(parts) >= 2:
                size_3d = int(parts[1])
                continue
            if head == "LUT_1D_SIZE" and len(parts) >= 2:
                size_1d = int(parts[1])
                continue
            if head == "DOMAIN_MIN" and len(parts) >= 4:
                dmin = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue
            if head == "DOMAIN_MAX" and len(parts) >= 4:
                dmax = (float(parts[1]), float(parts[2]), float(parts[3]))
                continue
            if head in ("LUT_1D_INPUT_RANGE", "LUT_3D_INPUT_RANGE"):
                if len(parts) >= 3:
                    v0, v1 = float(parts[1]), float(parts[2])
                    dmin = (v0, v0, v0)
                    dmax = (v1, v1, v1)
                continue
            if len(parts) >= 3:
                try:
                    rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
                except ValueError:
                    continue

        if size_3d > 0:
            expected = size_3d ** 3
            if len(rows) < expected:
                raise ValueError(f"Expected {expected} rows for {size_3d}^3 3D LUT, got {len(rows)}")
            arr = np.asarray(rows[:expected], dtype=np.float32).reshape((size_3d, size_3d, size_3d, 3))
            # In Adobe/Resolve .cube format, Red changes fastest, then Green, then Blue: axis order is [b, g, r, c].
            return cls(arr, is_3d=True, domain_min=dmin, domain_max=dmax, title=title)

        if size_1d > 0 or rows:
            n = size_1d if size_1d > 0 else len(rows)
            arr = np.asarray(rows[:n], dtype=np.float32)
            return cls(arr, is_3d=False, domain_min=dmin, domain_max=dmax, title=title)

        raise ValueError("No valid LUT data found in .cube input")

    def apply(self, rgb_f32: np.ndarray) -> np.ndarray:
        """Apply LUT to `[..., 3]` float32 RGB normalized in [0..1] domain."""
        img = np.asarray(rgb_f32, dtype=np.float32)
        orig_shape = img.shape
        denom = np.maximum(self.domain_max - self.domain_min, 1e-6)
        norm = np.clip((img - self.domain_min) / denom, 0.0, 1.0).reshape(-1, 3)

        if not self.is_3d:
            n = self.table.shape[0]
            if n < 2:
                return img.copy()
            pos = norm * (n - 1)
            i0 = np.floor(pos).astype(np.int32)
            i1 = np.minimum(i0 + 1, n - 1)
            f = pos - i0
            out = np.empty_like(norm)
            for ch in range(3):
                c0 = self.table[i0[:, ch], ch]
                c1 = self.table[i1[:, ch], ch]
                out[:, ch] = c0 * (1.0 - f[:, ch]) + c1 * f[:, ch]
            return out.reshape(orig_shape).astype(np.float32, copy=False)

        n = self.table.shape[0]
        if n < 2:
            return img.copy()
        scaled = norm * (n - 1)
        r0 = np.floor(scaled[:, 0]).astype(np.int32)
        g0 = np.floor(scaled[:, 1]).astype(np.int32)
        b0 = np.floor(scaled[:, 2]).astype(np.int32)
        r1 = np.minimum(r0 + 1, n - 1)
        g1 = np.minimum(g0 + 1, n - 1)
        b1 = np.minimum(b0 + 1, n - 1)

        fr = (scaled[:, 0] - r0)[:, None]
        fg = (scaled[:, 1] - g0)[:, None]
        fb = (scaled[:, 2] - b0)[:, None]

        # Trilinear lookup in [b, g, r, c] layout
        c000 = self.table[b0, g0, r0]
        c001 = self.table[b0, g0, r1]
        c010 = self.table[b0, g1, r0]
        c011 = self.table[b0, g1, r1]
        c100 = self.table[b1, g0, r0]
        c101 = self.table[b1, g0, r1]
        c110 = self.table[b1, g1, r0]
        c111 = self.table[b1, g1, r1]

        c00 = c000 * (1.0 - fr) + c001 * fr
        c01 = c010 * (1.0 - fr) + c011 * fr
        c10 = c100 * (1.0 - fr) + c101 * fr
        c11 = c110 * (1.0 - fr) + c111 * fr

        c0 = c00 * (1.0 - fg) + c01 * fg
        c1 = c10 * (1.0 - fg) + c11 * fg

        out = c0 * (1.0 - fb) + c1 * fb
        return out.reshape(orig_shape).astype(np.float32, copy=False)


_LUT_LOCK = threading.Lock()
_DISPLAY_LUT: CubeLUT | None = None
_DISPLAY_LUT_PATH: str = ""
_DISPLAY_LUT_ENABLED: bool = False


def set_display_lut(path: str | None) -> bool:
    """Load and cache a viewer-only Display .cube LUT (does not affect exports)."""
    global _DISPLAY_LUT, _DISPLAY_LUT_PATH
    if not path:
        with _LUT_LOCK:
            _DISPLAY_LUT = None
            _DISPLAY_LUT_PATH = ""
        return True
    try:
        lut = CubeLUT.from_file(path)
        with _LUT_LOCK:
            _DISPLAY_LUT = lut
            _DISPLAY_LUT_PATH = str(path)
        return True
    except Exception as exc:
        logger.warning("Failed to load LUT %s: %s", path, exc)
        return False


def set_display_lut_enabled(enabled: bool) -> None:
    global _DISPLAY_LUT_ENABLED
    with _LUT_LOCK:
        _DISPLAY_LUT_ENABLED = bool(enabled)


def get_display_lut_state() -> tuple[bool, str]:
    with _LUT_LOCK:
        return _DISPLAY_LUT_ENABLED and (_DISPLAY_LUT is not None), _DISPLAY_LUT_PATH


def apply_display_lut_rgb(rgb: np.ndarray) -> np.ndarray:
    """Apply cached viewer Display LUT if enabled (viewer-only). Safe identity when off."""
    with _LUT_LOCK:
        lut = _DISPLAY_LUT if (_DISPLAY_LUT_ENABLED and _DISPLAY_LUT is not None) else None
    if lut is None:
        return rgb
    was_u8 = rgb.dtype == np.uint8
    f32 = rgb.astype(np.float32) * (1.0 / 255.0) if was_u8 else np.asarray(rgb, dtype=np.float32)
    mapped = np.clip(lut.apply(f32), 0.0, 1.0)
    if was_u8:
        return (mapped * 255.0).astype(np.uint8)
    return mapped
