"""Optional post-processing & advanced keying engines (all toggle-gated, off by default)."""

from __future__ import annotations

import cv2
import numpy as np


def advanced_despill(
    rgb_f32: np.ndarray,
    strength: float = 0.5,
    screen_color: str = "green",
    *,
    mode: str = "average",
    luma_restore: float = 0.0,
) -> np.ndarray:
    """Remove green/blue screen spill with selectable limit law and luminance recovery."""
    out = np.asarray(rgb_f32, dtype=np.float32).copy()
    s = float(np.clip(strength, 0.0, 1.5))
    lr = float(np.clip(luma_restore, 0.0, 1.0))
    m = (mode or "average").strip().lower()
    sc = (screen_color or "green").strip().lower()

    r, g, b = out[..., 0], out[..., 1], out[..., 2]
    luma_before = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if sc == "blue":
        if m == "max":
            limit = np.maximum(r, g)
        elif m == "red_blue":
            limit = 0.35 * r + 0.65 * g
        else:
            limit = 0.5 * (r + g)
        excess = np.maximum(0.0, b - limit)
        out[..., 2] = b - excess * s
    else:
        if m == "max":
            limit = np.maximum(r, b)
        elif m == "red_blue":
            limit = 0.65 * r + 0.35 * b
        else:
            limit = 0.5 * (r + b)
        excess = np.maximum(0.0, g - limit)
        out[..., 1] = g - excess * s

    if lr > 0.0:
        luma_after = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
        delta = np.maximum(0.0, luma_before - luma_after) * lr
        out = out + delta[..., None]

    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply_edge_extend(
    fg_rgb_f32: np.ndarray,
    alpha_2d: np.ndarray,
    radius_px: int = 0,
) -> np.ndarray:
    """Extrude clean interior foreground colors outward into semi-transparent edge halos."""
    r = int(radius_px or 0)
    if r <= 0:
        return np.asarray(fg_rgb_f32, dtype=np.float32)

    fg = np.clip(np.asarray(fg_rgb_f32, dtype=np.float32), 0.0, 1.0)
    a = alpha_2d[:, :, 0] if alpha_2d.ndim == 3 else alpha_2d
    a = np.clip(a.astype(np.float32, copy=False), 0.0, 1.0)

    core_w = np.clip((a - 0.80) / 0.20, 0.0, 1.0)
    ksize = max(3, (r * 2 + 1) | 1)
    num = cv2.GaussianBlur(fg * core_w[:, :, None], (ksize, ksize), 0.0)
    den = cv2.GaussianBlur(core_w, (ksize, ksize), 0.0)
    extended = num / np.maximum(den[:, :, None], 1e-4)

    # Blend pushed edge color only along transition zone (0.01 < alpha < 0.85), never interior core
    edge_mix = np.clip((0.85 - a) / 0.85, 0.0, 1.0) * (a > 0.01).astype(np.float32) * (den > 1e-3).astype(np.float32)
    out = fg * (1.0 - edge_mix[:, :, None]) + extended * edge_mix[:, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply_light_wrap(
    fg_rgb_f32: np.ndarray,
    bg_rgb_f32: np.ndarray,
    alpha_2d: np.ndarray,
    strength: float = 0.0,
    radius_px: int = 12,
) -> np.ndarray:
    """Wrap blurred backdrop light smoothly onto foreground silhouette edges."""
    st = float(np.clip(strength, 0.0, 1.0))
    if st <= 0.0:
        return np.asarray(fg_rgb_f32, dtype=np.float32)

    fg = np.clip(np.asarray(fg_rgb_f32, dtype=np.float32), 0.0, 1.0)
    bg = np.clip(np.asarray(bg_rgb_f32, dtype=np.float32), 0.0, 1.0)
    if bg.shape[:2] != fg.shape[:2]:
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]), interpolation=cv2.INTER_LINEAR)

    a = alpha_2d[:, :, 0] if alpha_2d.ndim == 3 else alpha_2d
    a = np.clip(a.astype(np.float32, copy=False), 0.0, 1.0)
    inv_a = 1.0 - a
    r = max(1, int(radius_px or 12))
    ksize = max(3, (r * 2 + 1) | 1)
    blurred_bg = cv2.GaussianBlur(bg * inv_a[:, :, None], (ksize, ksize), 0.0)

    # Light wrap amount fades toward deep opaque interior and concentrates along outer contour
    wrap_zone = (1.0 - cv2.GaussianBlur(a, (ksize, ksize), 0.0)) * a
    wrap = blurred_bg * (wrap_zone[:, :, None] * st * 2.0)
    out = 1.0 - (1.0 - fg) * (1.0 - np.clip(wrap, 0.0, 1.0))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def apply_clean_plate_difference(
    alpha_f32: np.ndarray,
    frame_rgb_f32: np.ndarray,
    clean_plate_rgb_f32: np.ndarray,
    strength: float = 0.5,
    screen_color: str = "green",
) -> np.ndarray:
    """IBK-style clean plate difference modulation preserving screen shadows & wispy detail."""
    st = float(np.clip(strength, 0.0, 1.0))
    if st <= 0.0:
        return np.clip(alpha_f32.astype(np.float32, copy=False), 0.0, 1.0)

    a = np.clip(alpha_f32.astype(np.float32, copy=False), 0.0, 1.0)
    fr = np.clip(frame_rgb_f32.astype(np.float32, copy=False), 0.0, 1.0)
    cp = np.clip(clean_plate_rgb_f32.astype(np.float32, copy=False), 0.0, 1.0)
    if cp.shape[:2] != fr.shape[:2]:
        cp = cv2.resize(cp, (fr.shape[1], fr.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Measure screen-corrected chroma distance vs luminance ratio (separates cast shadows from subject)
    diff = np.abs(fr - cp)
    sc = (screen_color or "green").strip().lower()
    if sc == "blue":
        chroma_dist = np.maximum(diff[..., 0], diff[..., 1]) + 0.5 * diff[..., 2]
    else:
        chroma_dist = np.maximum(diff[..., 0], diff[..., 2]) + 0.5 * diff[..., 1]

    # Soft normalized difference density in [0..1]
    diff_alpha = np.clip((chroma_dist - 0.03) / 0.18, 0.0, 1.0)
    # Only attenuate false-positive spill/uneven-bg inside fringe & background (keep solid interior core)
    mod = np.where(a > 0.92, a, a * ((1.0 - st) + st * diff_alpha))
    return np.clip(mod, 0.0, 1.0).astype(np.float32, copy=False)


def generate_auto_clean_plate(frame_rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Generate a clean background plate by inpainting the masked subject from surrounding screen."""
    m = mask_u8[:, :, 0] if mask_u8.ndim == 3 else mask_u8
    if m.shape[:2] != frame_rgb_u8.shape[:2]:
        m = cv2.resize(m, (frame_rgb_u8.shape[1], frame_rgb_u8.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary = (m > 25).astype(np.uint8) * 255
    h, w = frame_rgb_u8.shape[:2]
    k = max(3, min(19, (min(h, w) // 8) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    inpaint_mask = cv2.dilate(binary, kernel)
    inpaint_r = max(3, min(11, k))
    return cv2.inpaint(frame_rgb_u8, inpaint_mask, inpaint_r, cv2.INPAINT_TELEA)


def apply_temporal_alpha_deflicker(
    alpha_seq: list[np.ndarray],
    strength: float = 0.35,
    radius: int = 1,
) -> list[np.ndarray]:
    """Temporal median stabilization along semi-transparent edge pixels (`0.02..0.98`)."""
    n = len(alpha_seq)
    st = float(np.clip(strength, 0.0, 1.0))
    rad = max(1, min(3, int(radius or 1)))
    if n < 3 or st <= 0.0:
        return [np.clip(a.astype(np.float32, copy=False), 0.0, 1.0) for a in alpha_seq]

    stack = np.stack([np.clip(a.astype(np.float32, copy=False), 0.0, 1.0) for a in alpha_seq], axis=0)
    out: list[np.ndarray] = []
    for i in range(n):
        lo = max(0, i - rad)
        hi = min(n, i + rad + 1)
        curr = stack[i]
        med = np.median(stack[lo:hi], axis=0).astype(np.float32, copy=False)
        edge_band = ((curr > 0.02) & (curr < 0.98) | ((med > 0.02) & (med < 0.98))).astype(np.float32)
        w = st * edge_band
        blended = curr * (1.0 - w) + med * w
        out.append(np.clip(blended, 0.0, 1.0))
    return out
