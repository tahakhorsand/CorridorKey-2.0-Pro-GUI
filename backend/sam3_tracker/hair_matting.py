"""
High-precision hair & fine-detail matte refinement for SAM 3.1 segmentation.

Replaces naive binary thresholding around hair strokes with a 3-zone trimap,
spatially-varying foreground/background color fields interpolated via
multi-scale normalized Gaussian convolution, a chroma-weighted color-line
alpha estimator, and a two-pass 3-channel RGB Guided Filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class HairMattingConfig:
    """Tuning parameters for local color-line & guided-filter hair matting."""

    roi_padding_px: int = 56
    roi_feather_px: int = 10
    bg_inner_exclude_px: int = 5
    fg_seed_erode_px: int = 3
    non_roi_edge_band_px: int = 4
    guided_radius_coarse: int = 8
    guided_eps_coarse: float = 1e-3
    guided_radius_fine: int = 2
    guided_eps_fine: float = 1.5e-4
    chroma_weight: float = 1.35
    luma_weight: float = 0.45
    alpha_floor_clip: float = 0.025
    alpha_ceil_clip: float = 0.985


def build_hair_roi_mask(
    height: int,
    width: int,
    hair_points: Optional[Sequence[Tuple[float, float]]] = None,
    hair_strokes: Optional[Sequence[Tuple[Sequence[Tuple[float, float]], float]]] = None,
    default_point_radius: float = 22.0,
    expand_scale: float = 1.35,
) -> np.ndarray:
    """Rasterize hair brush strokes and fallback points into a soft [0, 1] ROI mask.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        hair_points: Individual `(x, y)` hair prompt coordinates.
        hair_strokes: Full stroke polylines `([(x, y), ...], brush_radius)`.
        default_point_radius: Fallback radius when only sampled `hair_points` exist.
        expand_scale: Multiplier expanding stroke radius to capture surrounding
            transition/background seeds around wispy strands.
    """
    roi = np.zeros((height, width), dtype=np.float32)
    has_strokes = False

    if hair_strokes:
        for pts, radius in hair_strokes:
            if not pts:
                continue
            has_strokes = True
            rad = max(4, int(round(float(radius) * expand_scale)))
            if len(pts) == 1:
                cx, cy = int(round(pts[0][0])), int(round(pts[0][1]))
                cv2.circle(roi, (cx, cy), rad, 1.0, thickness=-1, lineType=cv2.LINE_AA)
            else:
                arr = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(roi, [arr], isClosed=False, color=1.0, thickness=rad * 2, lineType=cv2.LINE_AA)

    if hair_points and not has_strokes:
        rad = max(6, int(round(default_point_radius * expand_scale)))
        for x, y in hair_points:
            cx, cy = int(round(x)), int(round(y))
            cv2.circle(roi, (cx, cy), rad, 1.0, thickness=-1, lineType=cv2.LINE_AA)
    elif hair_points and has_strokes:
        # Ensure any standalone hair points outside painted strokes are also covered.
        small_rad = max(6, int(round(default_point_radius)))
        for x, y in hair_points:
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cy < height and 0 <= cx < width and roi[cy, cx] < 0.25:
                cv2.circle(roi, (cx, cy), small_rad, 1.0, thickness=-1, lineType=cv2.LINE_AA)

    return np.clip(roi, 0.0, 1.0)


def compute_mask_bbox(
    mask: np.ndarray,
    threshold: float = 0.05,
    pad: int = 0,
) -> Optional[Tuple[int, int, int, int]]:
    """Return `(x0, y0, x1, y1)` exclusive bounds for pixels > `threshold`, or None."""
    ys, xs = np.where(mask > threshold)
    if ys.size == 0:
        return None
    h, w = mask.shape[:2]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return (x0, y0, x1, y1)


def _rasterize_negative_mask(
    height: int,
    width: int,
    negative_points: Optional[Sequence[Tuple[float, float]]] = None,
    negative_strokes: Optional[Sequence[Tuple[Sequence[Tuple[float, float]], float]]] = None,
) -> np.ndarray:
    """Rasterize explicit background points/strokes into a [0, 1] exclusion mask."""
    neg = np.zeros((height, width), dtype=np.float32)
    if negative_strokes:
        for pts, radius in negative_strokes:
            if not pts:
                continue
            rad = max(2, int(round(float(radius))))
            if len(pts) == 1:
                cv2.circle(
                    neg,
                    (int(round(pts[0][0])), int(round(pts[0][1]))),
                    rad,
                    1.0,
                    thickness=-1,
                )
            else:
                arr = np.round(np.asarray(pts, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(neg, [arr], isClosed=False, color=1.0, thickness=rad * 2)
    if negative_points:
        for x, y in negative_points:
            cv2.circle(neg, (int(round(x)), int(round(y))), 4, 1.0, thickness=-1)
    return neg


def guided_filter_rgb(
    guide_rgb: np.ndarray,
    src_alpha: np.ndarray,
    radius: int = 4,
    eps: float = 2e-4,
) -> np.ndarray:
    """Fast 3-channel RGB Guided Filter (He et al., TPAMI 2013).

    Preserves sharp sub-pixel hair strand boundaries aligned with color edges
    in `guide_rgb` (`float32` `[H, W, 3]` in `[0, 1]`).
    """
    r = max(1, int(radius))
    ksize = (2 * r + 1, 2 * r + 1)

    I = np.ascontiguousarray(guide_rgb, dtype=np.float32)
    p = np.ascontiguousarray(src_alpha, dtype=np.float32)

    Ir = I[..., 0]
    Ig = I[..., 1]
    Ib = I[..., 2]

    mean_Ir = cv2.boxFilter(Ir, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_Ig = cv2.boxFilter(Ig, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_Ib = cv2.boxFilter(Ib, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_p = cv2.boxFilter(p, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)

    cov_Irp = cv2.boxFilter(Ir * p, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ir * mean_p
    cov_Igp = cv2.boxFilter(Ig * p, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ig * mean_p
    cov_Ibp = cv2.boxFilter(Ib * p, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ib * mean_p

    var_rr = cv2.boxFilter(Ir * Ir, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ir * mean_Ir + eps
    var_rg = cv2.boxFilter(Ir * Ig, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ir * mean_Ig
    var_rb = cv2.boxFilter(Ir * Ib, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ir * mean_Ib
    var_gg = cv2.boxFilter(Ig * Ig, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ig * mean_Ig + eps
    var_gb = cv2.boxFilter(Ig * Ib, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ig * mean_Ib
    var_bb = cv2.boxFilter(Ib * Ib, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101) - mean_Ib * mean_Ib + eps

    # Exact closed-form 3x3 symmetric covariance inverse via cofactors.
    inv_rr = var_gg * var_bb - var_gb * var_gb
    inv_rg = var_gb * var_rb - var_rg * var_bb
    inv_rb = var_rg * var_gb - var_gg * var_rb
    inv_gg = var_rr * var_bb - var_rb * var_rb
    inv_gb = var_rb * var_rg - var_rr * var_gb
    inv_bb = var_rr * var_gg - var_rg * var_rg

    det = var_rr * inv_rr + var_rg * inv_rg + var_rb * inv_rb
    inv_det = 1.0 / np.maximum(det, 1e-12)

    ar = (inv_rr * cov_Irp + inv_rg * cov_Igp + inv_rb * cov_Ibp) * inv_det
    ag = (inv_rg * cov_Irp + inv_gg * cov_Igp + inv_gb * cov_Ibp) * inv_det
    ab = (inv_rb * cov_Irp + inv_gb * cov_Igp + inv_bb * cov_Ibp) * inv_det

    b = mean_p - ar * mean_Ir - ag * mean_Ig - ab * mean_Ib

    mean_ar = cv2.boxFilter(ar, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_ag = cv2.boxFilter(ag, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_ab = cv2.boxFilter(ab, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)
    mean_b = cv2.boxFilter(b, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT101)

    q = mean_ar * Ir + mean_ag * Ig + mean_ab * Ib + mean_b
    return np.clip(q, 0.0, 1.0).astype(np.float32)


def _interpolate_field(
    values: np.ndarray,
    weights: np.ndarray,
    sigma_coarse: float = 28.0,
    sigma_fine: float = 9.0,
    fallback_color: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Interpolate a dense spatially-varying `(h, w, 3)` color field from sparse seeds.

    Uses multi-scale normalized Gaussian convolution (coarse fill bridged into
    fine local color detail wherever seed support is dense).
    """
    w = np.clip(weights, 0.0, 1.0).astype(np.float32)
    h, wd = w.shape
    weighted_vals = values.astype(np.float32) * w[..., None]

    den_coarse = cv2.GaussianBlur(w, (0, 0), sigmaX=sigma_coarse, sigmaY=sigma_coarse, borderType=cv2.BORDER_REPLICATE)
    num_coarse = cv2.GaussianBlur(weighted_vals, (0, 0), sigmaX=sigma_coarse, sigmaY=sigma_coarse, borderType=cv2.BORDER_REPLICATE)

    if fallback_color is None:
        total_w = float(w.sum())
        if total_w > 1e-6:
            fallback_color = (weighted_vals.sum(axis=(0, 1)) / total_w).astype(np.float32)
        else:
            fallback_color = values.mean(axis=(0, 1)).astype(np.float32)

    coarse_valid = den_coarse > 1e-5
    field_coarse = np.broadcast_to(fallback_color.reshape(1, 1, 3), (h, wd, 3)).copy()
    field_coarse[coarse_valid] = num_coarse[coarse_valid] / den_coarse[coarse_valid, None]

    den_fine = cv2.GaussianBlur(w, (0, 0), sigmaX=sigma_fine, sigmaY=sigma_fine, borderType=cv2.BORDER_REPLICATE)
    num_fine = cv2.GaussianBlur(weighted_vals, (0, 0), sigmaX=sigma_fine, sigmaY=sigma_fine, borderType=cv2.BORDER_REPLICATE)

    blend = np.clip(den_fine / (den_fine + 0.08), 0.0, 1.0).astype(np.float32)
    field_fine = np.zeros_like(field_coarse)
    fine_valid = den_fine > 1e-6
    field_fine[fine_valid] = num_fine[fine_valid] / den_fine[fine_valid, None]

    return np.clip(blend[..., None] * field_fine + (1.0 - blend[..., None]) * field_coarse, 0.0, 1.0)


def _weighted_color_diff(
    a: np.ndarray,
    b: np.ndarray,
    luma_w: float,
    chroma_w: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute perceptually + chroma-weighted difference vector and magnitude.

    On saturated backgrounds (e.g., studio green/blue screen), shadows shift
    luminance while keeping chromaticity steady; separating luma (`dY`) and
    zero-mean opponent chroma (`dC`) avoids false positives on green creases
    while sharply separating fine dark/blonde hair strands from screen pixels.
    """
    luma_coeffs = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    y_a = (a * luma_coeffs).sum(axis=-1, keepdims=True)
    y_b = (b * luma_coeffs).sum(axis=-1, keepdims=True)

    c_a = a - y_a
    c_b = b - y_b

    # When local background is nearly neutral/achromatic, equalize weights cleanly.
    bg_sat = np.sqrt((c_b * c_b).sum(axis=-1))
    sat_mix = np.clip((bg_sat - 0.04) / 0.12, 0.0, 1.0).astype(np.float32)

    eff_lw = float(luma_w) * sat_mix + (1.0 - sat_mix) * 1.0
    eff_cw = float(chroma_w) * sat_mix + (1.0 - sat_mix) * 1.0

    dy = (y_a - y_b)[..., 0]
    dc = c_a - c_b

    # Weighted 4-component vector [sqrt(lw)*dY, sqrt(cw)*dCr, sqrt(cw)*dCg, sqrt(cw)*dCb]
    slw = np.sqrt(eff_lw)
    scw = np.sqrt(eff_cw)
    vec = np.empty(a.shape[:-1] + (4,), dtype=np.float32)
    vec[..., 0] = slw * dy
    vec[..., 1:] = scw[..., None] * dc

    dist = np.sqrt(np.maximum((vec * vec).sum(axis=-1) / 3.0, 0.0))
    return vec, dist, bg_sat


def refine_hair_matte(
    frame_rgb: np.ndarray,
    sam_prob: np.ndarray,
    hair_points: Optional[Sequence[Tuple[float, float]]] = None,
    hair_strokes: Optional[Sequence[Tuple[Sequence[Tuple[float, float]], float]]] = None,
    negative_points: Optional[Sequence[Tuple[float, float]]] = None,
    negative_strokes: Optional[Sequence[Tuple[Sequence[Tuple[float, float]], float]]] = None,
    roi_mask: Optional[np.ndarray] = None,
    cfg: Optional[HairMattingConfig] = None,
    apply_boundary_polish: bool = True,
) -> np.ndarray:
    """Refine a SAM 3.1 segmentation probability map into a continuous soft alpha matte.

    Unlike additive hair heuristics (`np.maximum`), this computes a 3-zone trimap
    and local foreground/background fields so trapped background pockets inside
    SAM's coarse binary mask (e.g., green screen visible between bangs) are
    carved out while individual sub-pixel wispy strands are recovered.

    Args:
        frame_rgb: `[H, W, 3]` RGB image (`uint8` `[0..255]` or `float` `[0..1]`).
        sam_prob: `[H, W]` SAM probability map in `[0, 1]`.
        hair_points: Optional user hair prompt `(x, y)` coordinates.
        hair_strokes: Optional full hair brush polylines `([(x, y), ...], radius)`.
        negative_points: Optional user negative `(x, y)` coordinates.
        negative_strokes: Optional negative brush polylines.
        roi_mask: Optional pre-built `[H, W]` float ROI mask in `[0, 1]`.
        cfg: Optional `HairMattingConfig`.
        apply_boundary_polish: If True, also smooths the tight `~4px` perimeter
            ribbon outside the hair ROI with RGB Guided Filtering.

    Returns:
        `[H, W]` `float32` continuous matte in `[0, 1]`.
    """
    if cfg is None:
        cfg = HairMattingConfig()

    h, w = sam_prob.shape[:2]
    prob = np.clip(np.asarray(sam_prob, dtype=np.float32), 0.0, 1.0)

    if frame_rgb.dtype == np.uint8:
        img = frame_rgb.astype(np.float32) * (1.0 / 255.0)
    else:
        img = np.clip(frame_rgb.astype(np.float32), 0.0, 1.0)
        if float(img.max()) > 1.5:
            img = img * (1.0 / 255.0)

    if roi_mask is None:
        roi_mask = build_hair_roi_mask(h, w, hair_points=hair_points, hair_strokes=hair_strokes)
    else:
        roi_mask = np.clip(np.asarray(roi_mask, dtype=np.float32), 0.0, 1.0)

    neg_mask = _rasterize_negative_mask(h, w, negative_points=negative_points, negative_strokes=negative_strokes)
    out_matte = prob.copy()

    # Smooth transition outside hair ROI first on a tight edge ribbon so the
    # body contour is anti-aliased without disturbing solid interior foreground.
    if apply_boundary_polish:
        core_bin = (prob >= 0.5).astype(np.uint8)
        if core_bin.any() and not core_bin.all():
            k = max(3, int(cfg.non_roi_edge_band_px) * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            dil = cv2.dilate(core_bin, kernel)
            ero = cv2.erode(core_bin, kernel)
            edge_band = ((dil ^ ero) > 0).astype(np.float32)
            bbox_obj = compute_mask_bbox(edge_band, threshold=0.5, pad=cfg.guided_radius_coarse + 8)
            if bbox_obj is not None:
                x0, y0, x1, y1 = bbox_obj
                gf_edge = guided_filter_rgb(
                    img[y0:y1, x0:x1],
                    prob[y0:y1, x0:x1],
                    radius=max(2, cfg.guided_radius_fine),
                    eps=cfg.guided_eps_fine,
                )
                eb_crop = cv2.GaussianBlur(edge_band[y0:y1, x0:x1], (0, 0), sigmaX=1.5, sigmaY=1.5)
                out_matte[y0:y1, x0:x1] = (
                    eb_crop * gf_edge + (1.0 - eb_crop) * prob[y0:y1, x0:x1]
                )

    bbox_roi = compute_mask_bbox(roi_mask, threshold=0.05, pad=cfg.roi_padding_px)
    if bbox_roi is None:
        return np.clip(out_matte, 0.0, 1.0).astype(np.float32)

    x0, y0, x1, y1 = bbox_roi
    img_c = img[y0:y1, x0:x1]
    prob_c = prob[y0:y1, x0:x1]
    roi_c = roi_mask[y0:y1, x0:x1]
    neg_c = neg_mask[y0:y1, x0:x1]
    ch, cw = prob_c.shape

    # 1. Definite background seeds: pixels safely outside SAM foreground or marked negative,
    #    avoiding the immediate 5px silhouette boundary so halos don't pollute B(x, y).
    raw_fg = (prob_c > 0.15).astype(np.uint8)
    bg_ex_k = max(3, cfg.bg_inner_exclude_px * 2 + 1)
    bg_ex_kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bg_ex_k, bg_ex_k))
    near_fg = cv2.dilate(raw_fg, bg_ex_kern) > 0
    sure_bg = ((~near_fg) & (roi_c < 0.85)) | (neg_c > 0.5)
    if int(sure_bg.sum()) < 24:
        sure_bg = (prob_c < 0.08) | (neg_c > 0.5)
    if int(sure_bg.sum()) < 16:
        sure_bg = (prob_c < 0.25) | (neg_c > 0.5)

    if int(sure_bg.sum()) < 8:
        # Degenerate crop completely filled with foreground: only apply gentle guided polish.
        gf = guided_filter_rgb(img_c, prob_c, radius=cfg.guided_radius_fine, eps=cfg.guided_eps_fine)
        w_blend = cv2.GaussianBlur(roi_c, (0, 0), sigmaX=4.0, sigmaY=4.0)
        out_matte[y0:y1, x0:x1] = np.clip(
            w_blend * gf + (1.0 - w_blend) * out_matte[y0:y1, x0:x1], 0.0, 1.0
        )
        return out_matte

    bg_weights = sure_bg.astype(np.float32)
    # Trim any outlier FG strand accidentally overlapping bg seeds via median fallback color.
    bg_median = np.median(img_c[sure_bg], axis=0).astype(np.float32)
    bg_field = _interpolate_field(
        img_c, bg_weights, sigma_coarse=28.0, sigma_fine=8.0, fallback_color=bg_median
    )

    # 2. Measure distance from local spatially-varying background B(x, y)
    vec_ib, dist_bg, bg_sat = _weighted_color_diff(
        img_c, bg_field, luma_w=cfg.luma_weight, chroma_w=cfg.chroma_weight
    )

    bg_dists = dist_bg[sure_bg]
    bg_p95 = float(np.percentile(bg_dists, 95)) if bg_dists.size > 0 else 0.05
    bg_med_dist = float(np.median(bg_dists)) if bg_dists.size > 0 else 0.02
    tau_bg = float(np.clip(max(0.032, bg_p95 * 1.18, bg_med_dist * 2.2), 0.032, 0.16))

    # Measure how uniform/chroma-separable the local backdrop is.
    bg_consistency = float(np.clip(1.0 - (bg_p95 - 0.045) / 0.20, 0.28, 1.0))

    # 3. Definite foreground seeds: eroded core with genuine contrast over local B(x, y).
    fg_raw = (prob_c > 0.65).astype(np.uint8)
    ero_k = max(3, cfg.fg_seed_erode_px * 2 + 1)
    fg_eroded = cv2.erode(fg_raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero_k, ero_k))) > 0

    fg_min_dist = max(0.16, 2.2 * tau_bg)
    sure_fg = (fg_eroded | ((prob_c > 0.55) & (roi_c > 0.1))) & (dist_bg > fg_min_dist) & (neg_c < 0.5)
    if int(sure_fg.sum()) < 24:
        sure_fg = (prob_c > 0.50) & (dist_bg > max(0.12, 1.6 * tau_bg)) & (neg_c < 0.5)
    if int(sure_fg.sum()) < 12:
        # Take highest-contrast pixels inside SAM/ROI as foreground color anchor.
        cand = (prob_c > 0.35) | (roi_c > 0.5)
        if cand.any():
            thresh_hi = float(np.percentile(dist_bg[cand], 78))
            sure_fg = cand & (dist_bg >= max(thresh_hi, tau_bg * 1.35)) & (neg_c < 0.5)

    if int(sure_fg.sum()) < 6:
        return np.clip(out_matte, 0.0, 1.0).astype(np.float32)

    # Weight FG seeds by contrast over background so pure interior hair/skin dominates F(x, y)
    fg_conf = np.clip((dist_bg - tau_bg) / 0.25, 0.0, 1.0) * sure_fg.astype(np.float32)
    fg_median = np.median(img_c[sure_fg], axis=0).astype(np.float32)
    fg_field = _interpolate_field(
        img_c, fg_conf, sigma_coarse=24.0, sigma_fine=9.0, fallback_color=fg_median
    )

    # 4. Closed-form Color-Line Projection + Noise-Floored Distance Alpha
    vec_fb, dist_fb, _ = _weighted_color_diff(
        fg_field, bg_field, luma_w=cfg.luma_weight, chroma_w=cfg.chroma_weight
    )
    dot_num = (vec_ib * vec_fb).sum(axis=-1)
    dot_den = (vec_fb * vec_fb).sum(axis=-1)
    alpha_proj = np.clip(dot_num / np.maximum(dot_den, 1e-4), 0.0, 1.0)

    # Subtract background compression/shadow noise floor smoothly.
    denom_dist = np.maximum(dist_fb - tau_bg * 0.75, 0.09)
    alpha_dist = np.clip((dist_bg - tau_bg * 0.85) / denom_dist, 0.0, 1.0)

    # How close is pixel I to the linear segment [B(x, y), F(x, y)]?
    # Well-aligned strand pixels trust exact linear projection; off-line tinted pixels
    # safely fall back to background distance.
    proj_point = vec_fb * alpha_proj[..., None]
    orth_err = np.sqrt(np.maximum(((vec_ib - proj_point) ** 2).sum(axis=-1) / 3.0, 0.0))
    line_conf = np.clip(1.0 - orth_err / np.maximum(dist_fb * 0.55, 0.14), 0.25, 1.0)
    alpha_color = line_conf * alpha_proj + (1.0 - line_conf) * alpha_dist

    # 5. Fuse Color-Line Alpha with SAM Probability WITHOUT keeping trapped green pockets!
    # Crucially: if SAM predicted high probability (trapped background between bangs),
    # suppress SAM's influence proportionally to bg_consistency and background proximity.
    is_solid_fg_outside_roi = (fg_eroded) & (roi_c < 0.05) & (dist_bg > fg_min_dist)
    is_solid_fg_inside_roi = (prob_c > 0.75) & (dist_bg > np.maximum(0.32, 0.82 * dist_fb)) & (alpha_color > 0.82)

    # Trapped background suppression factor inside hair ROI
    bg_match = np.clip(1.0 - (dist_bg - tau_bg) / np.maximum(tau_bg * 1.4, 0.08), 0.0, 1.0)
    sam_gated = prob_c * (1.0 - bg_consistency * bg_match)

    # Inside hair ROI, trust color_alpha strongly while still keeping SAM spatial awareness
    # when backdrop is busy (low bg_consistency). Outside ROI, keep out_matte.
    sam_weight = 0.08 + 0.34 * (1.0 - bg_consistency)
    alpha_raw = np.clip((1.0 - sam_weight) * alpha_color + sam_weight * sam_gated, 0.0, 1.0)

    # Boost crisp foreground skin/thick hair core that differs strongly from background
    alpha_raw = np.where(is_solid_fg_outside_roi, np.maximum(alpha_raw, prob_c), alpha_raw)
    alpha_raw = np.where(is_solid_fg_inside_roi, np.maximum(alpha_raw, 0.96), alpha_raw)
    alpha_raw = np.where(neg_c > 0.5, 0.0, alpha_raw)

    # Clean clear background pixels inside ROI before guided filtering so halo doesn't spread outward
    clear_bg = (dist_bg <= tau_bg * 0.78) & (alpha_color < 0.06)
    alpha_raw = np.where(clear_bg, 0.0, alpha_raw)

    # 6. Two-Pass 3-Channel RGB Guided Filter
    # Pass 1: coarse edge snap & pinhole fill
    alpha_gf1 = guided_filter_rgb(
        img_c,
        alpha_raw,
        radius=cfg.guided_radius_coarse,
        eps=cfg.guided_eps_coarse,
    )
    # Preserve strongest sub-pixel peak responses from color alpha along ultra-thin 1px tips
    # that wide box averages slightly dampened.
    tip_preserve = np.where(
        (alpha_color > alpha_gf1) & (dist_bg > tau_bg * 1.45),
        0.55 * alpha_color + 0.45 * alpha_gf1,
        alpha_gf1,
    )
    alpha_mid = np.clip(
        np.where(is_solid_fg_inside_roi, np.maximum(tip_preserve, 0.96), tip_preserve),
        0.0,
        1.0,
    )

    # Pass 2: fine strand-level sharpening aligned cleanly to RGB edges
    alpha_gf2 = guided_filter_rgb(
        img_c,
        alpha_mid,
        radius=cfg.guided_radius_fine,
        eps=cfg.guided_eps_fine,
    )

    # Re-pin near-background and solid foreground after filtering, avoiding residual matte haze
    alpha_final = np.where((dist_bg < tau_bg * 0.65) & (alpha_gf2 < 0.10), 0.0, alpha_gf2)
    alpha_final = np.where(alpha_final < cfg.alpha_floor_clip, 0.0, alpha_final)
    alpha_final = np.where(alpha_final > cfg.alpha_ceil_clip, 1.0, alpha_final)
    alpha_final = np.where(neg_c > 0.5, 0.0, alpha_final)

    # 7. Smooth feathered composite back into full frame via roi_c
    # Expand roi_c gently at its border so transition between refined hair ROI and SAM silhouette has zero seam
    roi_blend = cv2.GaussianBlur(
        roi_c,
        (0, 0),
        sigmaX=max(2.0, cfg.roi_feather_px * 0.5),
        sigmaY=max(2.0, cfg.roi_feather_px * 0.5),
        borderType=cv2.BORDER_REPLICATE,
    )
    # Within core ROI where user painted, commit 100% to refined matte
    roi_blend = np.clip(np.maximum(roi_c, roi_blend * 1.15), 0.0, 1.0)

    base_crop = out_matte[y0:y1, x0:x1]
    blended_crop = roi_blend * alpha_final + (1.0 - roi_blend) * base_crop
    out_matte[y0:y1, x0:x1] = np.clip(blended_crop, 0.0, 1.0)

    return out_matte.astype(np.float32)


def refine_hair_boundary_tracked(
    frame_rgb: np.ndarray,
    sam_prob: np.ndarray,
    norm_hair_rois: Optional[Sequence[Tuple[float, float, float, float]]] = None,
    ref_bbox: Optional[Tuple[int, int, int, int]] = None,
    boundary_band_px: int = 16,
    cfg: Optional[HairMattingConfig] = None,
) -> np.ndarray:
    """Propagate hair matting refinement onto tracked video frames.

    Maps normalized ROI boxes anchored to the object's SAM bounding box on the
    keyframe onto the tracked frame's SAM bounding box, combined with a perimeter
    ribbon around uncertain boundary pixels, then runs `refine_hair_matte`.
    """
    h, w = sam_prob.shape[:2]
    prob = np.clip(np.asarray(sam_prob, dtype=np.float32), 0.0, 1.0)
    core = (prob >= 0.5).astype(np.uint8)
    if not core.any():
        return prob

    roi = np.zeros((h, w), dtype=np.float32)

    cur_bbox = compute_mask_bbox(prob, threshold=0.5, pad=0)
    if norm_hair_rois and cur_bbox is not None:
        bx0, by0, bx1, by1 = cur_bbox
        bw = max(1, bx1 - bx0)
        bh = max(1, by1 - by0)
        for u0, v0, u1, v1 in norm_hair_rois:
            rx0 = max(0, int(round(bx0 + u0 * bw)) - 14)
            ry0 = max(0, int(round(by0 + v0 * bh)) - 14)
            rx1 = min(w, int(round(bx0 + u1 * bw)) + 14)
            ry1 = min(h, int(round(by0 + v1 * bh)) + 14)
            if rx1 > rx0 and ry1 > ry0:
                roi[ry0:ry1, rx0:rx1] = 1.0

    # Also cover thin perimeter edge band on hair-enabled clips when no explicit norm ROI matches
    if float(roi.max()) < 0.1:
        k = max(5, int(boundary_band_px) * 2 + 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        band = ((cv2.dilate(core, kern) ^ cv2.erode(core, kern)) > 0).astype(np.float32)
        roi = band

    return refine_hair_matte(
        frame_rgb=frame_rgb,
        sam_prob=prob,
        roi_mask=roi,
        cfg=cfg,
        apply_boundary_polish=True,
    )
