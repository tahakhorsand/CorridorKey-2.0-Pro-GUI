"""Native SAM 3.1 video mask tracker & interactive segmenter for CorridorKey 2.0.

Uses local weights:
- backend/weights/sam3/sam3.1-INT8-ConvRot.safetensors (with on-the-fly Hadamard ConvRot dequantization)
- backend/weights/sam3/sam3.1-tracker-head.safetensors (multiplex tracker & interactive prompt decoder)

Strictly manages VRAM: moves weights to CUDA only during active segmentation/tracking and
offloads back to CPU + empties CUDA cache immediately in `finally:` blocks.
"""
from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from .sam3_core import SAM3Model, unpack_masks

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SAM31_INT8_PATH = _PROJECT_ROOT / "weights" / "sam3" / "sam3.1-INT8-ConvRot.safetensors"
_DEFAULT_SAM31_TRACKER_HEAD_PATH = _PROJECT_ROOT / "weights" / "sam3" / "sam3.1-tracker-head.safetensors"
_COMFY_SAM31_FP16_FALLBACK = Path(r"E:\ComfyUI\ComfyUI\models\checkpoints\sam3.1_multiplex_fp16.safetensors")


class SAM3NotInstalledError(RuntimeError):
    """Raised when SAM 3.1 weights are missing and fallback is disabled."""


@dataclass
class PromptFrame:
    """Sparse user prompt on a single frame."""
    frame_index: int
    positive_points: list[tuple[int, int]] = field(default_factory=list)
    negative_points: list[tuple[int, int]] = field(default_factory=list)
    box: Optional[tuple[int, int, int, int]] = None
    hair_points: list[tuple[int, int]] = field(default_factory=list)
    negative_strokes: list[tuple[list[tuple[int, int]], float]] = field(default_factory=list)


def _build_hadamard_matrices() -> tuple[torch.Tensor, torch.Tensor]:
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    ) / 2.0
    h64 = torch.kron(torch.kron(h4, h4), h4)
    h256 = torch.kron(h64, h4)
    return h64, h256


def _load_sam31_state_dict(
    int8_path: Path,
    tracker_head_path: Path,
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Load and dequantize sam3.1-INT8-ConvRot.safetensors + tracker head into SAM31Model format."""
    h64, h256 = _build_hadamard_matrices()
    raw_hf: dict[str, torch.Tensor] = {}

    with safe_open(str(int8_path), framework="pt", device="cpu") as f:
        meta_raw = f.metadata() or {}
        rot_fqns: set[str] = set()
        if "unsloth_metadata" in meta_raw:
            meta = json.loads(meta_raw["unsloth_metadata"])
            rot_fqns = set(meta.get("activation_rotation_fqns", []))

        all_keys = set(f.keys())
        processed: set[str] = set()
        for k in sorted(all_keys):
            if k in processed or k.startswith("text_encoder."):
                continue
            if k.endswith("._weight_qdata"):
                base = k[:-len("._weight_qdata")]
                # Only dequantize vision_encoder & text_projection needed by SAM 3.1 tracker/segmenter
                if not (base.startswith("vision_encoder.") or base.startswith("text_projection")):
                    processed.update([k, base + "._weight_scale", base + "._weight_zero_point"])
                    continue
                qdata = f.get_tensor(k).float()
                scale = f.get_tensor(base + "._weight_scale").float()
                zp_key = base + "._weight_zero_point"
                zp = f.get_tensor(zp_key).float() if zp_key in all_keys else torch.zeros_like(scale)
                processed.update([k, base + "._weight_scale", zp_key])
                if scale.ndim == 1:
                    scale = scale.unsqueeze(1)
                if zp.ndim == 1:
                    zp = zp.unsqueeze(1)
                w = (qdata - zp) * scale
                if base in rot_fqns:
                    out_f, in_f = w.shape
                    h_mat = h256 if in_f % 256 == 0 else h64
                    g_sz = h_mat.shape[0]
                    w = (w.reshape(out_f, -1, g_sz) @ h_mat.T).reshape(out_f, in_f)
                raw_hf[base + ".weight"] = w.to(dtype)
            elif k.endswith("._weight_scale") or k.endswith("._weight_zero_point"):
                continue
            else:
                if not (k.startswith("vision_encoder.") or k.startswith("text_projection")):
                    continue
                t = f.get_tensor(k)
                base = k[:-len(".weight")] if k.endswith(".weight") else None
                if base and base in rot_fqns and t.ndim == 2:
                    w = t.float()
                    out_f, in_f = w.shape
                    h_mat = h256 if in_f % 256 == 0 else h64
                    g_sz = h_mat.shape[0]
                    w = (w.reshape(out_f, -1, g_sz) @ h_mat.T).reshape(out_f, in_f)
                    raw_hf[k] = w.to(dtype)
                else:
                    raw_hf[k] = t.to(dtype) if t.is_floating_point() else t

    sd: dict[str, torch.Tensor] = {}
    sd["detector.backbone.vision_backbone.trunk.patch_embed.proj.weight"] = raw_hf.pop(
        "vision_encoder.backbone.embeddings.patch_embeddings.projection.weight"
    )
    pos_emb = raw_hf.pop("vision_encoder.backbone.embeddings.position_embeddings")
    if pos_emb.shape[1] == 576:
        pos_emb = torch.cat([torch.zeros(1, 1, pos_emb.shape[2], dtype=pos_emb.dtype), pos_emb], dim=1)
    sd["detector.backbone.vision_backbone.trunk.pos_embed"] = pos_emb
    sd["detector.backbone.vision_backbone.trunk.ln_pre.weight"] = raw_hf.pop("vision_encoder.backbone.layer_norm.weight")
    sd["detector.backbone.vision_backbone.trunk.ln_pre.bias"] = raw_hf.pop("vision_encoder.backbone.layer_norm.bias")

    for i in range(32):
        p_hf = f"vision_encoder.backbone.layers.{i}."
        p_dt = f"detector.backbone.vision_backbone.trunk.blocks.{i}."
        for s in ("weight", "bias"):
            q = raw_hf.pop(p_hf + f"attention.q_proj.{s}")
            k = raw_hf.pop(p_hf + f"attention.k_proj.{s}")
            v = raw_hf.pop(p_hf + f"attention.v_proj.{s}")
            sd[p_dt + f"attn.qkv.{s}"] = torch.cat([q, k, v], dim=0)
            sd[p_dt + f"attn.proj.{s}"] = raw_hf.pop(p_hf + f"attention.o_proj.{s}")
            sd[p_dt + f"norm1.{s}"] = raw_hf.pop(p_hf + f"layer_norm1.{s}")
            sd[p_dt + f"norm2.{s}"] = raw_hf.pop(p_hf + f"layer_norm2.{s}")
            sd[p_dt + f"mlp.fc1.{s}"] = raw_hf.pop(p_hf + f"mlp.fc1.{s}")
            sd[p_dt + f"mlp.fc2.{s}"] = raw_hf.pop(p_hf + f"mlp.fc2.{s}")

    for idx in range(3):
        p_hf = f"vision_encoder.neck.fpn_layers.{idx}."
        p_dt = f"detector.backbone.vision_backbone.convs.{idx}."
        for s in ("weight", "bias"):
            sd[p_dt + f"conv_1x1.{s}"] = raw_hf.pop(p_hf + f"proj1.{s}")
            sd[p_dt + f"conv_3x3.{s}"] = raw_hf.pop(p_hf + f"proj2.{s}")
            if idx == 0:
                sd[p_dt + f"dconv_2x2_0.{s}"] = raw_hf.pop(p_hf + f"scale_layers.0.{s}")
                sd[p_dt + f"dconv_2x2_1.{s}"] = raw_hf.pop(p_hf + f"scale_layers.2.{s}")
            elif idx == 1:
                sd[p_dt + f"dconv_2x2.{s}"] = raw_hf.pop(p_hf + f"scale_layers.0.{s}")

    for s in ("weight", "bias"):
        if f"text_projection.{s}" in raw_hf:
            sd[f"detector.backbone.language_backbone.resizer.{s}"] = raw_hf.pop(f"text_projection.{s}")

    # Load tracker head weights
    head_source = tracker_head_path if tracker_head_path.is_file() else _COMFY_SAM31_FP16_FALLBACK
    if head_source.is_file():
        with safe_open(str(head_source), framework="pt", device="cpu") as f_trk:
            for k in f_trk.keys():
                if k.startswith("tracker.") or "propagation_convs" in k or "interactive_convs" in k:
                    t = f_trk.get_tensor(k)
                    sd[k] = t.to(dtype) if t.is_floating_point() else t

    # Standardize tracker keys for SAM31Tracker
    for k in list(sd.keys()):
        if k.startswith("tracker.model."):
            sd["tracker." + k[len("tracker.model."):]] = sd.pop(k)
    for k in [k for k in list(sd.keys()) if ".attn.freqs_cis" in k]:
        sd.pop(k)
    for k in [k for k in list(sd.keys()) if k.endswith((".in_proj_weight", ".in_proj_bias"))]:
        t = sd.pop(k)
        base, suffix = k.rsplit(".in_proj_", 1)
        s = ".weight" if suffix == "weight" else ".bias"
        d = t.shape[0] // 3
        sd[base + ".q_proj" + s] = t[:d]
        sd[base + ".k_proj" + s] = t[d : 2 * d]
        sd[base + ".v_proj" + s] = t[2 * d :]
    for k in list(sd.keys()):
        if "sam_mask_decoder.transformer." not in k:
            continue
        new_k = (
            k.replace(".mlp.lin1.", ".mlp.0.")
            .replace(".mlp.lin2.", ".mlp.2.")
            .replace(".norm_final_attn.", ".norm_final.")
        )
        if new_k != k:
            sd[new_k] = sd.pop(k)

    return sd


def refine_hair_matte(
    frame_rgb: np.ndarray,
    base_mask: np.ndarray,
    hair_points: list[tuple[int, int]],
    roi_radius: int = 40,
) -> np.ndarray:
    """Color-guided soft detail refinement for hair, fur, and fuzzy edges.

    Uses RGB frame guidance on top of the clean base mask to preserve fine hair strands,
    flyaways, and semi-transparent edges without green spill or mask disruption.
    """
    h, w = frame_rgb.shape[:2]
    if not hair_points or base_mask is None:
        return base_mask

    roi = np.zeros((h, w), dtype=np.uint8)
    for x, y in hair_points:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(roi, (ix, iy), roi_radius, 255, -1)

    if not np.any(roi):
        return base_mask

    # Sample background color in ROI outside base mask
    bg_mask = (roi > 0) & (base_mask == 0)
    if np.count_nonzero(bg_mask) > 50:
        bg_mean = np.mean(frame_rgb[bg_mask], axis=0)
    else:
        bg_mean = np.mean(frame_rgb[:20, :20], axis=(0, 1))

    # Detect green or blue screen background
    r_bg, g_bg, b_bg = float(bg_mean[0]), float(bg_mean[1]), float(bg_mean[2])
    is_green = (g_bg > r_bg + 5.0) and (g_bg > b_bg + 5.0)
    is_blue = (b_bg > r_bg + 5.0) and (b_bg > g_bg + 5.0)

    r = frame_rgb[:, :, 0].astype(np.float32)
    g = frame_rgb[:, :, 1].astype(np.float32)
    b = frame_rgb[:, :, 2].astype(np.float32)

    if is_green:
        excess_bg = max(1.0, g_bg - max(r_bg, b_bg))
        excess_px = g - np.maximum(r, b)
        hair_alpha = np.clip(1.0 - (excess_px / excess_bg), 0.0, 1.0)
    elif is_blue:
        excess_bg = max(1.0, b_bg - max(r_bg, g_bg))
        excess_px = b - np.maximum(r, g)
        hair_alpha = np.clip(1.0 - (excess_px / excess_bg), 0.0, 1.0)
    else:
        diff_bg = np.linalg.norm(frame_rgb.astype(np.float32) - bg_mean, axis=2)
        fg_mask = (roi > 0) & (base_mask > 200)
        fg_mean = np.mean(frame_rgb[fg_mask], axis=0) if np.count_nonzero(fg_mask) > 50 else np.array([40.0, 40.0, 40.0])
        diff_fg = np.linalg.norm(frame_rgb.astype(np.float32) - fg_mean, axis=2)
        hair_alpha = np.clip(diff_bg / (diff_bg + diff_fg + 1e-4), 0.0, 1.0)

    # Core erosion preserves the solid subject inside the hair ROI
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m_eroded = cv2.erode(base_mask, kernel).astype(np.float32) / 255.0

    refined_roi = np.maximum(m_eroded, hair_alpha)
    roi_soft = cv2.GaussianBlur(roi.astype(np.float32) / 255.0, (15, 15), 0)

    base_f = base_mask.astype(np.float32) / 255.0
    final_f = np.clip((1.0 - roi_soft) * base_f + roi_soft * refined_roi, 0.0, 1.0)
    return (final_f * 255.0).astype(np.uint8)


def clean_binary_mask(
    mask_u8: np.ndarray,
    min_island_area: int = 500,
    max_hole_area: int = 50000,
    negative_points: list[tuple[int, int]] | None = None,
    negative_strokes: list[tuple[list[tuple[int, int]], float]] | None = None,
) -> np.ndarray:
    """Fast morphological cleanup for high-res segmentation masks.

    Removes small disconnected noise islands (< min_island_area) and fills internal
    Swiss-cheese holes on fabrics (< max_hole_area) while strictly preserving user-specified
    holes (e.g. gaps between arms and torso marked with negative points or negative brush strokes)
    and genuine large openings.
    Runs in ~10ms on 4K frames using OpenCV CCOMP hierarchy.
    """
    if mask_u8 is None or mask_u8.size == 0 or not np.any(mask_u8):
        return mask_u8

    h, w = mask_u8.shape[:2]
    area_scale = (h * w) / (3840.0 * 2160.0)
    effective_min_island = max(50, int(round(min_island_area * area_scale)))
    effective_max_hole = max(500, int(round(max_hole_area * area_scale)))

    contours, hierarchy = cv2.findContours(
        (mask_u8 > 0).astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours or hierarchy is None:
        return mask_u8

    cleaned = np.zeros_like(mask_u8)
    # 1. Draw large external contours filled
    for idx, h_info in enumerate(hierarchy[0]):
        if h_info[3] == -1:  # external contour
            if cv2.contourArea(contours[idx]) >= effective_min_island:
                cv2.drawContours(cleaned, contours, idx, 255, -1)

    # 2. Re-cut legitimate large holes OR holes containing negative user points/strokes
    for idx, h_info in enumerate(hierarchy[0]):
        if h_info[3] != -1:  # internal hole
            parent_idx = h_info[3]
            if cv2.contourArea(contours[parent_idx]) >= effective_min_island:
                hole_area = cv2.contourArea(contours[idx])
                keep_hole = (hole_area > effective_max_hole)
                if not keep_hole and negative_points:
                    for nx, ny in negative_points:
                        if cv2.pointPolygonTest(contours[idx], (float(nx), float(ny)), False) >= -2.0:
                            keep_hole = True
                            break
                if not keep_hole and negative_strokes:
                    for s_pts, _ in negative_strokes:
                        for sx, sy in s_pts:
                            if cv2.pointPolygonTest(contours[idx], (float(sx), float(sy)), False) >= -2.0:
                                keep_hole = True
                                break
                        if keep_hole:
                            break
                if keep_hole:
                    cv2.drawContours(cleaned, contours, idx, 0, -1)

    return cleaned


def apply_negative_carveout(
    mask_u8: np.ndarray,
    negative_strokes: list[tuple[list[tuple[int, int]], float]] | None = None,
    negative_points: list[tuple[int, int]] | None = None,
    default_radius: float = 15.0,
) -> np.ndarray:
    """Explicitly zero out pixels marked by user negative brush strokes or negative points.

    Guarantees that negative cuts (e.g. red brush between arm and jacket) are 100% subtracted
    from the mask and are never restored by hole filling or hair blending.
    """
    if mask_u8 is None or mask_u8.size == 0:
        return mask_u8
    has_strokes = bool(negative_strokes)
    has_points = bool(negative_points)
    if not has_strokes and not has_points:
        return mask_u8

    out = mask_u8.copy()
    if negative_points:
        r = max(5, int(round(default_radius)))
        for nx, ny in negative_points:
            cv2.circle(out, (int(round(nx)), int(round(ny))), r, 0, -1)

    if negative_strokes:
        for pts, radius in negative_strokes:
            r = max(1, int(round(radius)))
            for i, (px, py) in enumerate(pts):
                ix, iy = int(round(px)), int(round(py))
                cv2.circle(out, (ix, iy), r, 0, -1)
                if i > 0:
                    prev_ix, prev_iy = int(round(pts[i - 1][0])), int(round(pts[i - 1][1]))
                    cv2.line(out, (prev_ix, prev_iy), (ix, iy), 0, thickness=r * 2)

    return out


class SAM3Tracker:
    """SAM 3.1 video mask tracker and interactive point/box segmenter.

    Runs SAM 3.1 (`sam3.1-INT8-ConvRot.safetensors`) under the hood with strict VRAM lifecycle.
    """

    def __init__(
        self,
        model_id: str = str(_DEFAULT_SAM31_INT8_PATH),
        device: str = "cuda",
        vos_optimized: bool = False,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.vos_optimized = vos_optimized
        self.offload_video_to_cpu = offload_video_to_cpu
        self.offload_state_to_cpu = offload_state_to_cpu
        self._model: Optional[SAM3Model] = None

    def _resolve_int8_path(self) -> Path:
        candidate = Path(self.model_id)
        if candidate.is_file():
            return candidate
        if _DEFAULT_SAM31_INT8_PATH.is_file():
            return _DEFAULT_SAM31_INT8_PATH
        raise SAM3NotInstalledError(f"SAM 3.1 checkpoint not found at {_DEFAULT_SAM31_INT8_PATH}")

    def prepare(
        self,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._model is not None:
            return
        int8_path = self._resolve_int8_path()
        if on_status:
            on_status(f"Loading SAM 3.1 ({int8_path.name})...")
        logger.info("Loading SAM 3.1 weights from %s", int8_path)
        sd = _load_sam31_state_dict(int8_path, _DEFAULT_SAM31_TRACKER_HEAD_PATH, dtype=self.dtype)
        model = SAM3Model(image_model="SAM31", dtype=self.dtype)
        model.load_state_dict(sd, strict=False)
        model.eval()
        self._model = model
        del sd
        gc.collect()
        logger.info("SAM 3.1 model loaded successfully on CPU (ready for on-demand GPU execution).")

    def _segment_single_frame_sam31(
        self,
        frame_rgb: np.ndarray,
        prompt: PromptFrame,
        refine_iterations: int = 2,
    ) -> np.ndarray:
        """Segment a single frame on GPU using SAM 3.1 interactive decoder with points and/or box."""
        self.prepare()
        assert self._model is not None

        h, w = frame_rgb.shape[:2]
        if frame_rgb.dtype == np.uint8:
            img_f = frame_rgb.astype(np.float32) / 255.0
        else:
            img_f = np.clip(frame_rgb[..., :3], 0.0, 1.0).astype(np.float32)

        dev = torch.device(self.device)
        dt = self.dtype

        frame_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0)
        frame_1008 = F.interpolate(frame_t, size=(1008, 1008), mode="bilinear", align_corners=False).to(device=dev, dtype=dt)

        point_inputs = None
        if prompt.positive_points or prompt.negative_points:
            coords = (
                [[px / w * 1008.0, py / h * 1008.0] for px, py in prompt.positive_points]
                + [[nx / w * 1008.0, ny / h * 1008.0] for nx, ny in prompt.negative_points]
            )
            labels = [1] * len(prompt.positive_points) + [0] * len(prompt.negative_points)
            point_inputs = {
                "point_coords": torch.tensor([coords], dtype=dt, device=dev),
                "point_labels": torch.tensor([labels], dtype=torch.int32, device=dev),
            }

        box_inputs = None
        if prompt.box is not None:
            x1, y1, x2, y2 = prompt.box
            x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
            box_inputs = torch.tensor(
                [[[x1 / w * 1008.0, y1 / h * 1008.0], [x2 / w * 1008.0, y2 / h * 1008.0]]],
                dtype=dt,
                device=dev,
            )

        with torch.inference_mode():
            mask_logit = self._model.forward_segment(
                frame_1008,
                point_inputs=point_inputs,
                box_inputs=box_inputs,
            )
            for _ in range(max(0, refine_iterations - 1)):
                mask_logit = self._model.forward_segment(
                    frame_1008,
                    point_inputs=point_inputs,
                    box_inputs=box_inputs,
                    mask_inputs=mask_logit,
                )
            mask_full = F.interpolate(mask_logit.float(), size=(h, w), mode="bilinear", align_corners=False)
            prob = torch.sigmoid(mask_full[0, 0]).cpu().numpy()
            binary = (prob > 0.5).astype(np.uint8) * 255
            cleaned = clean_binary_mask(
                binary,
                negative_points=prompt.negative_points,
                negative_strokes=getattr(prompt, "negative_strokes", None),
            )
            cleaned = apply_negative_carveout(
                cleaned,
                negative_strokes=getattr(prompt, "negative_strokes", None),
                negative_points=prompt.negative_points,
            )

            if getattr(prompt, "hair_points", None):
                refined = refine_hair_matte(frame_rgb, cleaned, prompt.hair_points)
                return apply_negative_carveout(
                    refined,
                    negative_strokes=getattr(prompt, "negative_strokes", None),
                    negative_points=prompt.negative_points,
                )

            return cleaned

    def preview_prompt(self, frame_rgb: np.ndarray, prompt: PromptFrame) -> np.ndarray:
        """Preview a single-frame prompt using SAM 3.1 and immediately release GPU memory."""
        self.prepare()
        assert self._model is not None
        try:
            self._model.to(self.device)
            return self._segment_single_frame_sam31(frame_rgb, prompt)
        finally:
            if self._model is not None:
                try:
                    for buf in self._model.buffers():
                        if buf.is_cuda:
                            buf.data = buf.data.cpu()
                    for m in self._model.modules():
                        if hasattr(m, "_cache") and isinstance(m._cache, dict):
                            m._cache.clear()
                    self._model.to("cpu")
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def track_video(
        self,
        frames: list[np.ndarray],
        prompts: Optional[list[PromptFrame]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        check_cancel: Optional[Callable[[], None]] = None,
        prompt_frames: Optional[list[PromptFrame]] = None,
        **kwargs,
    ) -> list[np.ndarray]:
        if prompts is None:
            prompts = prompt_frames or []
        """Track sparse prompts across all video frames using SAM 3.1 memory propagation."""
        self.prepare(on_progress=on_progress, on_status=on_status)
        assert self._model is not None

        num_frames = len(frames)
        if num_frames == 0:
            return []

        prompt_map = {p.frame_index: p for p in prompts if 0 <= p.frame_index < num_frames}
        if not prompt_map:
            raise ValueError("No valid prompt frames in range.")

        h, w = frames[0].shape[:2]
        dev = torch.device(self.device)
        dt = self.dtype

        try:
            self._model.to(dev)
            if check_cancel:
                check_cancel()

            # Single-frame fast path
            if num_frames == 1:
                first_prompt = next(iter(prompt_map.values()))
                mask = self._segment_single_frame_sam31(frames[0], first_prompt)
                if on_progress:
                    on_progress(1, 1)
                return [mask]

            first_key = min(prompt_map.keys())
            if on_status:
                on_status(f"SAM 3.1: Segmenting keyframe {first_key}...")

            init_mask_u8 = self._segment_single_frame_sam31(frames[first_key], prompt_map[first_key])
            init_mask_t = torch.from_numpy((init_mask_u8 > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device=dev, dtype=dt)

            def _frames_to_cpu_tensor(frame_list: list[np.ndarray]) -> torch.Tensor:
                arrs = []
                for f in frame_list:
                    if f.dtype == np.uint8:
                        arrs.append(torch.from_numpy(f).permute(2, 0, 1).to(torch.float16).div_(255.0))
                    else:
                        arrs.append(torch.from_numpy(np.clip(f[..., :3], 0.0, 1.0)).permute(2, 0, 1).to(torch.float16))
                return torch.stack(arrs, dim=0)

            out_masks: list[Optional[np.ndarray]] = [None] * num_frames
            out_masks[first_key] = init_mask_u8

            class _PBarAdapter:
                def __init__(self, start_count: int):
                    self.count = start_count

                def update(self, n: int = 1):
                    if check_cancel:
                        check_cancel()
                    self.count += n
                    if on_progress:
                        on_progress(min(self.count, num_frames), num_frames)

            pbar = _PBarAdapter(0)

            # Propagate forward from first_key .. num_frames - 1
            if first_key < num_frames - 1:
                if on_status:
                    on_status("SAM 3.1: Propagating mask forward...")
                fwd_frames = _frames_to_cpu_tensor(frames[first_key:])
                with torch.inference_mode():
                    res_fwd = self._model.forward_video(
                        images=fwd_frames,
                        initial_masks=init_mask_t,
                        pbar=pbar,
                        target_device=dev,
                        target_dtype=dt,
                    )
                    packed_fwd = res_fwd.get("packed_masks")
                    if packed_fwd is not None:
                        for i in range(packed_fwd.shape[0]):
                            unpacked = unpack_masks(packed_fwd[i : i + 1, 0]).unsqueeze(1).float()
                            resized = F.interpolate(unpacked, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
                            prob_val = torch.sigmoid(resized).cpu().numpy()
                            raw_u8 = ((prob_val > 0.5).astype(np.uint8)) * 255
                            curr_frame_idx = first_key + i
                            curr_prompt = prompt_map.get(curr_frame_idx, prompt_map[first_key])
                            cleaned_mask = clean_binary_mask(
                                raw_u8,
                                negative_points=curr_prompt.negative_points if curr_prompt else None,
                                negative_strokes=getattr(curr_prompt, "negative_strokes", None) if curr_prompt else None,
                            )
                            cleaned_mask = apply_negative_carveout(
                                cleaned_mask,
                                negative_strokes=getattr(curr_prompt, "negative_strokes", None) if curr_prompt else None,
                                negative_points=curr_prompt.negative_points if curr_prompt else None,
                            )
                            frame_hair_pts = curr_prompt.hair_points if curr_prompt else []
                            if frame_hair_pts:
                                refined = refine_hair_matte(frames[curr_frame_idx], cleaned_mask, frame_hair_pts)
                                out_masks[curr_frame_idx] = apply_negative_carveout(
                                    refined,
                                    negative_strokes=getattr(curr_prompt, "negative_strokes", None) if curr_prompt else None,
                                    negative_points=curr_prompt.negative_points if curr_prompt else None,
                                )
                            else:
                                out_masks[curr_frame_idx] = cleaned_mask
                del fwd_frames, res_fwd

            # Propagate backward from first_key .. 0 if first_key > 0
            if first_key > 0:
                if on_status:
                    on_status("SAM 3.1: Propagating mask backward...")
                bwd_frames = _frames_to_cpu_tensor(frames[first_key::-1])
                with torch.inference_mode():
                    res_bwd = self._model.forward_video(
                        images=bwd_frames,
                        initial_masks=init_mask_t,
                        pbar=pbar,
                        target_device=dev,
                        target_dtype=dt,
                    )
                    packed_bwd = res_bwd.get("packed_masks")
                    if packed_bwd is not None:
                        for i in range(1, packed_bwd.shape[0]):
                            unpacked = unpack_masks(packed_bwd[i : i + 1, 0]).unsqueeze(1).float()
                            resized = F.interpolate(unpacked, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
                            prob_val = torch.sigmoid(resized).cpu().numpy()
                            raw_u8 = ((prob_val > 0.5).astype(np.uint8)) * 255
                            curr_frame_idx = first_key - i
                            curr_prompt = prompt_map.get(curr_frame_idx, prompt_map[first_key])
                            cleaned_mask = clean_binary_mask(
                                raw_u8,
                                negative_points=curr_prompt.negative_points if curr_prompt else None,
                                negative_strokes=getattr(curr_prompt, "negative_strokes", None) if curr_prompt else None,
                            )
                            cleaned_mask = apply_negative_carveout(
                                cleaned_mask,
                                negative_strokes=getattr(curr_prompt, "negative_strokes", None) if curr_prompt else None,
                                negative_points=curr_prompt.negative_points if curr_prompt else None,
                            )
                            frame_hair_pts = curr_prompt.hair_points if curr_prompt else []
                            if frame_hair_pts:
                                refined = refine_hair_matte(frames[curr_frame_idx], cleaned_mask, frame_hair_pts)
                                out_masks[curr_frame_idx] = apply_negative_carveout(
                                    refined,
                                    negative_strokes=getattr(curr_prompt, "negative_strokes", None) if curr_prompt else None,
                                    negative_points=curr_prompt.negative_points if curr_prompt else None,
                                )
                            else:
                                out_masks[curr_frame_idx] = cleaned_mask
                del bwd_frames, res_bwd

            if on_progress:
                on_progress(num_frames, num_frames)

            return [m if m is not None else np.zeros((h, w), dtype=np.uint8) for m in out_masks]
        finally:
            if self._model is not None:
                try:
                    for buf in self._model.buffers():
                        if buf.is_cuda:
                            buf.data = buf.data.cpu()
                    for m in self._model.modules():
                        if hasattr(m, "_cache") and isinstance(m._cache, dict):
                            m._cache.clear()
                    self._model.to("cpu")
                except Exception:
                    pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def offload(self) -> None:
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self) -> None:
        self.offload()

