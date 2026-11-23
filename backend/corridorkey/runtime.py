from __future__ import annotations

import contextlib
import gc
import json
import os
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import color_utils as cu
from .model import CorridorKeyV3
from .video_io import (
    ExrSequenceWriter,
    ThreadedVideoWriter,
    WriterConfig,
    apply_linear_adjust,
    ceil_to_multiple,
    center_pad_to_shape,
    frame_paths,
    has_linear_adjust,
    iter_frame_dir_chunks,
    iter_video_frame_chunks,
    load_initial_hint_frame,
    make_checker,
    make_hint_reader,
    none_path,
    probe_frame_dir_meta,
    probe_video_meta,
    raw_clip_name,
    resize_hwc_lanczos,
    safe_remove,
    scaled_hw,
    taper_weights,
    window_starts,
    SequentialFrameDirWindowReader,
    SequentialVideoWindowReader,
)


@dataclass
class RuntimePaths:
    package_root: Path
    weights: Path
    foundation_stats: Path
    rvm_ckpt: Path
    moge_checkpoint: Path
    cradio_dir: Path

    @staticmethod
    def _resolve_user_or_package_path(root: Path, value, default: str) -> Path:
        if value is None:
            return (root / default).resolve()
        path = Path(value)
        if path.is_absolute():
            return path.resolve()
        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path
        return (root / path).resolve()

    @staticmethod
    def _prefer_safetensors(root: Path, stem: str) -> str:
        # Legacy pickle .pt files are still accepted if that is all that is present.
        if not (root / f"{stem}.safetensors").exists() and (root / f"{stem}.pt").exists():
            return f"{stem}.pt"
        return f"{stem}.safetensors"

    @classmethod
    def from_package(cls, package_root: str | os.PathLike,
                     weights: str | os.PathLike | None = None):
        root = Path(package_root).resolve()
        return cls(
            package_root=root,
            weights=cls._resolve_user_or_package_path(
                root, weights, cls._prefer_safetensors(root, "weights/corridorkey_v2")),
            foundation_stats=(
                root / cls._prefer_safetensors(root, "weights/foundation_stats")).resolve(),
            rvm_ckpt=(root / "vfms/rvm/pretrain_rvm_small16_256_204031069.npz").resolve(),
            moge_checkpoint=(root / "vfms/moge/model.pt").resolve(),
            cradio_dir=(root / "vfms/cradio/C-RADIOv4-SO400M").resolve(),
        )

    def validate(self) -> None:
        missing = [
            p for p in (
                self.weights, self.foundation_stats, self.rvm_ckpt,
                self.moge_checkpoint, self.cradio_dir,
            )
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "missing runtime asset(s): " + ", ".join(str(p) for p in missing)
                + "\nRun `python scripts/download_weights.py` to fetch them.")


@dataclass
class InferenceConfig:
    hann_chunk: int = 80
    hann_stride: int = 40
    carry_hint: bool = True
    hint_quality: float = 0.95
    carry_hint_quality: float = 0.95
    shape_quantum: int = 128
    model_scale: float = 1.0
    start_frame: int = 0
    num_frames: int = -1
    start_mode: str = "begin"
    seed: int = 0
    low_vram: bool = False
    # Pick hann_chunk/hann_stride/low_vram per clip from free GPU memory.
    auto_vram: bool = False
    low_vram_foundation_chunk: int = 4
    low_vram_mlp_chunk_tokens: int = 65536
    low_vram_enc_frame_chunk: int = 1
    low_vram_swin_window_batch: int = 16
    low_vram_full_attn_query_chunk_tokens: int = 2048
    stage2_batch: int = 1
    outputs: set[str] = field(default_factory=lambda: {"alpha", "fg"})
    temp_dir: Path = Path(tempfile.gettempdir()) / "corridorkey_v2"
    linear_brightness: float = 1.0
    linear_contrast: float = 1.0
    linear_contrast_pivot: float = 0.18
    frame_dir_fps: float = 24.0
    frame_dir_linear: bool = False
    cutout_linear: bool = False
    despill_strength: float = 0.5
    screen_color: str = "green"
    output_format: str = "video"
    exr_dtype: str = "half"


def _amp_dtype() -> torch.dtype:
    if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    return torch.float16


def _autocast_ctx(device):
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if dev_type == "cuda":
        return torch.autocast("cuda", dtype=_amp_dtype())
    return contextlib.nullcontext()


def _raise_cancelled(source: str, frame_idx: int = 0) -> None:
    try:
        from backend.errors import JobCancelledError
        raise JobCancelledError(source, frame_idx)
    except ImportError:
        raise RuntimeError("Cancelled by user")


# Peak GPU memory in GiB by megapixels of model input, measured at each Hann
# chunk size. Keys are low_vram False/True.
_VRAM_PEAK_GB = {
    False: ((8, 24, 48, 80), (
        (0.98, (6.19, 7.52, 9.77, 12.89)),
        (2.21, (7.50, 11.33, 18.66, 24.89)),
        (3.93, (9.06, 14.78, 23.92, 33.19)),
        (8.36, (13.94, 25.96, 45.15, 70.45)),
    )),
    True: ((1, 2, 4, 8, 24, 48, 80), (
        (0.98, (2.71, 2.78, 2.94, 3.25, 4.54, 6.57, 9.50)),
        (2.21, (2.80, 2.97, 3.35, 4.12, 7.69, 13.20, 20.56)),
        (3.93, (2.92, 3.23, 3.85, 5.09, 10.09, 17.57, 27.75)),
        (8.36, (3.25, 3.92, 5.28, 8.38, 20.78, 39.66, 64.65)),
    )),
}
_VRAM_MARGIN_SCALE = 1.1
_VRAM_MARGIN_GB = 0.3


def _interp(x: float, points) -> float:
    """Piecewise-linear through (x, y) points, extrapolating past the ends."""
    points = list(points)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1 or (x1, y1) == points[-1]:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[0][1]


def estimate_peak_vram_gb(megapixels: float, frames: int, low_vram: bool) -> float:
    """Estimated peak GPU memory for a Hann window of `frames` frames."""
    chunks, rows = _VRAM_PEAK_GB[low_vram]
    frames = max(frames, chunks[0])
    per_chunk = [
        max(_interp(megapixels, [(mp, peaks[i]) for mp, peaks in rows]), rows[0][1][i])
        for i in range(len(chunks))
    ]
    return _interp(frames, zip(chunks, per_chunk))


def select_vram_profile(shape_hw, num_frames: int, budget_gb: float):
    """Largest Hann window that fits `budget_gb`, as (chunk, stride, low_vram, estimate).

    Bigger windows give better temporal consistency, and low-VRAM mode is
    exact, so a larger low-VRAM window is preferred over a smaller normal one.
    Falls back to a single-frame window when nothing is estimated to fit.
    """
    megapixels = shape_hw[0] * shape_hw[1] / 1e6
    for chunk in reversed(_VRAM_PEAK_GB[True][0]):
        for low_vram in (False, True):
            if chunk not in _VRAM_PEAK_GB[low_vram][0]:
                continue
            estimate = estimate_peak_vram_gb(megapixels, min(chunk, num_frames), low_vram)
            if estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB <= budget_gb:
                return chunk, max(1, chunk // 2), low_vram, estimate
    return 1, 1, True, estimate


def apply_auto_vram(cfg: InferenceConfig, shape_hw, num_frames: int) -> None:
    if not torch.cuda.is_available():
        cfg.hann_chunk, cfg.hann_stride, cfg.low_vram = 8, 4, True
        return
    free_bytes, _ = torch.cuda.mem_get_info()
    # Memory this process already holds (the loaded model) is part of the budget.
    budget_gb = (free_bytes + torch.cuda.memory_reserved()) / 2**30
    chunk, stride, low_vram, estimate = select_vram_profile(shape_hw, num_frames, budget_gb)
    cfg.hann_chunk, cfg.hann_stride, cfg.low_vram = chunk, stride, low_vram
    print(f"  Auto VRAM: {budget_gb:.1f} GB available at {shape_hw[1]}x{shape_hw[0]} -> "
          f"hann_chunk={chunk}, hann_stride={stride}, low_vram={low_vram} "
          f"(estimated peak {estimate:.1f} GB)")
    if estimate * _VRAM_MARGIN_SCALE + _VRAM_MARGIN_GB > budget_gb:
        print("  [WARNING] Even a single-frame window may not fit this GPU at this "
              "resolution. Inference may run out of memory.")


def prepare_runtime_environment(paths: RuntimePaths) -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    moge_root = paths.package_root / "third_party/MoGe"
    if str(moge_root) not in sys.path:
        sys.path.insert(0, str(moge_root))


def _resolve_package_path(paths: RuntimePaths, value: str | None) -> str | None:
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = paths.package_root / p
    return str(p.resolve())


def load_runtime_checkpoint(path: Path) -> dict:
    """Read a runtime checkpoint from .safetensors (preferred) or legacy .pt."""
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        from safetensors.torch import load_file
        with safe_open(str(path), framework="pt") as f:
            header = f.metadata() or {}
        return {
            "format": header.get("format"),
            "model_config": json.loads(header.get("model_config", "{}")),
            "metadata": json.loads(header.get("metadata", "{}")),
            "state_dict": load_file(str(path), device="cpu"),
        }
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def load_model(paths: RuntimePaths, device: str = "cuda") -> CorridorKeyV3:
    paths.validate()
    prepare_runtime_environment(paths)
    ckpt = load_runtime_checkpoint(paths.weights)
    if ckpt.get("format") != "corridorkey_v2_runtime_checkpoint":
        raise RuntimeError(f"{paths.weights} is not a CorridorKey v2 runtime checkpoint")
    cfg = dict(ckpt["model_config"])
    cfg["cradio_repo"] = str(paths.cradio_dir)
    cfg["moge_checkpoint"] = str(paths.moge_checkpoint)
    cfg["rvm_ckpt"] = str(paths.rvm_ckpt)
    cfg["foundation_stats_path"] = str(paths.foundation_stats)
    cfg["gradient_checkpoint"] = False
    cfg["device"] = device
    cfg["stage1_recurrent_64_sites"] = ()

    print(f"Loading CorridorKey v2 runtime weights: {paths.weights}")
    model = CorridorKeyV3(**cfg)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    train_missing = [k for k in missing if not k.startswith("foundations.")]
    train_unexpected = [k for k in unexpected if not k.startswith("foundations.")]
    if train_missing or train_unexpected:
        raise RuntimeError(
            "runtime checkpoint did not match model: "
            f"missing={train_missing[:8]} unexpected={train_unexpected[:8]}")
    print(f"  loaded weights; omitted frozen foundation tensors={len(missing)}")
    model.eval()
    del ckpt
    return model


def configure_low_vram_inference(model, cfg: InferenceConfig):
    if cfg.low_vram:
        enc_chunk = max(0, int(cfg.low_vram_enc_frame_chunk))
        mlp_chunk = max(0, int(cfg.low_vram_mlp_chunk_tokens))
        swin_batch = max(0, int(cfg.low_vram_swin_window_batch))
        attn_chunk = max(0, int(cfg.low_vram_full_attn_query_chunk_tokens))
    else:
        enc_chunk = mlp_chunk = swin_batch = attn_chunk = 0

    model.stage1.low_vram_enc_frame_chunk = enc_chunk
    for module in model.modules():
        name = module.__class__.__name__
        if name in ("SwiGLU", "FusionLinSwiglu"):
            module.inference_chunk_tokens = mlp_chunk
        if name == "SwinAttention3D":
            module.inference_window_batch = swin_batch
        if name == "CSwinAttention3D":
            module.inference_strip_batch = swin_batch
        if name in ("FullSpatialAttention2D", "FullAttention3D"):
            module.inference_query_chunk_tokens = attn_chunk


def _build_foundation_inputs(rgb_bt: torch.Tensor):
    B, T, _, H, W = rgb_bt.shape
    flat = rgb_bt.reshape(B * T, 3, H, W)
    rgb_radio = F.interpolate(flat, size=(H // 4, W // 4), mode="area")
    H_moge, W_moge = (H * 7) // 32, (W * 7) // 32
    rgb_moge = F.interpolate(flat, size=(H_moge, W_moge), mode="area")
    return (
        rgb_radio.reshape(B, T, 3, H // 4, W // 4),
        rgb_moge.reshape(B, T, 3, H_moge, W_moge),
    )


def _next_or_none(iterator):
    try:
        return next(iterator)
    except StopIteration:
        return None


@torch.no_grad()
def precompute_rvm_to_memmap(model, source, meta, rvm_path: Path,
                             *, chunk: int, reader_kind: str, is_cancelled=None):
    assert model.foundations.rvm is not None
    device = next(model.foundations.rvm.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    H_e, W_e = H // 4, W // 4
    D = model.foundations.rvm.embed_dim
    h_e, w_e = H_e // 16, W_e // 16
    print(f"    RVM continuous prepass: T={T_all}, chunk={chunk}, "
          f"input={W_e}x{H_e}, grid={w_e}x{h_e}")
    out = np.memmap(rvm_path, dtype=np.float16, mode="w+",
                    shape=(T_all, D, h_e, w_e))
    mean = model.foundations.rvm_mean.view(1, 1, -1, 1, 1).float().cpu()
    std = model.foundations.rvm_std.view(1, 1, -1, 1, 1).float().cpu().clamp(min=1e-6)
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    state = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            if is_cancelled is not None and is_cancelled():
                _raise_cancelled(str(source))
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            rgb = display.unsqueeze(0).to(device, non_blocking=True).float()
            rgb_rvm = F.interpolate(
                rgb.reshape(Tc, 3, H, W), size=(H_e, W_e),
                mode="bilinear", align_corners=False,
            ).reshape(1, Tc, 3, H_e, W_e).clamp(0.0, 1.0)
            with _autocast_ctx(device):
                feats, state = model.foundations.rvm(rgb_rvm, state=state)
            feats_cpu = (feats.float().cpu() - mean) / std
            out[t0:t0 + Tc] = feats_cpu[0].numpy().astype(np.float16)
            del rgb, rgb_rvm, feats, feats_cpu, display
    out.flush()
    del out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def precompute_foundations_to_memmap(model, source, meta, prefix: Path,
                                     *, chunk: int, reader_kind: str, is_cancelled=None):
    device = next(model.foundations.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    h64, w64 = H // 64, W // 64
    C_cradio = int(getattr(model.stage1, "cradio_dim", 0) or 0)
    C_moge = int(getattr(model.stage1, "moge_dim", 0) or 0)
    if C_moge <= 0:
        raise RuntimeError("MoGe is required for this model")
    paths = {
        "moge": str(prefix) + "_moge.f32",
        "moge_shape": (T_all, C_moge, h64, w64),
    }
    moge_out = np.memmap(paths["moge"], dtype=np.float32, mode="w+",
                         shape=paths["moge_shape"])
    cradio_out = None
    if C_cradio > 0:
        paths["cradio"] = str(prefix) + "_cradio.f32"
        paths["cradio_shape"] = (T_all, C_cradio, h64, w64)
        cradio_out = np.memmap(paths["cradio"], dtype=np.float32, mode="w+",
                               shape=paths["cradio_shape"])
    print(f"    frozen VFM prepass: T={T_all}, grid={w64}x{h64}, chunk={chunk}")
    if reader_kind == "video":
        chunks = iter_video_frame_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    else:
        chunks = iter_frame_dir_chunks(str(source), meta, chunk_size=chunk, need_model=False)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_next_or_none, chunks)
        while True:
            if is_cancelled is not None and is_cancelled():
                _raise_cancelled(str(source))
            item = future.result()
            if item is None:
                break
            future = pool.submit(_next_or_none, chunks)
            t0, _model, display = item
            Tc = display.shape[0]
            rgb = display.unsqueeze(0).to(device, non_blocking=True)
            rgb_radio, rgb_moge = _build_foundation_inputs(rgb)
            with _autocast_ctx(device):
                ff = model.foundations(
                    rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                    rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                )
            moge_out[t0:t0 + Tc] = ff["moge"].float().cpu().numpy()
            if cradio_out is not None and "cradio" in ff:
                cradio_out[t0:t0 + Tc] = ff["cradio"].float().cpu().numpy()
            del rgb, rgb_radio, rgb_moge, ff, display
    moge_out.flush()
    del moge_out
    if cradio_out is not None:
        cradio_out.flush()
        del cradio_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return paths


def _target_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _phase1_setup(model, *, stage1_on_gpu: bool = True):
    dev = _target_device_str()
    model.foundations.to(dev)
    # Low-VRAM mode precomputes foundation features first, so stage 1 waits on CPU.
    if stage1_on_gpu:
        model.stage1.to(dev)
    else:
        model.stage1.cpu()
    model.stage2.cpu()
    if dev == "cuda":
        torch.cuda.empty_cache()


def _phase2_setup(model):
    dev = _target_device_str()
    model.foundations.cpu()
    model.stage1.cpu()
    if dev == "cuda":
        torch.cuda.empty_cache()
    model.stage2.to(dev)
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def hann_stage1_stream(model, source, meta, *, reader_cls, reader_kind: str,
                       chunk: int, stride: int, baton_path: Path,
                       initial_hint, initial_hint_quality: float,
                       full_hint_video_path: str | None,
                       full_hint_quality: float,
                       rvm_path: Path | None,
                       foundation_paths,
                       carry_hint_from_aux: bool,
                       carry_hint_quality: float,
                       on_baton_frames,
                       on_stage1_window=None,
                       is_cancelled=None,
                       initial_hint_frame_index: int = 0):
    device = next(model.stage1.parameters()).device
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    d_16 = model.stage1.d_16
    h16, w16 = H // 16, W // 16
    starts = window_starts(T_all, chunk, stride)
    overlap = max(0, chunk - stride)
    baton_fp16 = None
    if on_baton_frames is None:
        baton_fp16 = np.memmap(baton_path, dtype=np.float16, mode="w+",
                               shape=(T_all, d_16, h16, w16))
    live_acc = {}
    live_w = {}
    print(f"    Hann stage1: chunk={chunk}, stride={stride}, "
          f"windows={len(starts)}, baton_grid={w16}x{h16}")

    rvm_map = None
    if rvm_path is not None:
        D = model.foundations.rvm.embed_dim
        rvm_map = np.memmap(rvm_path, dtype=np.float16, mode="r",
                            shape=(T_all, D, H // 64, W // 64))
    cradio_map = moge_map = None
    if foundation_paths is not None:
        moge_map = np.memmap(foundation_paths["moge"], dtype=np.float32,
                             mode="r", shape=tuple(foundation_paths["moge_shape"]))
        if model.stage1.cradio_dim > 0:
            cradio_map = np.memmap(foundation_paths["cradio"], dtype=np.float32,
                                   mode="r", shape=tuple(foundation_paths["cradio_shape"]))

    reader = reader_cls(str(source), meta)
    hint_reader = make_hint_reader(full_hint_video_path, meta)
    carry_cache = {}
    stage2_frame_cache = {}

    def flush_before(t_limit: int):
        ready = []
        for k in sorted(k for k in live_acc if k < t_limit):
            frame = (live_acc.pop(k) / max(live_w.pop(k), 1e-6)).astype(np.float16)
            if on_baton_frames is None:
                baton_fp16[k] = frame
            else:
                model_frame = stage2_frame_cache.pop(k)
                ready.append((k, frame, model_frame))
        if ready:
            on_baton_frames(ready)

    try:
        for i, t0 in enumerate(starts):
            if is_cancelled is not None and is_cancelled():
                _raise_cancelled(str(source), t0)
            if on_stage1_window is not None:
                on_stage1_window(i + 1, len(starts))
            t1 = min(t0 + chunk, T_all)
            Tc = t1 - t0
            model_chunk, display_chunk, review_chunk = reader.window(t0, t1)
            hint_cpu = torch.zeros(1, Tc, 1, H, W, dtype=torch.bfloat16)
            hint_quality_cpu = torch.zeros(1, Tc, dtype=torch.bfloat16)
            hint_present = torch.zeros(1, Tc, dtype=torch.bool)
            if hint_reader is not None:
                hint_cpu[0] = hint_reader.window(t0, t1)
                hint_quality_cpu.fill_(float(full_hint_quality))
                hint_present.fill_(True)
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs in carry_cache:
                    hint_cpu[0, j] = carry_cache[t_abs]
                    hint_quality_cpu[0, j] = float(carry_hint_quality)
                    hint_present[0, j] = True
            if initial_hint is not None and (t0 <= initial_hint_frame_index < t1):
                j_hint = initial_hint_frame_index - t0
                hint_cpu[0, j_hint] = initial_hint.to(dtype=hint_cpu.dtype)
                hint_quality_cpu[0, j_hint] = float(initial_hint_quality)
                hint_present[0, j_hint] = True
            if not bool(hint_present.any().item()):
                hint_quality_cpu.fill_(-1.0)

            if on_baton_frames is not None:
                for j, t_abs in enumerate(range(t0, t1)):
                    stage2_frame_cache[t_abs] = model_chunk[j]

            rgb_c = model_chunk.unsqueeze(0).to(device, non_blocking=True)
            hint_c = hint_cpu.to(device, non_blocking=True)
            hint_quality_c = hint_quality_cpu.to(device, non_blocking=True)
            ff = rgb_radio = rgb_moge = None
            if foundation_paths is None:
                rgb_found = display_chunk.unsqueeze(0).to(device, non_blocking=True)
                rgb_radio, rgb_moge = _build_foundation_inputs(rgb_found)
                with _autocast_ctx(device):
                    ff = model.foundations(
                        rgb_cradio=rgb_radio.reshape(Tc, 3, *rgb_radio.shape[-2:]),
                        rgb_moge=rgb_moge.reshape(Tc, 3, *rgb_moge.shape[-2:]),
                    )
                cradio_feat = (
                    ff["cradio"].reshape(1, Tc, -1, H // 64, W // 64)
                    if "cradio" in ff and getattr(model.stage1, "cradio_dim", 0) > 0
                    else None
                )
                moge_feat = ff["moge"].reshape(1, Tc, -1, H // 64, W // 64)
            else:
                moge_feat = torch.from_numpy(np.asarray(moge_map[t0:t1]).copy())
                moge_feat = moge_feat.unsqueeze(0).to(device, non_blocking=True).float()
                if cradio_map is not None:
                    cradio_feat = torch.from_numpy(np.asarray(cradio_map[t0:t1]).copy())
                    cradio_feat = cradio_feat.unsqueeze(0).to(device, non_blocking=True).float()
                else:
                    cradio_feat = None
            rvm_feat = None
            if rvm_map is not None:
                rvm_feat = torch.from_numpy(np.asarray(rvm_map[t0:t1]).copy())
                rvm_feat = rvm_feat.unsqueeze(0).to(device, non_blocking=True).float()
            with _autocast_ctx(device):
                s1 = model.stage1(
                    rgb_c, hint_c, moge_feat, cradio_feat=cradio_feat,
                    rvm_feat=rvm_feat, t_stride=1.0,
                    hint_quality=hint_quality_c,
                )
            baton_gpu = s1["baton"][0].float()
            if carry_hint_from_aux:
                aux_alpha = s1["aux"][0, :, 0:1].float()
                aux_full = F.interpolate(
                    aux_alpha, size=(H, W), mode="bilinear",
                    align_corners=False).clamp(0.0, 1.0)
                aux_cpu = aux_full.to("cpu", dtype=torch.bfloat16)
                for j, t_abs in enumerate(range(t0, t1)):
                    carry_cache[t_abs] = aux_cpu[j].clone()
                next_start = starts[i + 1] if i + 1 < len(starts) else T_all
                for old_t in [k for k in carry_cache if k < next_start]:
                    del carry_cache[old_t]
                del aux_alpha, aux_full, aux_cpu
            w = taper_weights(Tc, overlap, is_first=(i == 0),
                              is_last=(i == len(starts) - 1),
                              device="cpu", dtype=torch.float32)
            weighted = (baton_gpu * w.to(device).view(Tc, 1, 1, 1)).to("cpu").numpy()
            w_np = w.numpy()
            for j, t_abs in enumerate(range(t0, t1)):
                if t_abs not in live_acc:
                    live_acc[t_abs] = weighted[j].copy()
                    live_w[t_abs] = float(w_np[j])
                else:
                    live_acc[t_abs] += weighted[j]
                    live_w[t_abs] += float(w_np[j])
            next_start = starts[i + 1] if i + 1 < len(starts) else T_all
            del s1, baton_gpu, weighted, ff, cradio_feat, moge_feat, rvm_feat
            del rgb_c, hint_c, hint_quality_c, hint_cpu, hint_quality_cpu
            del model_chunk, display_chunk, review_chunk
            flush_before(next_start)
            print(f"      window {i + 1}/{len(starts)} [{t0}:{t1}]", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        flush_before(T_all + 1)
    finally:
        reader.close()
        if hint_reader is not None:
            hint_reader.close()
        live_acc.clear()
        live_w.clear()
        carry_cache.clear()
        stage2_frame_cache.clear()
        if baton_fp16 is not None:
            try:
                baton_fp16.flush()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _crop_native_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    h, w = meta["model_native_hw"]
    pad_l, _, pad_t, _ = meta["pad"]
    return x[..., pad_t:pad_t + h, pad_l:pad_l + w]


def _resize_output_torch(x: torch.Tensor, meta: dict) -> torch.Tensor:
    out_hw = tuple(meta["output_hw"])
    if tuple(meta["model_native_hw"]) == out_hw and tuple(x.shape[-2:]) == out_hw:
        return x
    return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)


def _linear_adjust_inverse_torch(linear: torch.Tensor, meta: dict) -> torch.Tensor:
    gain = float(meta.get("linear_brightness", 1.0))
    contrast = float(meta.get("linear_contrast", 1.0))
    if abs(gain - 1.0) <= 1e-6 and abs(contrast - 1.0) <= 1e-6:
        return linear
    pivot = float(meta.get("linear_contrast_pivot", 0.18))
    return (linear / gain - pivot) / contrast + pivot


def _despill_average_srgb_torch(
    fg: torch.Tensor,
    strength: float = 0.5,
    screen_color: str = "green",
) -> torch.Tensor:
    if strength <= 0.0:
        return fg
    r = fg[:, 0:1]
    g = fg[:, 1:2]
    b = fg[:, 2:3]
    if str(screen_color).lower() == "blue":
        spill = (b - (r + g) * 0.5).clamp(min=0.0)
        full = torch.cat([r + spill * 0.5, g + spill * 0.5, b - spill], dim=1)
    else:
        spill = (g - (r + b) * 0.5).clamp(min=0.0)
        full = torch.cat([r + spill * 0.5, g - spill, b + spill * 0.5], dim=1)
    if strength >= 1.0:
        return full
    return fg * (1.0 - strength) + full * strength


def _srgb_to_bgr_uint16_cpu(x: torch.Tensor) -> np.ndarray:
    x = x.clamp(0.0, 1.0)
    x = x[:, [2, 1, 0]].permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (x * 65535.0).astype(np.uint16)


def _srgb_to_bgra_uint16_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    rgb = rgb.clamp(0.0, 1.0)
    alpha = alpha.clamp(0.0, 1.0)
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    bgra = bgra.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    return (bgra * 65535.0).astype(np.uint16)


def _chw_to_bgr_f32_cpu(x: torch.Tensor) -> np.ndarray:
    return (x[:, [2, 1, 0]].permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _chw_to_bgra_f32_cpu(rgb: torch.Tensor, alpha: torch.Tensor) -> np.ndarray:
    bgra = torch.cat([rgb[:, [2, 1, 0]], alpha], dim=1)
    return (bgra.permute(0, 2, 3, 1)
            .contiguous().cpu().numpy().astype(np.float32, copy=False))


def _render_predictions_exr_torch(meta: dict, outputs: set[str],
                                  checker: torch.Tensor,
                                  logits: torch.Tensor) -> dict[str, np.ndarray]:
    """Render a stage-2 batch as unclamped scene-linear float32 NHWC arrays.

    HDR contract: `fg`, `cutout`, and `checker` color stays in decoded linear
    light with no [0, 1] clamp and no quantization, so values above 1.0 (and
    negative values the model may emit) survive into the EXR files. Alpha is a
    matte and stays clamped to [0, 1].
    """
    alpha_model = logits[:, 0:1].float()
    pa_model = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
    despill_strength = float(meta.get("despill_strength", 0.5))
    screen_color = str(meta.get("screen_color", "green"))

    pa = _resize_output_torch(pa_model, meta).clamp(0.0, 1.0)
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = (pa.permute(0, 2, 3, 1)
                             .contiguous().cpu().numpy()
                             .astype(np.float32, copy=False))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    pf_linear = _resize_output_torch(pf_linear, meta)
    if "fg" in outputs:
        rendered["fg"] = _chw_to_bgr_f32_cpu(pf_linear)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_average_srgb_torch(
            pf_linear, strength=despill_strength, screen_color=screen_color)
        rendered["cutout"] = _chw_to_bgra_f32_cpu(pf_linear_despilled, pa)
    if "checker" in outputs:
        checker_b = checker.to(device=pf_linear.device,
                               dtype=pf_linear.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        pf_despill = _despill_average_srgb_torch(
            pf_linear, strength=despill_strength, screen_color=screen_color)
        comp_lin = pf_despill * pa + bg_lin * (1.0 - pa)
        rendered["checker"] = _chw_to_bgr_f32_cpu(comp_lin)
    return rendered


def _render_predictions_torch(meta: dict, outputs: set[str],
                              checker: torch.Tensor,
                              logits: torch.Tensor) -> dict[str, np.ndarray]:
    """Render a stage-2 batch on GPU, returning uint16 BGR/BGRA NHWC arrays.

    This removes the previous per-frame numpy HDR decode, sRGB encode,
    composition, channel shuffle, and uint8 conversion hot path.
    """
    alpha_model = logits[:, 0:1].float()
    pa_model = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
    despill_strength = float(meta.get("despill_strength", 0.5))
    screen_color = str(meta.get("screen_color", "green"))

    pa = _resize_output_torch(pa_model, meta).clamp(0.0, 1.0)
    rendered: dict[str, np.ndarray] = {}
    if "alpha" in outputs:
        rendered["alpha"] = _srgb_to_bgr_uint16_cpu(pa.expand(-1, 3, -1, -1))
    if not (outputs & {"fg", "cutout", "checker"}):
        return rendered

    fg_model = logits[:, 1:4].float()
    pf_model = _crop_native_torch(fg_model, meta)
    pf_linear = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_model), meta)
    pf_linear_pos = pf_linear.clamp(min=0.0)
    pf = _resize_output_torch(cu.linear_to_srgb(pf_linear_pos), meta).clamp(0.0, 1.0)
    if "fg" in outputs:
        rendered["fg"] = _srgb_to_bgr_uint16_cpu(pf)
    if "cutout" in outputs:
        pf_linear_despilled = _despill_average_srgb_torch(
            pf_linear_pos, strength=despill_strength, screen_color=screen_color)
        if meta.get("cutout_linear", False):
            pf_linear_despilled_resized = _resize_output_torch(pf_linear_despilled, meta).clamp(0.0, 1.0)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_linear_despilled_resized, pa)
        else:
            pf_despill = _despill_average_srgb_torch(
                pf, strength=despill_strength, screen_color=screen_color)
            rendered["cutout"] = _srgb_to_bgra_uint16_cpu(pf_despill, pa)
    if "checker" in outputs:
        checker_b = checker.to(device=pf.device, dtype=pf.dtype).unsqueeze(0)
        bg_lin = cu.srgb_to_linear(checker_b)
        alpha = pa
        pf_despill = _despill_average_srgb_torch(
            pf, strength=despill_strength, screen_color=screen_color)
        comp_despill_lin = (
            cu.srgb_to_linear(pf_despill) * alpha + bg_lin * (1.0 - alpha))
        comp_despill = cu.linear_to_srgb(
            comp_despill_lin.clamp(min=0.0)).clamp(0.0, 1.0)
        rendered["checker"] = _srgb_to_bgr_uint16_cpu(comp_despill)
    return rendered


def run_inference(model, input_path: str, output_dir: str, cfg: InferenceConfig,
                  writer_cfg: WriterConfig, *, initial_hint_path: str | None,
                  hint_video_path: str | None,
                  initial_hint_frame_index: int = 0,
                  on_frame_rendered=None,
                  on_progress=None,
                  on_status=None,
                  is_cancelled=None):
    input_path = str(input_path)
    output_dir = str(output_dir)
    cfg.temp_dir.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    is_dir = os.path.isdir(input_path)
    reader_kind = "frame_dir" if is_dir else "video"
    reader_cls = SequentialFrameDirWindowReader if is_dir else SequentialVideoWindowReader
    if is_dir:
        meta = probe_frame_dir_meta(
            input_path, num_frames=cfg.num_frames,
            shape_quantum=cfg.shape_quantum, fps=cfg.frame_dir_fps,
            exr_linear=cfg.frame_dir_linear, model_scale=cfg.model_scale,
            linear_brightness=cfg.linear_brightness,
            linear_contrast=cfg.linear_contrast,
            linear_contrast_pivot=cfg.linear_contrast_pivot)
        if (not cfg.frame_dir_linear
                and meta["frame_paths"]
                and meta["frame_paths"][0].lower().endswith(".exr")):
            print("  [WARNING] EXR input frames are being treated as "
                  "display-coded and clipped to [0, 1]. For scene-linear HDR "
                  "EXR plates, pass --frame_dir_linear.")
    else:
        meta = probe_video_meta(
            input_path, num_frames=cfg.num_frames,
            shape_quantum=cfg.shape_quantum, native_resolution=True,
            start_mode=cfg.start_mode, seed=cfg.seed,
            model_scale=cfg.model_scale,
            linear_brightness=cfg.linear_brightness,
            linear_contrast=cfg.linear_contrast,
            linear_contrast_pivot=cfg.linear_contrast_pivot,
            start_frame=cfg.start_frame)
    T_all = int(meta["num_frames"])
    H, W = meta["shape_hw"]
    if cfg.auto_vram:
        apply_auto_vram(cfg, (H, W), T_all)
    H_out, W_out = meta["output_hw"]
    meta["cutout_linear"] = cfg.cutout_linear
    meta["despill_strength"] = cfg.despill_strength
    meta["screen_color"] = cfg.screen_color
    fps = float(meta["native_fps"])
    clip_name = raw_clip_name(input_path)
    run_id = uuid.uuid4().hex[:8]
    rvm_path = cfg.temp_dir / f"{clip_name}_{run_id}_rvm_feat.f16"
    baton_path = cfg.temp_dir / f"{clip_name}_{run_id}_baton.f16"
    foundation_prefix = cfg.temp_dir / f"{clip_name}_{run_id}_foundation"
    checker = make_checker(H_out, W_out)
    checker_t = torch.from_numpy(checker)
    writers = {}
    suffix = {
        "alpha": "alpha",
        "fg": "fg",
        "checker": f"checker_comp_despill{int(cfg.despill_strength * 100):03d}",
        "cutout": "cutout",
    }
    try:
        use_exr = cfg.output_format == "exr"
        if cfg.output_format not in ("video", "exr"):
            raise ValueError(f"unsupported output format: {cfg.output_format}")
        for name in sorted(cfg.outputs):
            if name not in suffix:
                raise ValueError(f"unsupported output choice: {name}")
            if use_exr:
                channels = 4 if name == "cutout" else (1 if name == "alpha" else 3)
                writers[name] = ExrSequenceWriter(
                    os.path.join(output_dir, f"{clip_name}_{suffix[name]}_exr"),
                    fps, (W_out, H_out), channels=channels,
                    dtype=cfg.exr_dtype, frame_name=suffix[name])
            else:
                ext = ".mov" if (name == "cutout" or writer_cfg.codec == "prores_ks") else ".mp4"
                channels = 4 if name == "cutout" else 3
                color_trc = "linear" if (name == "cutout" and cfg.cutout_linear) else "iec61966-2-1"
                writers[name] = ThreadedVideoWriter(
                    os.path.join(output_dir, f"{clip_name}_{suffix[name]}{ext}"),
                    fps, (W_out, H_out), writer_cfg, channels=channels, color_trc=color_trc)

        # Write settings log file in the output folder
        settings_path = os.path.join(output_dir, f"{clip_name}_settings.txt")
        try:
            with open(settings_path, "w") as sf:
                sf.write("CorridorKey v2 Inference Settings Log\n")
                sf.write("====================================\n")
                sf.write(f"Input: {input_path}\n")
                sf.write(f"Output Directory: {output_dir}\n")
                sf.write(f"Resolution: {W_out}x{H_out}\n")
                sf.write(f"FPS: {fps}\n")
                sf.write(f"Outputs: {', '.join(sorted(cfg.outputs))}\n")
                sf.write(f"Output Format: {cfg.output_format}\n")
                if cfg.output_format == "exr":
                    sf.write(f"EXR Pixel Type: {cfg.exr_dtype}\n")
                sf.write(f"Hint First Frame: {initial_hint_path}\n")
                sf.write(f"Hint Frame Index: {initial_hint_frame_index}\n")
                sf.write(f"Hint Video: {hint_video_path}\n")
                sf.write(f"Hint Quality (Trust): {cfg.hint_quality}\n")
                sf.write(f"Carry Hint: {cfg.carry_hint}\n")
                sf.write(f"Carry Hint Quality: {cfg.carry_hint_quality}\n")
                sf.write(f"Despill Strength: {cfg.despill_strength}\n")
                sf.write(f"Cutout Linear: {cfg.cutout_linear}\n")
                sf.write(f"Frame Dir Linear: {cfg.frame_dir_linear}\n")
                sf.write(f"Hann Chunk: {cfg.hann_chunk}\n")
                sf.write(f"Hann Stride: {cfg.hann_stride}\n")
                sf.write(f"Low VRAM: {cfg.low_vram}\n")
                sf.write(f"Model Scale: {cfg.model_scale}\n")
                sf.write(f"Num Frames: {cfg.num_frames}\n")
                sf.write(f"Start Mode: {cfg.start_mode}\n")
                sf.write(f"Seed: {cfg.seed}\n")
                sf.write(f"Linear Brightness: {cfg.linear_brightness}\n")
                sf.write(f"Linear Contrast: {cfg.linear_contrast}\n")
                sf.write(f"Linear Contrast Pivot: {cfg.linear_contrast_pivot}\n")
        except Exception as e:
            print(f"[WARNING] Failed to write settings file: {e}")

        initial_hint = (
            load_initial_hint_frame(initial_hint_path, meta)
            if none_path(initial_hint_path) is not None else None
        )
        _phase1_setup(model, stage1_on_gpu=not cfg.low_vram)
        configure_low_vram_inference(model, cfg)
        for p in (
            rvm_path, baton_path, Path(str(foundation_prefix) + "_cradio.f32"),
            Path(str(foundation_prefix) + "_moge.f32"),
        ):
            safe_remove(p)
        if model.foundations.rvm is not None:
            precompute_rvm_to_memmap(
                model, input_path, meta, rvm_path,
                chunk=max(1, min(int(cfg.hann_chunk), 8)),
                reader_kind=reader_kind,
                is_cancelled=is_cancelled)
            rvm_feat_path = rvm_path
        else:
            rvm_feat_path = None
        foundation_paths = None
        if cfg.low_vram:
            foundation_paths = precompute_foundations_to_memmap(
                model, input_path, meta, foundation_prefix,
                chunk=max(1, int(cfg.low_vram_foundation_chunk)),
                reader_kind=reader_kind,
                is_cancelled=is_cancelled)
            model.foundations.cpu()
            model.stage2.cpu()
            model.stage1.to(_target_device_str())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            model.stage2.to(_target_device_str())
            model.stage2.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stage2_batch = max(1, int(cfg.stage2_batch))

        def write_prediction_batches(items):
            device = next(model.stage2.parameters()).device
            for offset in range(0, len(items), stage2_batch):
                if is_cancelled is not None and is_cancelled():
                    _raise_cancelled(str(input_path), offset)
                sub = items[offset:offset + stage2_batch]
                rgb_t = torch.stack([item[2] for item in sub], dim=0)
                rgb_t = rgb_t.unsqueeze(0).to(device, non_blocking=True)
                baton_np = np.stack([item[1] for item in sub], axis=0)
                baton_t = torch.from_numpy(baton_np).unsqueeze(0)
                baton_t = baton_t.to(device, non_blocking=True).float()
                with torch.inference_mode(), _autocast_ctx(device):
                    logits = model.stage2(rgb_t, baton_t)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                render_fn = (_render_predictions_exr_torch if use_exr
                             else _render_predictions_torch)
                rendered = render_fn(meta, cfg.outputs, checker_t, logits[0])
                for name, frames in rendered.items():
                    writers[name].write_many(frames)
                if on_frame_rendered is not None:
                    for sub_idx, (frame_k, _, _) in enumerate(sub):
                        f_dict = {name: frames[sub_idx] for name, frames in rendered.items()}
                        on_frame_rendered(frame_k, f_dict)
                frames_done_counter[0] += len(sub)
                if on_progress is not None:
                    on_progress(frames_done_counter[0], T_all)
                del logits, baton_t, rgb_t, baton_np, rendered

        frames_done_counter = [0]
        def _on_s1(cur, tot):
            if on_status is not None:
                on_status(f"Stage 1 analysis: window {cur}/{tot}...")

        hann_stage1_stream(
            model, input_path, meta, reader_cls=reader_cls,
            reader_kind=reader_kind, chunk=cfg.hann_chunk, stride=cfg.hann_stride,
            baton_path=baton_path, initial_hint=initial_hint,
            initial_hint_quality=cfg.hint_quality,
            full_hint_video_path=hint_video_path,
            full_hint_quality=cfg.hint_quality,
            rvm_path=rvm_feat_path, foundation_paths=foundation_paths,
            carry_hint_from_aux=cfg.carry_hint,
            carry_hint_quality=cfg.carry_hint_quality,
            on_baton_frames=None if cfg.low_vram else write_prediction_batches,
            on_stage1_window=_on_s1,
            is_cancelled=is_cancelled,
            initial_hint_frame_index=initial_hint_frame_index)

        if cfg.low_vram:
            _phase2_setup(model)
            baton = np.memmap(
                baton_path, dtype=np.float16, mode="r",
                shape=(T_all, model.stage1.d_16, H // 16, W // 16))
            reader = reader_cls(input_path, meta)
            try:
                for t0 in range(0, T_all, stage2_batch):
                    t1 = min(T_all, t0 + stage2_batch)
                    model_chunk, _display_chunk, _review_chunk = reader.window(t0, t1)
                    items = [
                        (t, np.asarray(baton[t]).copy(), model_chunk[t - t0])
                        for t in range(t0, t1)
                    ]
                    write_prediction_batches(items)
                    del model_chunk, _display_chunk, _review_chunk, items
            finally:
                reader.close()
                del baton
    finally:
        for writer in writers.values():
            writer.release()
        safe_remove(rvm_path)
        safe_remove(baton_path)
        safe_remove(Path(str(foundation_prefix) + "_cradio.f32"))
        safe_remove(Path(str(foundation_prefix) + "_moge.f32"))
        try:
            if hasattr(model, "foundations"):
                model.foundations.cpu()
            if hasattr(model, "stage1"):
                model.stage1.cpu()
            if hasattr(model, "stage2"):
                model.stage2.cpu()
            model.cpu()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    print(f"  wrote {len(writers)} output video(s) for {clip_name} "
          f"({T_all} frames @ {W_out}x{H_out})")


@torch.no_grad()
def infer_single_frame(
    model,
    frame_rgb: np.ndarray,
    *,
    is_linear: bool = False,
    hint_mask: np.ndarray | None = None,
    hint_quality: float = 0.95,
    model_scale: float = 1.0,
    linear_brightness: float = 1.0,
    linear_contrast: float = 1.0,
    linear_contrast_pivot: float = 0.18,
    low_vram: bool = False,
    shape_quantum: int = 128,
    auto_roi_crop: bool = False,
    auto_roi_pad_px: int = 96,
) -> dict[str, np.ndarray]:
    """Run in-memory single-frame CorridorKey 2.0 inference without writing to disk.

    Args:
        model: Loaded `CorridorKeyV3` instance (on CPU or GPU).
        frame_rgb: Input RGB frame `[H, W, 3]` (`uint8` or `float32`).
        is_linear: Whether `frame_rgb` is already scene-linear HDR float.
        hint_mask: Optional `[H, W]` soft/binary mask guide (`uint8` or `[0, 1]` float).
        hint_quality: Confidence weight applied to `hint_mask`.
        model_scale: Internal processing scale multiplier.
        linear_brightness: Linear exposure gain before model encoding.
        linear_contrast: Linear contrast multiplier around `linear_contrast_pivot`.
        linear_contrast_pivot: Contrast pivot in linear light.
        low_vram: Enable attention/MLP chunking during forward pass.
        shape_quantum: Pad model grid multiples (128 recommended, must be % 64 == 0).

    Returns:
        Dict containing float32 CPU arrays at native `(H, W)` resolution:
        `alpha_raw` `(H, W)`, `fg_raw` `(H, W, 3)` sRGB, and `fg_linear` `(H, W, 3)`.
    """
    import cv2

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] < 3:
        raise ValueError(f"Expected [H, W, 3+] RGB frame, got shape {frame_rgb.shape}")

    rgb_in = frame_rgb[..., :3]
    if rgb_in.dtype == np.uint8:
        rgb_f = rgb_in.astype(np.float32) * (1.0 / 255.0)
    elif rgb_in.dtype == np.uint16:
        rgb_f = rgb_in.astype(np.float32) * (1.0 / 65535.0)
    else:
        rgb_f = rgb_in.astype(np.float32, copy=False)

    H_orig, W_orig = rgb_f.shape[:2]
    roi_box: tuple[int, int, int, int] | None = None
    hm_full_f32: np.ndarray | None = None
    if hint_mask is not None:
        hm_tmp = hint_mask
        if hm_tmp.ndim == 3:
            hm_tmp = hm_tmp[:, :, 3] if hm_tmp.shape[2] == 4 else hm_tmp[:, :, 0]
        if hm_tmp.dtype == np.uint8:
            hm_full_f32 = hm_tmp.astype(np.float32) * (1.0 / 255.0)
        elif hm_tmp.dtype == np.uint16:
            hm_full_f32 = hm_tmp.astype(np.float32) * (1.0 / 65535.0)
        else:
            hm_full_f32 = hm_tmp.astype(np.float32, copy=False)
            if float(hm_full_f32.max()) > 1.5:
                hm_full_f32 = hm_full_f32 * (1.0 / 255.0)
        hm_full_f32 = np.clip(hm_full_f32, 0.0, 1.0)
        if auto_roi_crop and H_orig > 64 and W_orig > 64:
            hm_for_bbox = hm_full_f32
            if hm_for_bbox.shape[:2] != (H_orig, W_orig):
                hm_for_bbox = cv2.resize(hm_for_bbox, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(hm_for_bbox > 0.04)
            if ys.size > 0:
                pad_px = max(16, int(auto_roi_pad_px))
                x0 = max(0, int(xs.min()) - pad_px)
                y0 = max(0, int(ys.min()) - pad_px)
                x1 = min(W_orig, int(xs.max()) + 1 + pad_px)
                y1 = min(H_orig, int(ys.max()) + 1 + pad_px)
                if (x1 - x0) < 64:
                    cx = (x0 + x1) // 2
                    x0 = max(0, min(W_orig - 64, cx - 32))
                    x1 = min(W_orig, x0 + 64)
                if (y1 - y0) < 64:
                    cy = (y0 + y1) // 2
                    y0 = max(0, min(H_orig - 64, cy - 32))
                    y1 = min(H_orig, y0 + 64)
                if (x1 - x0) * (y1 - y0) < int(0.85 * H_orig * W_orig):
                    roi_box = (x0, y0, x1, y1)
                    rgb_f_full = rgb_f
                    rgb_f = rgb_f[y0:y1, x0:x1]
                    hint_mask = hm_for_bbox[y0:y1, x0:x1]

    H_out, W_out = rgb_f.shape[:2]
    h_model, w_model = scaled_hw(H_out, W_out, float(model_scale))
    q = max(64, int(shape_quantum))
    if q % 64 != 0:
        q = 128
    h_pad = ceil_to_multiple(h_model, q)
    w_pad = ceil_to_multiple(w_model, q)
    pad_l = (w_pad - w_model) // 2
    pad_r = w_pad - w_model - pad_l
    pad_t = (h_pad - h_model) // 2
    pad_b = h_pad - h_model - pad_t

    meta = {
        "output_hw": (H_out, W_out),
        "model_native_hw": (h_model, w_model),
        "pad": (pad_l, pad_r, pad_t, pad_b),
        "linear_brightness": float(linear_brightness),
        "linear_contrast": float(linear_contrast),
        "linear_contrast_pivot": float(linear_contrast_pivot),
    }

    if is_linear:
        linear_native = np.maximum(rgb_f, 0.0)
        display_native = np.clip(cu.linear_to_srgb(linear_native), 0.0, 1.0)
    else:
        display_native = np.clip(rgb_f, 0.0, 1.0)
        linear_native = cu.srgb_to_linear(display_native)

    display_scaled = resize_hwc_lanczos(display_native, (h_model, w_model))
    linear_scaled = resize_hwc_lanczos(linear_native, (h_model, w_model))

    if has_linear_adjust(meta):
        linear_adj = apply_linear_adjust(linear_scaled, meta)
        foundation_np = np.clip(cu.linear_to_vfm_sdr(linear_adj), 0.0, 1.0)
    else:
        linear_adj = linear_scaled
        foundation_np = display_scaled

    linear_adj_pad, _ = center_pad_to_shape(linear_adj, h_pad, w_pad)
    foundation_pad, _ = center_pad_to_shape(foundation_np, h_pad, w_pad)

    rgb_asinh_np = cu.linear_to_asinh(linear_adj_pad).astype(np.float32, copy=False)

    dev_str = _target_device_str()
    device = torch.device(dev_str)
    dt = _amp_dtype() if device.type == "cuda" else torch.float32

    rgb_t = (
        torch.from_numpy(np.ascontiguousarray(rgb_asinh_np.transpose(2, 0, 1)))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=dt)
    )
    rgb_vfm = (
        torch.from_numpy(np.ascontiguousarray(foundation_pad.transpose(2, 0, 1)))
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device=device, dtype=dt)
    )
    rgb_radio, rgb_moge = _build_foundation_inputs(rgb_vfm)

    if hint_mask is not None:
        hm = hint_mask
        if hm.ndim == 3:
            hm = hm[:, :, 3] if hm.shape[2] == 4 else hm[:, :, 0]
        if hm.dtype == np.uint8:
            hm_f = hm.astype(np.float32) * (1.0 / 255.0)
        elif hm.dtype == np.uint16:
            hm_f = hm.astype(np.float32) * (1.0 / 65535.0)
        else:
            hm_f = hm.astype(np.float32, copy=False)
            if float(hm_f.max()) > 1.5:
                hm_f = hm_f * (1.0 / 255.0)
        hm_f = np.clip(hm_f, 0.0, 1.0)
        if hm_f.shape[:2] != (h_model, w_model):
            hm_f = cv2.resize(hm_f, (w_model, h_model), interpolation=cv2.INTER_LINEAR)
        hm_pad, _ = center_pad_to_shape(hm_f, h_pad, w_pad)
        hint_t = (
            torch.from_numpy(np.ascontiguousarray(hm_pad))
            .view(1, 1, 1, h_pad, w_pad)
            .to(device=device, dtype=dt)
        )
        hq_t = torch.full((1, 1), float(hint_quality), device=device, dtype=dt)
    else:
        hint_t = torch.zeros((1, 1, 1, h_pad, w_pad), device=device, dtype=dt)
        hq_t = torch.zeros((1, 1), device=device, dtype=dt)

    cfg = InferenceConfig(low_vram=bool(low_vram))
    configure_low_vram_inference(model, cfg)
    model.eval()

    try:
        model.to(device)
        with torch.inference_mode(), _autocast_ctx(device):
            out = model(
                rgb_t,
                hint_t,
                rgb_cradio=rgb_radio,
                rgb_moge=rgb_moge,
                hint_quality=hq_t,
            )

        alpha_model = out["alpha"][:, 0].float()
        pa_native = _crop_native_torch(alpha_model, meta).clamp(0.0, 1.0)
        pa = _resize_output_torch(pa_native, meta).clamp(0.0, 1.0)
        alpha_raw = (
            pa[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        )

        fg_model = out["fg"][:, 0].float()
        pf_native = _crop_native_torch(fg_model, meta)
        pf_lin = _linear_adjust_inverse_torch(cu.asinh_to_linear(pf_native), meta)
        pf_lin_out = _resize_output_torch(pf_lin, meta)
        fg_linear = (
            pf_lin_out[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float32, copy=False)
        )
        fg_raw = np.clip(cu.linear_to_srgb(np.maximum(fg_linear, 0.0)), 0.0, 1.0).astype(
            np.float32, copy=False
        )
        if roi_box is not None:
            x0, y0, x1, y1 = roi_box
            alpha_full = np.zeros((H_orig, W_orig), dtype=np.float32)
            alpha_full[y0:y1, x0:x1] = alpha_raw
            if is_linear:
                fg_lin_full = np.maximum(rgb_f_full, 0.0).astype(np.float32, copy=True)
                fg_raw_full = np.clip(cu.linear_to_srgb(fg_lin_full), 0.0, 1.0).astype(np.float32, copy=False)
            else:
                fg_raw_full = np.clip(rgb_f_full, 0.0, 1.0).astype(np.float32, copy=True)
                fg_lin_full = cu.srgb_to_linear(fg_raw_full).astype(np.float32, copy=False)
            fg_raw_full[y0:y1, x0:x1] = fg_raw
            fg_lin_full[y0:y1, x0:x1] = fg_linear
            alpha_raw = alpha_full
            fg_raw = fg_raw_full
            fg_linear = fg_lin_full
        return {
            "alpha_raw": alpha_raw,
            "fg_raw": fg_raw,
            "fg_linear": fg_linear,
        }
    finally:
        del rgb_t, rgb_vfm, rgb_radio, rgb_moge, hint_t, hq_t
        try:
            if hasattr(model, "foundations"):
                model.foundations.cpu()
            if hasattr(model, "stage1"):
                model.stage1.cpu()
            if hasattr(model, "stage2"):
                model.stage2.cpu()
            model.cpu()
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
