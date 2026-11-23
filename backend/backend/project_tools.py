"""Disk Cache Management & Portable Project Archival (Collect & Pack)."""

from __future__ import annotations

import os
import shutil
import zipfile
from typing import Iterable


def _dir_size_bytes(path: str) -> int:
    total = 0
    if not os.path.isdir(path):
        return 0
    for dp, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(dp, fn))
            except OSError:
                pass
    return total


def scan_project_disk_usage(root_dir: str) -> dict[str, int]:
    """Scan a project or clip folder and break down bytes per cache/output category."""
    usage = {
        "frames": 0,
        "masks": 0,
        "alpha_hints": 0,
        "outputs": 0,
        "exports": 0,
        "source": 0,
    }
    if not root_dir or not os.path.isdir(root_dir):
        return usage

    # Support either a clip root directly or a project folder containing clips/
    clip_dirs: list[str] = []
    if os.path.isdir(os.path.join(root_dir, "clips")):
        croot = os.path.join(root_dir, "clips")
        clip_dirs = [os.path.join(croot, d) for d in os.listdir(croot) if os.path.isdir(os.path.join(croot, d))]
    else:
        clip_dirs = [root_dir]
        # Also include child clip dirs if root_dir contains clip subfolders directly
        for d in os.listdir(root_dir):
            cand = os.path.join(root_dir, d)
            if os.path.isdir(cand) and any(
                os.path.isdir(os.path.join(cand, sub))
                for sub in ("Frames", "Source", "SamMaskHint", "AlphaHint", "FG")
            ):
                clip_dirs.append(cand)

    for cdir in clip_dirs:
        usage["frames"] += _dir_size_bytes(os.path.join(cdir, "Frames"))
        usage["masks"] += _dir_size_bytes(os.path.join(cdir, "SamMaskHint"))
        usage["alpha_hints"] += _dir_size_bytes(os.path.join(cdir, "AlphaHint"))
        usage["exports"] += _dir_size_bytes(os.path.join(cdir, "_EXPORTS"))
        usage["source"] += _dir_size_bytes(os.path.join(cdir, "Source"))
        for out_sub in ("FG", "Matte", "Comp", "Processed", "output"):
            usage["outputs"] += _dir_size_bytes(os.path.join(cdir, out_sub))

    return usage


def clean_project_cache(root_dir: str, categories: Iterable[str]) -> int:
    """Delete temporary cache folders for selected categories. Never deletes `Source/`."""
    cats = {c.lower().strip() for c in categories}
    folder_map = {
        "frames": ("Frames",),
        "masks": ("SamMaskHint",),
        "alpha_hints": ("AlphaHint",),
        "outputs": ("FG", "Matte", "Comp", "Processed", "output"),
        "exports": ("_EXPORTS",),
    }
    clip_dirs: list[str] = []
    if os.path.isdir(os.path.join(root_dir, "clips")):
        croot = os.path.join(root_dir, "clips")
        clip_dirs = [os.path.join(croot, d) for d in os.listdir(croot) if os.path.isdir(os.path.join(croot, d))]
    else:
        clip_dirs = [root_dir]
        for d in os.listdir(root_dir):
            cand = os.path.join(root_dir, d)
            if os.path.isdir(cand) and any(
                os.path.isdir(os.path.join(cand, sub))
                for sub in ("Frames", "SamMaskHint", "AlphaHint", "FG")
            ):
                clip_dirs.append(cand)

    freed = 0
    for cdir in clip_dirs:
        for cat in cats:
            for sub in folder_map.get(cat, ()):
                target = os.path.join(cdir, sub)
                if os.path.isdir(target):
                    freed += _dir_size_bytes(target)
                    shutil.rmtree(target, ignore_errors=True)
    return freed


def pack_project_zip(
    root_dir: str,
    out_zip_path: str,
    *,
    include_frames: bool = False,
) -> str:
    """Archive project metadata, annotations, polygons, masks, and optionally extracted Frames into `.zip`."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Project folder not found: {root_dir}")
    skip_dirs = set() if include_frames else {"Frames"}
    parent = os.path.dirname(out_zip_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with zipfile.ZipFile(out_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dp, dn, files in os.walk(root_dir):
            dn[:] = [d for d in dn if d not in skip_dirs and d != "__pycache__"]
            for fn in files:
                abs_f = os.path.join(dp, fn)
                if os.path.abspath(abs_f) == os.path.abspath(out_zip_path):
                    continue
                rel = os.path.relpath(abs_f, root_dir)
                zf.write(abs_f, rel)
    return out_zip_path
