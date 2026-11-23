"""Headless Batch / Watch-Folder CLI runner for CorridorKey 2.0 (no Qt dependency)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from backend.project import is_video_file


def discover_inputs(input_dir: str) -> list[str]:
    """Return sorted list of video files or image sequence folders inside `input_dir`."""
    if not input_dir or not os.path.isdir(input_dir):
        return []
    items: list[str] = []
    for name in sorted(os.listdir(input_dir)):
        full = os.path.join(input_dir, name)
        if os.path.isfile(full) and is_video_file(full):
            items.append(full)
        elif os.path.isdir(full):
            # Treat folders containing frames cleanly as sequence clips
            exts = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff")
            if any(f.lower().endswith(exts) for f in os.listdir(full)):
                items.append(full)
    return items


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CorridorKey 2.0 Headless Batch / Watch-Folder Runner")
    p.add_argument("--input", "-i", required=True, help="Folder of input videos or frame sequences")
    p.add_argument("--output", "-o", required=True, help="Output folder where processed clips are saved")
    p.add_argument("--preset", default="Default Balanced", help="Preset name from PresetManager")
    p.add_argument("--codec", default="h264", choices=("h264", "prores_4444", "prores_422hq", "png_seq", "exr"))
    p.add_argument("--watch", action="store_true", help="Continuously poll `--input` for newly dropped media")
    p.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval in watch mode")
    p.add_argument("--dry-run", action="store_true", help="Plan and print discovered jobs without running GPU inference")
    return p


def run_cli_batch(
    input_dir: str,
    output_dir: str,
    *,
    preset: str = "Default Balanced",
    codec: str = "h264",
    dry_run: bool = False,
    processed_set: Optional[set[str]] = None,
) -> list[str]:
    """Scan `input_dir` once and execute (or dry-run plan) keying for each new item."""
    from backend.presets import PresetManager
    from backend.service.core import InferenceParams, OutputConfig

    pm = PresetManager()
    pdata = pm.get_preset(preset) or {}
    merged = {**pdata, "export_codec": codec, "output_format": ("exr" if codec == "exr" else "video")}
    params = InferenceParams.from_dict(merged)
    out_cfg = OutputConfig(
        output_format=("exr" if codec == "exr" else "video"),
        export_codec=codec,
    )

    inputs = discover_inputs(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    ran: list[str] = []

    if not dry_run:
        from backend.clip_state import ClipEntry, ClipState
        from backend.project import create_clip_from_source
        from backend.service.core import CorridorKeyService
        svc = CorridorKeyService()
    else:
        svc = None  # type: ignore[assignment]

    for src_path in inputs:
        if processed_set is not None and src_path in processed_set:
            continue
        clip_name = os.path.splitext(os.path.basename(src_path))[0]
        if dry_run:
            print(f"[DRY-RUN] Would key '{clip_name}' ({src_path}) -> {output_dir} [codec={codec}, preset={preset}]")
            ran.append(src_path)
            if processed_set is not None:
                processed_set.add(src_path)
            continue

        clip_root = os.path.join(output_dir, clip_name)
        clip = create_clip_from_source(src_path, clip_root)
        if clip.state == ClipState.RAW and clip.alpha_asset is None and clip.mask_asset is None:
            svc.run_birefnet(clip)
        if clip.state in (ClipState.READY, ClipState.COMPLETE):
            svc.run_inference(clip, params, output_config=out_cfg)
        ran.append(src_path)
        if processed_set is not None:
            processed_set.add(src_path)

    return ran


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    seen: set[str] = set()
    if not args.watch:
        run_cli_batch(
            args.input,
            args.output,
            preset=args.preset,
            codec=args.codec,
            dry_run=bool(args.dry_run),
            processed_set=seen,
        )
        return 0

    print(f"Watching '{args.input}' every {args.poll_seconds:.1f}s (Ctrl+C to stop)...")
    try:
        while True:
            run_cli_batch(
                args.input,
                args.output,
                preset=args.preset,
                codec=args.codec,
                dry_run=bool(args.dry_run),
                processed_set=seen,
            )
            time.sleep(max(0.5, float(args.poll_seconds)))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
