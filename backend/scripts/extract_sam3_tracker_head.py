#!/usr/bin/env python3
"""Extract `sam3.1-tracker-head.safetensors` (~67 MB) from SAM 3.1 Multiplex weights.

Supports either a local `sam3.1_multiplex*.safetensors` file or direct HTTP Range
streaming from Hugging Face (`Comfy-Org/sam3.1`, ~67 MB downloaded instead of 3.4 GB).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import time
import urllib.request
from pathlib import Path
from typing import Callable

DEFAULT_MULTIPLEX_URL = (
    "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/"
    "checkpoints/sam3.1_multiplex_fp16.safetensors"
)
EXPECTED_TENSOR_COUNT = 493


def is_tracker_head_key(key: str) -> bool:
    """True for keys belonging to the SAM 3.1 multiplex tracker & interactive prompt head."""
    return (
        key.startswith("tracker.")
        or "propagation_convs" in key
        or "interactive_convs" in key
    )


def _fetch_range(url: str, start: int, end_inclusive: int, *, retries: int = 3, timeout: float = 30.0) -> bytes:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={start}-{end_inclusive}",
                    "User-Agent": "CorridorKey/2.0",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()
            expected_len = end_inclusive - start + 1
            if len(data) != expected_len:
                raise IOError(f"Short range read: got {len(data)} bytes, expected {expected_len}")
            return data
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(0.8 * (attempt + 1))
    raise RuntimeError(f"Failed HTTP range {start}-{end_inclusive} from {url}: {last_exc}") from last_exc


def _coalesce_spans(
    spans: list[tuple[int, int]],
    *,
    max_gap: int = 65536,
    max_chunk: int = 4 * 1024 * 1024,
) -> list[tuple[int, int]]:
    """Merge nearby byte ranges `[start, end_excl)` into bounded HTTP batch requests."""
    if not spans:
        return []
    ordered = sorted(spans)
    batches: list[tuple[int, int]] = []
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s - cur_e <= max_gap and (e - cur_s) <= max_chunk:
            cur_e = max(cur_e, e)
        else:
            batches.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    batches.append((cur_s, cur_e))
    return batches


def extract_sam3_tracker_head(
    out_path: str | Path,
    *,
    url: str = DEFAULT_MULTIPLEX_URL,
    local_multiplex_path: str | Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Path:
    """Extract the 493 tracker-head tensors into a standalone `.safetensors` file."""
    dst = Path(out_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_local = Path(local_multiplex_path) if local_multiplex_path else None
    if src_local is not None:
        with open(src_local, "rb") as rf:
            hdr_len = struct.unpack("<Q", rf.read(8))[0]
            rem_hdr = json.loads(rf.read(hdr_len).decode("utf-8"))
        data_base = 8 + hdr_len
    else:
        hdr_len = struct.unpack("<Q", _fetch_range(url, 0, 7))[0]
        rem_hdr = json.loads(_fetch_range(url, 8, 8 + hdr_len - 1).decode("utf-8"))
        data_base = 8 + hdr_len

    selected = sorted(k for k in rem_hdr if k != "__metadata__" and is_tracker_head_key(k))
    if not selected:
        raise RuntimeError("No tracker-head tensors found in source safetensors header")

    # Build output header with packed 0-based relative offsets
    out_hdr: dict[str, dict] = {}
    tensor_specs: list[tuple[str, int, int]] = []  # (key, abs_start, byte_len)
    cursor = 0
    for k in selected:
        meta = rem_hdr[k]
        s0, s1 = int(meta["data_offsets"][0]), int(meta["data_offsets"][1])
        blen = s1 - s0
        out_hdr[k] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [cursor, cursor + blen],
        }
        tensor_specs.append((k, data_base + s0, blen))
        cursor += blen

    hdr_bytes = json.dumps(out_hdr, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    pad = (8 - (len(hdr_bytes) % 8)) % 8
    hdr_padded = hdr_bytes + (b" " * pad)

    total_bytes = cursor
    done_bytes = 0
    tmp_path = dst.with_suffix(".safetensors.part")

    try:
        with open(tmp_path, "wb") as wf:
            wf.write(struct.pack("<Q", len(hdr_padded)))
            wf.write(hdr_padded)

            if src_local is not None:
                with open(src_local, "rb") as rf:
                    for _, abs_s, blen in tensor_specs:
                        rf.seek(abs_s)
                        chunk = rf.read(blen)
                        if len(chunk) != blen:
                            raise IOError(f"Unexpected EOF reading {blen} bytes from {src_local}")
                        wf.write(chunk)
                        done_bytes += blen
                        if progress_cb is not None:
                            progress_cb(done_bytes, total_bytes)
            else:
                # Group tensor byte spans into coalesced HTTP Range batches
                spans = [(abs_s, abs_s + blen) for _, abs_s, blen in tensor_specs]
                batches = _coalesce_spans(spans)
                # Keep specs in exact write order and slice from current batch buffer
                spec_idx = 0
                for b_start, b_end in batches:
                    buf = _fetch_range(url, b_start, b_end - 1)
                    while spec_idx < len(tensor_specs):
                        _, abs_s, blen = tensor_specs[spec_idx]
                        if abs_s < b_start or (abs_s + blen) > b_end:
                            break
                        off = abs_s - b_start
                        wf.write(buf[off : off + blen])
                        done_bytes += blen
                        spec_idx += 1
                        if progress_cb is not None:
                            progress_cb(done_bytes, total_bytes)

        # Verify written safetensors header integrity before atomic rename
        with open(tmp_path, "rb") as vf:
            vh_len = struct.unpack("<Q", vf.read(8))[0]
            vh = json.loads(vf.read(vh_len).decode("utf-8"))
            if len(vh) != len(selected):
                raise RuntimeError(f"Extracted tensor count mismatch: {len(vh)} != {len(selected)}")
            vf.seek(0, os.SEEK_END)
            actual_size = vf.tell()
            expected_size = 8 + vh_len + cursor
            if actual_size != expected_size:
                raise RuntimeError(f"Extracted file size mismatch: {actual_size} != {expected_size}")

        os.replace(tmp_path, dst)
        return dst
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def main() -> int:
    default_out = Path(__file__).resolve().parents[1] / "weights" / "sam3" / "sam3.1-tracker-head.safetensors"
    ap = argparse.ArgumentParser(description="Extract sam3.1-tracker-head.safetensors (~67 MB)")
    ap.add_argument("--output", "-o", default=str(default_out), help="Destination path")
    ap.add_argument("--source", "-s", default=None, help="Optional local sam3.1_multiplex*.safetensors path")
    ap.add_argument("--url", default=DEFAULT_MULTIPLEX_URL, help="Remote safetensors URL")
    args = ap.parse_args()

    def _on_prog(done: int, total: int) -> None:
        pct = (done * 100) // max(1, total)
        print(f"\r[sam3-tracker-head] {done / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB ({pct}%)", end="", flush=True)

    res = extract_sam3_tracker_head(
        args.output,
        url=args.url,
        local_multiplex_path=args.source,
        progress_cb=_on_prog,
    )
    print(f"\n[sam3-tracker-head] Saved -> {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
