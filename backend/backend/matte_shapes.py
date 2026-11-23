"""Vector / Polygon Garbage & Core Matte shapes with keyframe interpolation."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np


_SHAPES_FILENAME = "matte_shapes.json"


@dataclass
class PolygonMatteShape:
    """Keyframed closed polygon in normalized [0..1] image coordinates.

    kind:
        - "garbage": subtracts inside polygon (alpha *= (1 - mask)) by default,
          or keeps only inside when inverted=True.
        - "core": forces solid 1.0 alpha inside polygon by default,
          or outside when inverted=True.
    """

    shape_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: str = "garbage"  # "garbage" | "core"
    name: str = ""
    inverted: bool = False
    feather_px: int = 4
    enabled: bool = True
    curved: bool = False
    # Keyframe frame_index -> list of normalized (x, y) pairs in [0.0, 1.0]
    keyframes: dict[int, list[tuple[float, float]]] = field(default_factory=dict)

    def set_keyframe(self, frame_index: int, points: Iterable[tuple[float, float]]) -> None:
        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) >= 3:
            self.keyframes[int(frame_index)] = pts

    def has_keyframe(self, frame_index: int) -> bool:
        return int(frame_index) in self.keyframes

    def raw_points_at(self, frame_index: int) -> list[tuple[float, float]]:
        """Return linearly interpolated control vertices for `frame_index`."""
        if not self.keyframes:
            return []
        f = int(frame_index)
        if f in self.keyframes:
            return list(self.keyframes[f])
        keys = sorted(self.keyframes.keys())
        if f <= keys[0]:
            return list(self.keyframes[keys[0]])
        if f >= keys[-1]:
            return list(self.keyframes[keys[-1]])
        k0 = keys[0]
        k1 = keys[-1]
        for k in keys:
            if k < f:
                k0 = k
            elif k > f:
                k1 = k
                break
        pts0 = self.keyframes[k0]
        pts1 = self.keyframes[k1]
        if len(pts0) != len(pts1) or k1 == k0:
            return list(pts0)
        t = float(f - k0) / float(k1 - k0)
        return [
            (x0 * (1.0 - t) + x1 * t, y0 * (1.0 - t) + y1 * t)
            for (x0, y0), (x1, y1) in zip(pts0, pts1)
        ]

    def points_at(self, frame_index: int) -> list[tuple[float, float]]:
        """Return polygon outline points (subdivided Catmull-Rom curve if `curved=True`)."""
        raw = self.raw_points_at(frame_index)
        if self.curved and len(raw) >= 3:
            return _subdivide_catmull_rom(raw, steps=8)
        return raw

    def move_vertex(self, frame_index: int, vertex_idx: int, nx: float, ny: float) -> bool:
        """Move control vertex `vertex_idx` at `frame_index` (creates keyframe if interpolated)."""
        pts = self.raw_points_at(frame_index)
        if not (0 <= vertex_idx < len(pts)):
            return False
        pts[vertex_idx] = (float(nx), float(ny))
        self.keyframes[int(frame_index)] = pts
        return True

    def rasterize(self, height: int, width: int, frame_index: int) -> np.ndarray | None:
        """Rasterize shape to float32 [H, W] mask in [0, 1] at `frame_index`."""
        if not self.enabled or height <= 0 or width <= 0:
            return None
        pts = self.points_at(frame_index)
        if len(pts) < 3:
            return None
        poly = np.array(
            [
                [
                    int(round(np.clip(x, -0.25, 1.25) * (width - 1))),
                    int(round(np.clip(y, -0.25, 1.25) * (height - 1))),
                ]
                for x, y in pts
            ],
            dtype=np.int32,
        )
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(mask, [poly], 1.0)
        feather = int(max(0, self.feather_px))
        if feather > 0:
            ksize = max(3, (feather * 2 + 1) | 1)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0.0)
        if self.inverted:
            mask = 1.0 - mask
        return np.clip(mask, 0.0, 1.0)

    def to_dict(self) -> dict:
        return {
            "shape_id": self.shape_id,
            "kind": self.kind,
            "name": self.name,
            "inverted": bool(self.inverted),
            "feather_px": int(self.feather_px),
            "enabled": bool(self.enabled),
            "curved": bool(self.curved),
            "keyframes": {
                str(k): [[float(x), float(y)] for x, y in pts]
                for k, pts in sorted(self.keyframes.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PolygonMatteShape":
        raw_kf = d.get("keyframes", {}) or {}
        keyframes: dict[int, list[tuple[float, float]]] = {}
        for k, pts in raw_kf.items():
            try:
                fi = int(k)
            except (TypeError, ValueError):
                continue
            parsed = [
                (float(p[0]), float(p[1]))
                for p in (pts or [])
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]
            if len(parsed) >= 3:
                keyframes[fi] = parsed
        return cls(
            shape_id=str(d.get("shape_id") or uuid.uuid4().hex[:8]),
            kind="core" if d.get("kind") == "core" else "garbage",
            name=str(d.get("name") or ""),
            inverted=bool(d.get("inverted", False)),
            feather_px=int(d.get("feather_px", 4)),
            enabled=bool(d.get("enabled", True)),
            curved=bool(d.get("curved", False)),
            keyframes=keyframes,
        )


def _subdivide_catmull_rom(
    pts: list[tuple[float, float]],
    steps: int = 8,
) -> list[tuple[float, float]]:
    """Closed Centripetal/Uniform Catmull-Rom curve subdivision across polygon vertices."""
    n = len(pts)
    if n < 3 or steps < 2:
        return list(pts)
    out: list[tuple[float, float]] = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        for s in range(steps):
            t = float(s) / float(steps)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    return out


class PolygonUndoStack:
    """Multi-step Undo/Redo snapshot stack for PolygonMatteModel edits."""

    def __init__(self, max_depth: int = 40) -> None:
        self._max_depth = max_depth
        self._undo: list[list[dict]] = []
        self._redo: list[list[dict]] = []

    def push(self, model: "PolygonMatteModel") -> None:
        snap = model.to_list()
        self._undo.append(snap)
        if len(self._undo) > self._max_depth:
            self._undo.pop(0)
        self._redo.clear()

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, model: "PolygonMatteModel") -> bool:
        if not self._undo:
            return False
        self._redo.append(model.to_list())
        prev = self._undo.pop()
        restored = PolygonMatteModel.from_list(prev)
        model.shapes = restored.shapes
        return True

    def redo(self, model: "PolygonMatteModel") -> bool:
        if not self._redo:
            return False
        self._undo.append(model.to_list())
        nxt = self._redo.pop()
        restored = PolygonMatteModel.from_list(nxt)
        model.shapes = restored.shapes
        return True


class PolygonMatteModel:
    """Collection of per-clip polygon garbage/core mattes with clip JSON I/O."""

    def __init__(self, shapes: list[PolygonMatteShape] | None = None) -> None:
        self.shapes: list[PolygonMatteShape] = list(shapes or [])

    def add_shape(self, shape: PolygonMatteShape) -> PolygonMatteShape:
        self.shapes.append(shape)
        return shape

    def remove_shape(self, shape_id: str) -> bool:
        before = len(self.shapes)
        self.shapes = [s for s in self.shapes if s.shape_id != shape_id]
        return len(self.shapes) != before

    def clear(self) -> None:
        self.shapes.clear()

    def apply(self, alpha_f32: np.ndarray, frame_index: int) -> np.ndarray:
        return apply_polygon_mattes(alpha_f32, self.shapes, frame_index)

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.shapes]

    @classmethod
    def from_list(cls, items: list[dict] | None) -> "PolygonMatteModel":
        shapes = []
        for item in items or []:
            if isinstance(item, dict):
                s = PolygonMatteShape.from_dict(item)
                if s.keyframes:
                    shapes.append(s)
        return cls(shapes)

    def save_to_clip(self, clip_root: str) -> str:
        path = os.path.join(clip_root, _SHAPES_FILENAME)
        if not self.shapes:
            if os.path.isfile(path):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return path
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"shapes": self.to_list()}, f, indent=2)
        os.replace(tmp, path)
        return path

    @classmethod
    def load_from_clip(cls, clip_root: str) -> "PolygonMatteModel":
        if not clip_root:
            return cls()
        path = os.path.join(clip_root, _SHAPES_FILENAME)
        if not os.path.isfile(path):
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_list(data.get("shapes", []) if isinstance(data, dict) else [])
        except Exception:
            return cls()


def apply_polygon_mattes(
    alpha_f32: np.ndarray,
    shapes: list[PolygonMatteShape] | None,
    frame_index: int,
) -> np.ndarray:
    """Apply enabled polygon garbage and core mattes to a [H, W] float32 alpha array."""
    if not shapes:
        return np.clip(alpha_f32.astype(np.float32, copy=False), 0.0, 1.0)
    out = np.clip(alpha_f32.astype(np.float32, copy=True), 0.0, 1.0)
    h, w = out.shape[:2]
    # Apply core mattes first or garbage mattes first?
    # Garbage mattes erase unwanted regions around edges; Core mattes solidify interior holes.
    # Running core before garbage allows a garbage erase shape to override even a core edge if needed.
    for kind_pass in ("core", "garbage"):
        for shape in shapes:
            if not shape.enabled or shape.kind != kind_pass:
                continue
            m = shape.rasterize(h, w, frame_index)
            if m is None:
                continue
            if kind_pass == "core":
                out = np.maximum(out, m)
            else:
                # garbage: erase inside mask (when inverted=False, m is 1 inside -> multiply by 1 - m)
                out = out * (1.0 - m)
    return np.clip(out, 0.0, 1.0)
