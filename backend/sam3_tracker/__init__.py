"""SAM 3.1 & Guided Mask Tracker package for CorridorKey 2.0."""

from .hair_matting import (
    HairMattingConfig,
    build_hair_roi_mask,
    guided_filter_rgb,
    refine_hair_boundary_tracked,
)
from .wrapper import (
    PromptFrame,
    SAM3NotInstalledError,
    SAM3Tracker,
    clean_binary_mask,
    refine_hair_matte,
)

__all__ = [
    "HairMattingConfig",
    "PromptFrame",
    "SAM3NotInstalledError",
    "SAM3Tracker",
    "build_hair_roi_mask",
    "clean_binary_mask",
    "guided_filter_rgb",
    "refine_hair_boundary_tracked",
    "refine_hair_matte",
]

