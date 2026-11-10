"""SAM 3.1 & Guided Mask Tracker package for CorridorKey 2.0."""

from .wrapper import (
    PromptFrame,
    SAM3NotInstalledError,
    SAM3Tracker,
    clean_binary_mask,
    refine_hair_matte,
)

__all__ = [
    "PromptFrame",
    "SAM3NotInstalledError",
    "SAM3Tracker",
    "clean_binary_mask",
    "refine_hair_matte",
]

