from .model import CorridorKeyV3
from .chroma_key import chroma_key_matte


class CorridorKeyV2(CorridorKeyV3):
    """Public CorridorKey v2 runtime alias for the exported model."""


__all__ = ["CorridorKeyV2", "chroma_key_matte"]
