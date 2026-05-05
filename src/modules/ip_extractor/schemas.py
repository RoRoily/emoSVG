"""Data contracts for the ip_extractor module."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, field_validator


class ExtractionRequest(BaseModel):
    source_image_path: Path
    target_size: tuple[int, int] = (224, 224)   # (width, height) for CLIP encoder
    normalize: bool = True

    @field_validator("source_image_path")
    @classmethod
    def _path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"source_image_path does not exist: {v}")
        return v


class IPFeatures(BaseModel):
    """
    Extracted IP character features.

    image_embeds:   (1, 257, 1024) CLIP image embedding tensor (as numpy).
    face_embeds:    (1, 512) InsightFace embedding, or None if no face detected.
    preprocessed:   (H, W, 3) float32 RGB image after normalisation.
    backend_used:   "ip_adapter" | "clip_fallback"
    """
    image_embeds: object        # np.ndarray
    face_embeds: object | None = None
    preprocessed: object        # np.ndarray
    backend_used: str

    model_config = {"arbitrary_types_allowed": True}
