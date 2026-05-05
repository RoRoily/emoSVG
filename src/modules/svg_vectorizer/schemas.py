"""Data contracts for the svg_vectorizer module."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class VectorizationRequest(BaseModel):
    source_image_path: Path
    output_path: Path | None = None   # auto-derived from source if None
    num_segments: int = Field(0, ge=0)   # 0 = auto-detect via SAM
    bezier_tolerance: float = Field(1.5, ge=0.1, le=20.0)
    min_region_area: int = Field(100, ge=1)
    layer_naming: str = Field("semantic", pattern="^(semantic|index)$")
    precision: int = Field(2, ge=0, le=6)

    @field_validator("source_image_path")
    @classmethod
    def _path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"source_image_path does not exist: {v}")
        return v


class SVGLayer(BaseModel):
    layer_id: str
    label: str
    path_count: int
    fill_color: str    # CSS hex, e.g. "#ff6600"
    opacity: float = 1.0


class VectorizationResult(BaseModel):
    output_path: Path
    layer_count: int
    total_paths: int
    layers: list[SVGLayer]
    backend_used: str   # "sam+bezier" | "contour_fallback"

    model_config = {"arbitrary_types_allowed": True}
