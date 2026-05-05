"""Data contracts for the reconstructor_3d module."""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class ExportFormat(str, Enum):
    OBJ = "obj"
    GLB = "glb"


class ReconstructionRequest(BaseModel):
    source_image_path: Path
    export_formats: list[ExportFormat] = [ExportFormat.OBJ, ExportFormat.GLB]
    mc_resolution: int = Field(256, ge=64, le=512)
    remove_background: bool = True
    foreground_ratio: float = Field(0.85, ge=0.5, le=1.0)

    @field_validator("source_image_path")
    @classmethod
    def _path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"source_image_path does not exist: {v}")
        return v


class MeshStats(BaseModel):
    vertex_count: int
    face_count: int
    is_watertight: bool
    bounding_box: tuple[float, float, float]  # (x, y, z) extents


class ReconstructionResult(BaseModel):
    output_paths: dict[str, Path]   # format -> path, e.g. {"obj": Path(...)}
    mesh_stats: MeshStats
    backend_used: str               # "triposr" | "triposr_fallback"

    model_config = {"arbitrary_types_allowed": True}
