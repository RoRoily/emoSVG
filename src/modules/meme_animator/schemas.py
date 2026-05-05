"""Data contracts for the meme_animator module."""
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator


class MemeExpression(str, Enum):
    SHOCK     = "shock"
    LAUGH     = "laugh"
    CRY       = "cry"
    RAGE      = "rage"
    SMUG      = "smug"
    SURPRISED = "surprised"
    CUSTOM    = "custom"


class SquashParams(BaseModel):
    eye_bulge_scale: float     = Field(1.0, ge=0.1, le=5.0)
    eye_squint_scale: float    = Field(1.0, ge=0.1, le=2.0)
    brow_raise_offset: float   = Field(0.0, ge=-1.0, le=1.0)
    jaw_drop_scale: float      = Field(1.0, ge=0.5, le=3.0)
    mouth_width_scale: float   = Field(1.0, ge=0.5, le=2.5)
    mouth_corner_offset: float = Field(0.0, ge=-1.0, le=1.0)
    head_squash_scale: float   = Field(1.0, ge=0.3, le=1.5)
    head_stretch_scale: float  = Field(1.0, ge=0.5, le=2.0)
    head_tilt_deg: float       = Field(0.0, ge=-30.0, le=30.0)
    hold_frames: int           = Field(4, ge=1, le=30)
    bounce_frames: int         = Field(3, ge=0, le=10)


class AnimationRequest(BaseModel):
    source_image_path: Path
    expression: MemeExpression = MemeExpression.SHOCK
    custom_params: Optional[SquashParams] = None
    output_format: str = Field("gif", pattern="^(gif|mp4|webp)$")
    fps: int = Field(24, ge=8, le=60)
    resolution: tuple[int, int] = (512, 512)
    seed: Optional[int] = None
    # ToonCrafter inter-frame smoothing (applied after LivePortrait rendering)
    use_toon_crafter: bool = False
    frames_between: int = Field(4, ge=1, le=16)
    # IP-Adapter character consistency embedding (from IPExtractor).
    # When provided, LivePortrait uses it to condition appearance features
    # so the character identity is preserved across deformation frames.
    ip_image_embeds: Optional[object] = None  # np.ndarray (1, D) or None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("source_image_path")
    @classmethod
    def _path_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"source_image_path does not exist: {v}")
        return v

    @field_validator("custom_params", mode="after")
    @classmethod
    def _custom_requires_params(cls, v, info):
        data = getattr(info, "data", {})
        if data.get("expression") == MemeExpression.CUSTOM and v is None:
            raise ValueError("custom_params must be provided when expression=CUSTOM")
        return v


class KeyFrame(BaseModel):
    index: int
    timestamp_ms: float
    image: object
    params: SquashParams
    model_config = {"arbitrary_types_allowed": True}


class AnimationResult(BaseModel):
    output_path: Path
    frame_count: int
    duration_ms: float
    fps: int
    keyframes: list[KeyFrame]
    backend_used: str
    model_config = {"arbitrary_types_allowed": True}
