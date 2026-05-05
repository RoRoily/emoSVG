"""
Convenience partial pipelines for single-module invocation.
Each wraps one module and exposes the same BasePipeline interface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.core import ModelRegistry
from src.modules.meme_animator.animator import MemeAnimator
from src.modules.meme_animator.schemas import AnimationRequest, AnimationResult
from src.modules.reconstructor_3d.reconstructor import Reconstructor3D
from src.modules.reconstructor_3d.schemas import ReconstructionRequest, ReconstructionResult
from src.modules.svg_vectorizer.vectorizer import SVGVectorizer
from src.modules.svg_vectorizer.schemas import VectorizationRequest, VectorizationResult
from .base_pipeline import BasePipeline


class AnimatePipeline(BasePipeline):
    """Single-module pipeline: source image -> meme animation."""

    def __init__(
        self,
        live_portrait_path: Optional[Path] = None,
        toon_crafter_path: Optional[Path] = None,
        output_dir: Path = Path("outputs/animations"),
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._animator = MemeAnimator(
            live_portrait_path=live_portrait_path,
            toon_crafter_path=toon_crafter_path,
            output_dir=output_dir,
            registry=registry,
        )

    def run(self, **kwargs) -> AnimationResult:
        return self._animator.generate(AnimationRequest(**kwargs))

    @classmethod
    def from_config(cls, config_path: str = "configs/meme_animation.yaml") -> "AnimatePipeline":
        return MemeAnimator.from_config(config_path)


class ReconstructPipeline(BasePipeline):
    """Single-module pipeline: source image -> 3D mesh."""

    def __init__(
        self,
        model_id: str = "stabilityai/TripoSR",
        output_dir: Path = Path("outputs/meshes"),
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._reconstructor = Reconstructor3D(
            model_id=model_id,
            output_dir=output_dir,
            registry=registry,
        )

    def run(self, **kwargs) -> ReconstructionResult:
        return self._reconstructor.reconstruct(ReconstructionRequest(**kwargs))

    @classmethod
    def from_config(cls, config_path: str = "configs/reconstruction_3d.yaml") -> "ReconstructPipeline":
        return Reconstructor3D.from_config(config_path)


class VectorizePipeline(BasePipeline):
    """Single-module pipeline: source image -> SVG."""

    def __init__(
        self,
        sam_checkpoint: Optional[Path] = None,
        output_dir: Path = Path("outputs/svgs"),
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._vectorizer = SVGVectorizer(
            sam_checkpoint=sam_checkpoint,
            output_dir=output_dir,
            registry=registry,
        )

    def run(self, **kwargs) -> VectorizationResult:
        return self._vectorizer.vectorize(VectorizationRequest(**kwargs))

    @classmethod
    def from_config(cls, config_path: str = "configs/svg_vectorizer.yaml") -> "VectorizePipeline":
        return SVGVectorizer.from_config(config_path)
