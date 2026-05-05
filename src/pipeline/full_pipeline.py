"""
FullPipeline — orchestrates all four modules in priority order:
    IPExtractor -> MemeAnimator -> Reconstructor3D -> SVGVectorizer

The pipeline is designed for single-process execution with shared ModelRegistry
so VRAM is managed centrally across all modules.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.core import ModelRegistry
from src.modules.ip_extractor.extractor import IPExtractor
from src.modules.ip_extractor.schemas import ExtractionRequest, IPFeatures
from src.modules.meme_animator.animator import MemeAnimator
from src.modules.meme_animator.schemas import AnimationRequest, AnimationResult, MemeExpression
from src.modules.reconstructor_3d.reconstructor import Reconstructor3D
from src.modules.reconstructor_3d.schemas import ReconstructionRequest, ReconstructionResult
from src.modules.svg_vectorizer.vectorizer import SVGVectorizer
from src.modules.svg_vectorizer.schemas import VectorizationRequest, VectorizationResult
from .base_pipeline import BasePipeline

logger = logging.getLogger(__name__)


@dataclass
class FullPipelineRequest:
    source_image_path: Path
    expression: MemeExpression = MemeExpression.SHOCK
    output_format: str = "gif"
    fps: int = 24
    resolution: tuple[int, int] = (512, 512)
    run_3d: bool = True
    run_svg: bool = True
    seed: Optional[int] = None
    # False (default) = use the peak animation keyframe (exaggerated pose) as
    # input for 3D reconstruction and SVG vectorization.
    # True = use the original source image (preserves canonical proportions).
    use_source_for_3d_svg: bool = False
    # ToonCrafter inter-frame smoothing
    use_toon_crafter: bool = False
    frames_between: int = 4


@dataclass
class FullPipelineResult:
    ip_features: IPFeatures
    animation: AnimationResult
    reconstruction: Optional[ReconstructionResult] = None
    vectorization: Optional[VectorizationResult] = None
    elapsed_seconds: float = 0.0


class FullPipeline(BasePipeline):
    """
    End-to-end pipeline: IP extraction -> Meme animation -> 3D -> SVG.

    Parameters
    ----------
    registry:            Shared ModelRegistry (singleton by default).
    ip_adapter_path:     Path to IP-Adapter weights directory.
    live_portrait_path:  Path to LivePortrait weights directory.
    toon_crafter_path:   Path to ToonCrafter checkpoint directory.
    triposr_model_id:    HuggingFace model ID or local path for TripoSR.
    sam_checkpoint:      Path to SAM .pth weights file.
    output_root:         Root directory for all generated files.
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        ip_adapter_path: Optional[Path] = None,
        live_portrait_path: Optional[Path] = None,
        toon_crafter_path: Optional[Path] = None,
        triposr_model_id: str = "stabilityai/TripoSR",
        sam_checkpoint: Optional[Path] = None,
        output_root: Path = Path("outputs"),
    ) -> None:
        self._registry = registry or ModelRegistry.instance()
        self._output_root = output_root

        self._ip_extractor = IPExtractor(
            ip_adapter_path=ip_adapter_path,
            registry=self._registry,
        )
        self._animator = MemeAnimator(
            live_portrait_path=live_portrait_path,
            toon_crafter_path=toon_crafter_path,
            output_dir=output_root / "animations",
            registry=self._registry,
        )
        self._reconstructor = Reconstructor3D(
            model_id=triposr_model_id,
            output_dir=output_root / "meshes",
            registry=self._registry,
        )
        self._vectorizer = SVGVectorizer(
            sam_checkpoint=sam_checkpoint,
            output_dir=output_root / "svgs",
            registry=self._registry,
        )

    # ── BasePipeline interface ────────────────────────────────────────────

    def run(self, **kwargs) -> FullPipelineResult:
        request = FullPipelineRequest(**kwargs)
        return self.execute(request)

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/base.yaml",
        registry: Optional[ModelRegistry] = None,
    ) -> "FullPipeline":
        from src.core import load_config
        base = load_config(config_path)
        models_root = Path(base.get("models_root", "./models"))
        output_root = Path(base.get("output_root", "./outputs"))
        return cls(
            registry=registry,
            ip_adapter_path=models_root / "ip_adapter",
            live_portrait_path=models_root / "live_portrait",
            toon_crafter_path=models_root / "toon_crafter",
            sam_checkpoint=models_root / "sam" / "sam_vit_h_4b8939.pth",
            output_root=output_root,
        )

    # ── Main execution ────────────────────────────────────────────────────

    def execute(self, request: FullPipelineRequest) -> FullPipelineResult:
        t0 = time.monotonic()
        logger.info(
            "FullPipeline.execute: %s expression=%s",
            request.source_image_path.name,
            request.expression.value,
        )

        # Step 1 — IP feature extraction
        ip_features = self._ip_extractor.extract(
            ExtractionRequest(source_image_path=request.source_image_path)
        )
        logger.info("Step 1/4 IP extraction done (backend=%s)", ip_features.backend_used)

        # Step 2 — Meme animation (core deliverable)
        # Pass IP image embeddings so LivePortrait can condition on character identity.
        animation = self._animator.generate(
            AnimationRequest(
                source_image_path=request.source_image_path,
                expression=request.expression,
                output_format=request.output_format,
                fps=request.fps,
                resolution=request.resolution,
                seed=request.seed,
                ip_image_embeds=ip_features.image_embeds,
                use_toon_crafter=request.use_toon_crafter,
                frames_between=request.frames_between,
            )
        )
        logger.info(
            "Step 2/4 Animation done: %d frames -> %s",
            animation.frame_count,
            animation.output_path,
        )

        # Resolve the input path for steps 3 and 4.
        # use_source_for_3d_svg=True  → original source image (canonical proportions)
        # use_source_for_3d_svg=False → peak animation keyframe (exaggerated pose, default)
        def _resolve_secondary_input() -> Path:
            if request.use_source_for_3d_svg:
                return request.source_image_path
            if animation.keyframes:
                peak_kf = animation.keyframes[len(animation.keyframes) // 2]
                return self._save_keyframe(peak_kf.image, request.source_image_path.stem)
            return request.source_image_path  # fallback if no keyframes

        # Step 3 — 3D reconstruction (optional)
        reconstruction: Optional[ReconstructionResult] = None
        if request.run_3d:
            secondary_path = _resolve_secondary_input()
            logger.info(
                "Step 3/4 3D input: %s",
                "source image" if request.use_source_for_3d_svg else "peak keyframe",
            )
            reconstruction = self._reconstructor.reconstruct(
                ReconstructionRequest(source_image_path=secondary_path)
            )
            logger.info(
                "Step 3/4 3D reconstruction done: %d faces",
                reconstruction.mesh_stats.face_count,
            )

        # Step 4 — SVG vectorization (optional)
        vectorization: Optional[VectorizationResult] = None
        if request.run_svg:
            secondary_path = _resolve_secondary_input()
            logger.info(
                "Step 4/4 SVG input: %s",
                "source image" if request.use_source_for_3d_svg else "peak keyframe",
            )
            vectorization = self._vectorizer.vectorize(
                VectorizationRequest(source_image_path=secondary_path)
            )
            logger.info(
                "Step 4/4 SVG vectorization done: %d layers -> %s",
                vectorization.layer_count,
                vectorization.output_path,
            )

        elapsed = time.monotonic() - t0
        logger.info("FullPipeline.execute done in %.2f s", elapsed)

        return FullPipelineResult(
            ip_features=ip_features,
            animation=animation,
            reconstruction=reconstruction,
            vectorization=vectorization,
            elapsed_seconds=elapsed,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _save_keyframe(self, image, stem: str) -> Path:
        """Persist a keyframe numpy array to disk and return its path."""
        import cv2
        import numpy as np
        tmp_dir = self._output_root / "_tmp_keyframes"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"{stem}_peak_{int(time.time())}.png"
        cv2.imwrite(str(path), image)
        return path
