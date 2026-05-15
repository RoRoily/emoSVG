"""
FullPipeline — orchestrates all four modules in priority order:
    IPExtractor -> MemeAnimator -> Reconstructor3D -> SVGVectorizer

The pipeline is designed for single-process execution with shared ModelRegistry
so VRAM is managed centrally across all modules.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.core import ModelRegistry
from src.modules.cartoon_analyzer.analyzer import CartoonFaceAnalyzer
from src.modules.cartoon_analyzer.schemas import CartoonFaceAnalysis
from src.modules.cartoon_layer_parser.parser import CartoonLayerParser
from src.modules.cartoon_layer_parser.schemas import CartoonLayerParseResult
from src.modules.ip_extractor.extractor import IPExtractor
from src.modules.ip_extractor.schemas import ExtractionRequest, IPFeatures
from src.modules.meme_animator.animator import MemeAnimator
from src.modules.meme_animator.schemas import (
    AnimationBackend,
    AnimationRequest,
    AnimationResult,
    MemeExpression,
)
from src.modules.reconstructor_3d.reconstructor import Reconstructor3D
from src.modules.reconstructor_3d.schemas import ReconstructionRequest, ReconstructionResult
from src.modules.svg_vectorizer.schemas import VectorizationRequest, VectorizationResult
from src.modules.svg_vectorizer.vectorizer import SVGVectorizer

from .base_pipeline import BasePipeline
from .debug_artifacts import PipelineDebugWriter

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
    seed: int | None = None
    animation_backend: AnimationBackend = AnimationBackend.LIVE_PORTRAIT
    # False (default) = use the peak animation keyframe (exaggerated pose) as
    # input for 3D reconstruction and SVG vectorization.
    # True = use the original source image (preserves canonical proportions).
    use_source_for_3d_svg: bool = False
    # ToonCrafter inter-frame smoothing
    use_toon_crafter: bool = False
    frames_between: int = 4
    # Write stage-0/1/2 diagnostic artifacts for cartoon routing.
    debug: bool = False


@dataclass
class FullPipelineResult:
    ip_features: IPFeatures
    animation: AnimationResult
    reconstruction: ReconstructionResult | None = None
    vectorization: VectorizationResult | None = None
    cartoon_analysis: CartoonFaceAnalysis | None = None
    cartoon_layers: CartoonLayerParseResult | None = None
    debug_dir: Path | None = None
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
        registry: ModelRegistry | None = None,
        ip_adapter_path: Path | None = None,
        live_portrait_path: Path | None = None,
        toon_crafter_path: Path | None = None,
        triposr_model_id: str = "stabilityai/TripoSR",
        sam_checkpoint: Path | None = None,
        output_root: Path = Path("outputs"),
    ) -> None:
        self._registry = registry or ModelRegistry.instance()
        self._output_root = output_root
        self._cartoon_analyzer = CartoonFaceAnalyzer()
        self._cartoon_layer_parser = CartoonLayerParser(analyzer=self._cartoon_analyzer)

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
        registry: ModelRegistry | None = None,
    ) -> FullPipeline:
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

        # Stage 0/1/2 - cartoon geometry and lightweight layer parsing.
        debug_writer: PipelineDebugWriter | None = None
        if request.debug:
            debug_writer = PipelineDebugWriter(
                self._output_root,
                source_stem=request.source_image_path.stem,
                expression=request.expression.value,
            )

        cartoon_analysis: CartoonFaceAnalysis | None = None
        cartoon_layers: CartoonLayerParseResult | None = None
        needs_cartoon_diagnostics = request.debug or request.animation_backend in {
            AnimationBackend.AUTO,
            AnimationBackend.CARTOON_RIG,
        }
        if needs_cartoon_diagnostics:
            resized_source = None
            try:
                resized_source = self._load_debug_source(
                    request.source_image_path,
                    request.resolution,
                )
                cartoon_analysis = self._cartoon_analyzer.analyze_bgr(resized_source)
                cartoon_layers = self._cartoon_layer_parser.parse_bgr(
                    resized_source,
                    cartoon_analysis.geometry,
                    cartoon_analysis.foreground_mask,
                )
                logger.info(
                    "Stage 0/1/2 cartoon diagnostics done: confidence=%.2f layers=%d",
                    cartoon_analysis.geometry.confidence,
                    len(cartoon_layers.layers),
                )
                if debug_writer:
                    debug_writer.write_input(resized_source)
                    debug_writer.write_analysis_overlay(resized_source, cartoon_analysis)
                    debug_writer.write_layer_overlay(resized_source, cartoon_layers)
                    debug_writer.write_layers(cartoon_layers)
                    debug_writer.update_metrics(
                        cartoon_analysis=cartoon_analysis.to_dict(),
                        cartoon_layers=cartoon_layers.to_dict(),
                    )
            except Exception as exc:
                logger.warning("Cartoon diagnostics failed (%s); continuing pipeline.", exc)
                if debug_writer:
                    if resized_source is not None:
                        debug_writer.write_input(resized_source)
                    debug_writer.update_metrics(cartoon_diagnostics_error=str(exc))

        # Step 1 - IP feature extraction.
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
                animation_backend=request.animation_backend,
                ip_image_embeds=ip_features.image_embeds,
                cartoon_analysis=cartoon_analysis,
                cartoon_layers=cartoon_layers,
                use_toon_crafter=request.use_toon_crafter,
                frames_between=request.frames_between,
            )
        )
        logger.info(
            "Step 2/4 Animation done: %d frames -> %s",
            animation.frame_count,
            animation.output_path,
        )

        # Resolve the input path for steps 3 and 4 — computed once and reused.
        # use_source_for_3d_svg=True  → original source image (canonical proportions)
        # use_source_for_3d_svg=False → peak animation keyframe (exaggerated pose, default)
        if debug_writer:
            debug_writer.write_keyframe_contact_sheet(animation.keyframes)
            debug_writer.update_metrics(
                ip_backend=ip_features.backend_used,
                animation_backend=animation.backend_used,
                animation_path=str(animation.output_path),
                frame_count=animation.frame_count,
                duration_ms=animation.duration_ms,
                animation_metrics=animation.metrics,
            )

        def _resolve_secondary_input() -> Path:
            if request.use_source_for_3d_svg:
                return request.source_image_path
            if animation.keyframes:
                peak_kf = animation.keyframes[len(animation.keyframes) // 2]
                return self._save_keyframe(peak_kf.image, request.source_image_path.stem)
            return request.source_image_path

        secondary_input_label = "source image" if request.use_source_for_3d_svg else "peak keyframe"
        secondary_path: Path | None = None
        if request.run_3d or request.run_svg:
            secondary_path = _resolve_secondary_input()

        # Step 3 — 3D reconstruction (optional)
        reconstruction: ReconstructionResult | None = None
        if request.run_3d:
            assert secondary_path is not None
            logger.info("Step 3/4 3D input: %s", secondary_input_label)
            reconstruction = self._reconstructor.reconstruct(
                ReconstructionRequest(source_image_path=secondary_path)
            )
            logger.info(
                "Step 3/4 3D reconstruction done: %d faces",
                reconstruction.mesh_stats.face_count,
            )

        # Step 4 — SVG vectorization (optional)
        vectorization: VectorizationResult | None = None
        if request.run_svg:
            assert secondary_path is not None
            logger.info("Step 4/4 SVG input: %s", secondary_input_label)
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
        if debug_writer:
            debug_writer.update_metrics(
                elapsed_seconds=elapsed,
                reconstruction_backend=(
                    reconstruction.backend_used if reconstruction else "skipped"
                ),
                svg_backend=vectorization.backend_used if vectorization else "skipped",
            )
            debug_writer.write_metrics()

        return FullPipelineResult(
            ip_features=ip_features,
            animation=animation,
            reconstruction=reconstruction,
            vectorization=vectorization,
            cartoon_analysis=cartoon_analysis,
            cartoon_layers=cartoon_layers,
            debug_dir=debug_writer.debug_dir if debug_writer else None,
            elapsed_seconds=elapsed,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _save_keyframe(self, image, stem: str) -> Path:
        """Persist a keyframe numpy array to disk and return its path."""
        tmp_dir = self._output_root / "_tmp_keyframes"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        path = tmp_dir / f"{stem}_peak_{int(time.time())}.png"
        cv2.imwrite(str(path), image)
        return path

    @staticmethod
    def _load_debug_source(path: Path, resolution: tuple[int, int]):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype("float32") / 255.0
            bgr = img[:, :, :3].astype("float32") * alpha + 255.0 * (1.0 - alpha)
            bgr = bgr.astype("uint8")
        else:
            bgr = img[:, :, :3]
        w, h = resolution
        return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)
