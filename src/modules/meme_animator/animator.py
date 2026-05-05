"""
MemeAnimator — top-level orchestrator for the meme animation module.

Wires together:
    MotionDesigner  →  LivePortraitWrapper  →  FrameComposer

Usage
-----
    animator = MemeAnimator.from_config("configs/meme_animation.yaml")
    result   = animator.generate(request)
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import AnimationError
from .frame_composer import FrameComposer
from .live_portrait import LivePortraitWrapper
from .motion_designer import MotionDesigner
from .toon_crafter import ToonCrafterWrapper
from .schemas import AnimationRequest, AnimationResult, KeyFrame

logger = logging.getLogger(__name__)


class MemeAnimator:
    """
    Generates a Squash-and-Stretch meme animation from a single source image.

    Parameters
    ----------
    live_portrait_path:
        Path to LivePortrait model weights directory.
        Pass None to use the affine-warp fallback (testing / no-GPU mode).
    toon_crafter_path:
        Path to ToonCrafter checkpoint directory (model.ckpt + config.yaml).
        Pass None to skip inter-frame smoothing (or use linear-blend fallback).
    output_dir:
        Root directory for generated animation files.
    registry:
        ModelRegistry singleton. Defaults to the process-wide instance.
    """

    def __init__(
        self,
        live_portrait_path: Optional[Path] = None,
        toon_crafter_path: Optional[Path] = None,
        output_dir: Path = Path("outputs/animations"),
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._output_dir = output_dir
        self._motion_designer = MotionDesigner()
        self._live_portrait = LivePortraitWrapper(
            model_path=live_portrait_path,
            registry=registry,
        )
        self._toon_crafter = ToonCrafterWrapper(
            model_path=toon_crafter_path,
            registry=registry,
        )
        self._composer = FrameComposer()

    # ── Public API ────────────────────────────────────────────────────────

    def generate(self, request: AnimationRequest) -> AnimationResult:
        """
        Full pipeline: source image -> per-frame params -> rendered frames -> file.

        Parameters
        ----------
        request: Validated AnimationRequest.

        Returns
        -------
        AnimationResult with output path and frame metadata.
        """
        t0 = time.monotonic()
        logger.info(
            "MemeAnimator.generate: expression=%s format=%s fps=%d",
            request.expression.value,
            request.output_format,
            request.fps,
        )

        # 1. Load source image
        source_bgr = self._load_image(request.source_image_path, request.resolution)

        # 2. Design motion curve
        param_sequence = self._motion_designer.design(
            request.expression,
            custom_params=request.custom_params,
        )
        logger.info("Motion curve: %d frames designed", len(param_sequence))

        # 3. Render keyframes via LivePortrait (model stays on GPU for the full batch)
        try:
            rendered_frames = self._live_portrait.render_sequence(
                source_bgr,
                param_sequence,
                ip_image_embeds=request.ip_image_embeds,
            )
        except Exception as exc:
            raise AnimationError(f"Frame rendering failed: {exc}") from exc

        # 4. Optional ToonCrafter inter-frame smoothing.
        # LivePortrait produces one frame per SquashParams step; ToonCrafter fills
        # the gaps between consecutive keyframes for smoother cartoon motion.
        toon_crafter_used = False
        if request.use_toon_crafter and len(rendered_frames) >= 2:
            logger.info(
                "ToonCrafter smoothing: %d keyframes -> %d frames_between",
                len(rendered_frames),
                request.frames_between,
            )
            try:
                rendered_frames = self._toon_crafter.smooth_sequence(
                    rendered_frames,
                    frames_between=request.frames_between,
                )
                toon_crafter_used = not self._toon_crafter._use_fallback
                logger.info("ToonCrafter smoothing done: %d total frames", len(rendered_frames))
            except Exception as exc:
                logger.warning("ToonCrafter smoothing failed (%s) — using unsmoothed frames.", exc)

        # 5. Build KeyFrame metadata (sample every 5th frame to keep result small)
        ms_per_frame = 1000.0 / request.fps
        keyframes = [
            KeyFrame(
                index=i,
                timestamp_ms=i * ms_per_frame,
                image=rendered_frames[i],
                params=param_sequence[min(i, len(param_sequence) - 1)],
            )
            for i in range(0, len(rendered_frames), 5)
        ]

        # 6. Compose output file
        output_path = self._output_dir / self._make_filename(request)
        self._composer.compose(
            frames=rendered_frames,
            output_path=output_path,
            fps=request.fps,
            resolution=request.resolution,
            output_format=request.output_format,
        )

        elapsed = time.monotonic() - t0
        logger.info("MemeAnimator.generate done in %.2f s -> %s", elapsed, output_path)

        # Build backend_used string reflecting which backends were active
        lp_backend = (
            "live_portrait_fallback" if self._live_portrait._use_fallback else "live_portrait"
        )
        if request.use_toon_crafter:
            tc_backend = (
                "toon_crafter_fallback" if self._toon_crafter._use_fallback else "toon_crafter"
            )
            backend = f"{lp_backend}+{tc_backend}"
        else:
            backend = lp_backend

        return AnimationResult(
            output_path=output_path,
            frame_count=len(rendered_frames),
            duration_ms=len(rendered_frames) * ms_per_frame,
            fps=request.fps,
            keyframes=keyframes,
            backend_used=backend,
        )

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/meme_animation.yaml",
        registry: Optional[ModelRegistry] = None,
    ) -> "MemeAnimator":
        from src.core import load_config
        cfg = load_config(config_path)
        lp_path_str = cfg.get("live_portrait", {}).get("model_path")
        tc_path_str = cfg.get("toon_crafter", {}).get("model_path")
        return cls(
            live_portrait_path=Path(lp_path_str) if lp_path_str else None,
            toon_crafter_path=Path(tc_path_str) if tc_path_str else None,
            output_dir=Path(cfg.get("output", {}).get("output_dir", "outputs/animations")),
            registry=registry,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(path: Path, resolution: tuple[int, int]) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise AnimationError(f"Failed to read image: {path}")
        w, h = resolution
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _make_filename(request: AnimationRequest) -> str:
        stem = request.source_image_path.stem
        expr = request.expression.value
        ts   = int(time.time())
        return f"{stem}_{expr}_{ts}.{request.output_format}"
