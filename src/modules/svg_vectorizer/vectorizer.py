"""
SVGVectorizer — top-level orchestrator for the svg_vectorizer module.

Wires together:
    Segmentor  ->  BezierFitter  ->  SVGBuilder
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import VectorizationError

from .bezier_fitter import BezierFitter
from .schemas import VectorizationRequest, VectorizationResult
from .segmentor import Segmentor
from .svg_builder import SVGBuilder

logger = logging.getLogger(__name__)


class SVGVectorizer:
    """
    Converts a raster image to a layered SVG vector file.

    Parameters
    ----------
    sam_checkpoint:  Path to SAM weights. None = contour fallback.
    output_dir:      Root directory for SVG output files.
    registry:        ModelRegistry singleton.
    """

    def __init__(
        self,
        sam_checkpoint: Path | None = None,
        output_dir: Path = Path("outputs/svgs"),
        registry: ModelRegistry | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._segmentor = Segmentor(
            sam_checkpoint=sam_checkpoint,
            registry=registry,
        )
        self._fitter = BezierFitter()
        self._builder = SVGBuilder()

    # ── Public API ────────────────────────────────────────────────────────

    def vectorize(self, request: VectorizationRequest) -> VectorizationResult:
        t0 = time.monotonic()
        logger.info("SVGVectorizer.vectorize: %s", request.source_image_path.name)

        # 1. Load image
        image_bgr = self._load_image(request.source_image_path)

        # 2. Segment into regions
        masks = self._segmentor.segment(image_bgr, min_area=request.min_region_area)
        if not masks:
            raise VectorizationError("Segmentation produced no regions.")
        logger.info("Segmented into %d regions", len(masks))

        # 3. Fit Bezier curves per mask
        self._fitter.tolerance = request.bezier_tolerance
        self._fitter.precision = request.precision
        paths_per_mask = [self._fitter.mask_to_paths(m) for m in masks]

        # 4. Build SVG
        self._builder.precision = request.precision
        self._builder.layer_naming = request.layer_naming
        output_path = self._resolve_output(request)
        layers = self._builder.build(image_bgr, masks, paths_per_mask, output_path)

        elapsed = time.monotonic() - t0
        logger.info("SVGVectorizer done in %.2f s -> %s", elapsed, output_path)

        return VectorizationResult(
            output_path=output_path,
            layer_count=len(layers),
            total_paths=sum(layer.path_count for layer in layers),
            layers=layers,
            backend_used=(
                "contour_fallback"
                if self._segmentor._use_fallback
                else "sam+bezier"
            ),
        )

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/svg_vectorizer.yaml",
        registry: ModelRegistry | None = None,
    ) -> SVGVectorizer:
        from src.core import load_config
        cfg = load_config(config_path)
        checkpoint = cfg.get("sam", {}).get("checkpoint")
        return cls(
            sam_checkpoint=Path(checkpoint) if checkpoint else None,
            output_dir=Path(cfg.get("svg_output", {}).get("output_dir", "outputs/svgs")),
            registry=registry,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        img = cv2.imread(str(path))
        if img is None:
            raise VectorizationError(f"Failed to read image: {path}")
        return img

    def _resolve_output(self, request: VectorizationRequest) -> Path:
        if request.output_path:
            return request.output_path
        stem = request.source_image_path.stem
        ts = int(time.time())
        return self._output_dir / f"{stem}_{ts}.svg"
