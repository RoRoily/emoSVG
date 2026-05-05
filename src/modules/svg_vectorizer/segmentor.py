"""
Semantic segmentation wrapper.

Primary backend: Segment Anything Model (SAM / SAM2).
Fallback: OpenCV contour detection (no model weights required).

Each backend returns a list of binary masks (np.ndarray bool, HxW),
one per detected region, sorted by area descending.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import VectorizationError

logger = logging.getLogger(__name__)

_SAM_VRAM_GB = 6.5


class Segmentor:
    """
    Produces a list of binary region masks from an input image.

    Parameters
    ----------
    sam_checkpoint:  Path to SAM .pth weights file.
                     If None or missing, falls back to contour detection.
    model_type:      SAM model variant: "vit_h" | "vit_l" | "vit_b".
    registry:        ModelRegistry singleton.
    """

    MODEL_ID = "sam"

    def __init__(
        self,
        sam_checkpoint: Optional[Path] = None,
        model_type: str = "vit_h",
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._checkpoint = Path(sam_checkpoint) if sam_checkpoint else None
        self._model_type = model_type
        self._registry = registry or ModelRegistry.instance()
        self._use_fallback = not self._sam_available()

        if not self._use_fallback:
            self._register_model()
        else:
            logger.warning(
                "SAM weights not found — using contour-detection fallback."
            )

    # ── Public API ────────────────────────────────────────────────────────

    def segment(
        self,
        image_bgr: np.ndarray,
        min_area: int = 100,
    ) -> list[np.ndarray]:
        """
        Return a list of boolean masks (HxW), one per region, area-descending.

        Parameters
        ----------
        image_bgr:  HxWx3 uint8 BGR image.
        min_area:   Discard masks with fewer pixels than this threshold.
        """
        if self._use_fallback:
            masks = self._contour_fallback(image_bgr, min_area)
        else:
            masks = self._sam_segment(image_bgr, min_area)

        # Sort by area descending (largest region first = background)
        masks.sort(key=lambda m: m.sum(), reverse=True)
        return masks

    # ── SAM backend ───────────────────────────────────────────────────────

    def _sam_available(self) -> bool:
        if self._checkpoint is None or not self._checkpoint.exists():
            return False
        try:
            import segment_anything  # noqa: F401
            return True
        except ImportError:
            return False

    def _register_model(self) -> None:
        checkpoint = str(self._checkpoint)
        model_type = self._model_type

        def _loader():
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator  # type: ignore
                sam = sam_model_registry[model_type](checkpoint=checkpoint)
                return SamAutomaticMaskGenerator(
                    sam,
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.95,
                )
            except Exception as exc:
                raise VectorizationError(f"Failed to load SAM: {exc}") from exc

        self._registry.register(self.MODEL_ID, _loader, estimated_vram_gb=_SAM_VRAM_GB)

    def _sam_segment(self, image_bgr: np.ndarray, min_area: int) -> list[np.ndarray]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        with self._registry.model_context(self.MODEL_ID, offload_after=True) as generator:
            try:
                annotations = generator.generate(image_rgb)
            except Exception as exc:
                raise VectorizationError(f"SAM inference failed: {exc}") from exc

        masks = [
            ann["segmentation"].astype(bool)
            for ann in annotations
            if ann["area"] >= min_area
        ]
        return masks

    # ── Contour fallback ──────────────────────────────────────────────────

    @staticmethod
    def _contour_fallback(image_bgr: np.ndarray, min_area: int) -> list[np.ndarray]:
        """
        Produce region masks via colour quantisation + contour detection.
        No model weights required — suitable for testing and CPU-only environments.
        """
        h, w = image_bgr.shape[:2]

        # Quantise colours to reduce noise
        small = cv2.resize(image_bgr, (w, h))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        lab_flat = lab.reshape(-1, 3).astype(np.float32)

        # K-means colour clustering (k=8 gives reasonable region count)
        k = min(8, max(2, h * w // 2000))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(
            lab_flat, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS
        )
        label_map = labels.reshape(h, w)

        masks: list[np.ndarray] = []
        for cluster_id in range(k):
            binary = (label_map == cluster_id).astype(np.uint8) * 255
            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Split into connected components
            n_labels, comp_map, stats, _ = cv2.connectedComponentsWithStats(binary)
            for comp_id in range(1, n_labels):
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area >= min_area:
                    masks.append((comp_map == comp_id).astype(bool))

        return masks
