"""Build a minimal 2D rig asset from cartoon geometry and parsed layers."""
from __future__ import annotations

import cv2
import numpy as np

from src.modules.cartoon_analyzer.analyzer import CartoonFaceAnalyzer
from src.modules.cartoon_analyzer.schemas import CartoonFaceAnalysis, Point2D
from src.modules.cartoon_layer_parser.parser import CartoonLayerParser
from src.modules.cartoon_layer_parser.schemas import CartoonLayerParseResult

from .schemas import CartoonRigAsset, RigLayer


class CartoonRigBuilder:
    """Converts geometry-prompted masks into a simple editable 2D rig."""

    _Z_ORDER = {
        "face_base_clean": 0,
        "left_eye": 10,
        "right_eye": 10,
        "mouth": 20,
    }

    def __init__(
        self,
        analyzer: CartoonFaceAnalyzer | None = None,
        parser: CartoonLayerParser | None = None,
    ) -> None:
        self._analyzer = analyzer or CartoonFaceAnalyzer()
        self._parser = parser or CartoonLayerParser(analyzer=self._analyzer)

    def build(
        self,
        source_bgr: np.ndarray,
        analysis: CartoonFaceAnalysis | None = None,
        layers: CartoonLayerParseResult | None = None,
    ) -> CartoonRigAsset:
        if analysis is None:
            analysis = self._analyzer.analyze_bgr(source_bgr)
        if layers is None:
            layers = self._parser.parse_bgr(
                source_bgr,
                analysis.geometry,
                analysis.foreground_mask,
            )

        feature_mask = self._feature_mask(layers, source_bgr.shape[:2])
        base_bgr = self._erase_features(source_bgr, feature_mask)
        rig_layers: list[RigLayer] = []

        for layer in layers.layers:
            if layer.name not in self._Z_ORDER:
                continue
            rig_layers.append(
                RigLayer(
                    name=layer.name,
                    rgba=layer.rgba.copy(),
                    bbox=layer.bbox,
                    pivot=Point2D(layer.bbox.center.x, layer.bbox.center.y),
                    role=layer.role,
                    z_index=self._Z_ORDER[layer.name],
                )
            )

        rig_layers.sort(key=lambda layer: layer.z_index)
        h, w = source_bgr.shape[:2]
        return CartoonRigAsset(
            canvas_size=(w, h),
            base_bgr=base_bgr,
            geometry=analysis.geometry,
            layers=rig_layers,
        )

    @staticmethod
    def _feature_mask(layers: CartoonLayerParseResult, image_shape: tuple[int, int]) -> np.ndarray:
        feature_layers = [
            layer.mask
            for layer in layers.layers
            if layer.name in {"left_eye", "right_eye", "mouth"}
        ]
        if not feature_layers:
            return np.zeros(image_shape, dtype=np.uint8)
        mask = np.zeros_like(feature_layers[0])
        for part_mask in feature_layers:
            mask = cv2.bitwise_or(mask, (part_mask > 24).astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    @staticmethod
    def _erase_features(source_bgr: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
        if np.count_nonzero(feature_mask) == 0:
            return source_bgr.copy()
        radius = max(3, int(min(source_bgr.shape[:2]) * 0.018))
        clean = cv2.inpaint(source_bgr, feature_mask, radius, cv2.INPAINT_TELEA)
        alpha = cv2.GaussianBlur(feature_mask, (9, 9), 0).astype(np.float32) / 255.0
        alpha = alpha[:, :, None]
        return (clean.astype(np.float32) * alpha + source_bgr.astype(np.float32) * (1.0 - alpha)).astype(
            np.uint8
        )
