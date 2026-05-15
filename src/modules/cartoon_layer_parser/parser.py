"""Geometry-prompted cartoon layer parser.

The parser deliberately starts with deterministic masks around detected Q-style
parts. It gives downstream rigging a stable layer contract today, while leaving
room for SAM/HQ-SAM/See-through adapters behind the same interface later.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.modules.cartoon_analyzer.analyzer import CartoonFaceAnalyzer
from src.modules.cartoon_analyzer.schemas import BBox, CartoonFaceAnalysis, CartoonFaceGeometry

from .schemas import CartoonLayer, CartoonLayerParseResult

logger = logging.getLogger(__name__)


class CartoonLayerParser:
    """Create coarse rig-ready RGBA layers from cartoon geometry."""

    def __init__(self, analyzer: CartoonFaceAnalyzer | None = None) -> None:
        self._analyzer = analyzer or CartoonFaceAnalyzer()

    def parse(
        self,
        source_image_path: Path,
        analysis: CartoonFaceAnalysis | None = None,
        resolution: tuple[int, int] | None = None,
    ) -> CartoonLayerParseResult:
        bgr = CartoonFaceAnalyzer._load_bgr(source_image_path, resolution)
        if analysis is None:
            analysis = self._analyzer.analyze_bgr(bgr)
        return self.parse_bgr(bgr, analysis.geometry, analysis.foreground_mask)

    def parse_bgr(
        self,
        image_bgr: np.ndarray,
        geometry: CartoonFaceGeometry,
        foreground_mask: np.ndarray | None = None,
    ) -> CartoonLayerParseResult:
        h, w = image_bgr.shape[:2]
        if foreground_mask is None:
            foreground_mask = CartoonFaceAnalyzer._foreground_mask(image_bgr)

        eye_left_mask = self._soft_ellipse_mask((h, w), geometry.left_eye_bbox.pad(3, 3))
        eye_right_mask = self._soft_ellipse_mask((h, w), geometry.right_eye_bbox.pad(3, 3))
        mouth_mask = self._soft_ellipse_mask((h, w), geometry.mouth_bbox.pad(3, 2))
        face_mask = self._face_mask((h, w), geometry.face_bbox)
        feature_mask = cv2.bitwise_or(cv2.bitwise_or(eye_left_mask, eye_right_mask), mouth_mask)

        clean_face = self._inpaint_features(image_bgr, feature_mask)
        hair_mask = self._hair_like_mask(foreground_mask, face_mask, geometry.face_bbox)
        hand_mask = self._hand_like_mask(image_bgr, foreground_mask, geometry.face_bbox)

        layers = [
            CartoonLayer(
                name="foreground",
                rgba=self._rgba_from_mask(image_bgr, foreground_mask),
                mask=foreground_mask,
                bbox=self._mask_bbox_or_full(foreground_mask),
                confidence=0.9,
                role="source",
            ),
            CartoonLayer(
                name="face_base_clean",
                rgba=self._rgba_from_mask(clean_face, face_mask),
                mask=face_mask,
                bbox=geometry.face_bbox,
                confidence=0.55,
                role="base",
            ),
            CartoonLayer(
                name="left_eye",
                rgba=self._rgba_from_mask(image_bgr, eye_left_mask),
                mask=eye_left_mask,
                bbox=geometry.left_eye_bbox,
                confidence=geometry.confidence,
                role="eye",
            ),
            CartoonLayer(
                name="right_eye",
                rgba=self._rgba_from_mask(image_bgr, eye_right_mask),
                mask=eye_right_mask,
                bbox=geometry.right_eye_bbox,
                confidence=geometry.confidence,
                role="eye",
            ),
            CartoonLayer(
                name="mouth",
                rgba=self._rgba_from_mask(image_bgr, mouth_mask),
                mask=mouth_mask,
                bbox=geometry.mouth_bbox,
                confidence=geometry.confidence,
                role="mouth",
            ),
        ]

        warnings: list[str] = []
        if np.count_nonzero(hair_mask) > max(16, int(0.01 * h * w)):
            layers.append(
                CartoonLayer(
                    name="hair_front",
                    rgba=self._rgba_from_mask(image_bgr, hair_mask),
                    mask=hair_mask,
                    bbox=self._mask_bbox_or_full(hair_mask),
                    confidence=0.35,
                    role="occluder",
                )
            )
        else:
            warnings.append("hair_front mask too weak; skipped")

        if np.count_nonzero(hand_mask) > max(16, int(0.006 * h * w)):
            layers.append(
                CartoonLayer(
                    name="hands",
                    rgba=self._rgba_from_mask(image_bgr, hand_mask),
                    mask=hand_mask,
                    bbox=self._mask_bbox_or_full(hand_mask),
                    confidence=0.3,
                    role="occluder",
                )
            )
        else:
            warnings.append("hands mask too weak; skipped")

        logger.info("Cartoon layer parser produced %d layers", len(layers))
        return CartoonLayerParseResult(layers=layers, warnings=warnings)

    @staticmethod
    def _rgba_from_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        return rgba

    @staticmethod
    def _soft_ellipse_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
        h, w = shape
        box = bbox.clamp(w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(box.center.x), int(box.center.y))
        axes = (max(2, int(box.w * 0.58)), max(2, int(box.h * 0.62)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        blur = max(3, int(min(box.w, box.h) * 0.16) | 1)
        return cv2.GaussianBlur(mask, (blur, blur), 0)

    @staticmethod
    def _face_mask(shape: tuple[int, int], face: BBox) -> np.ndarray:
        h, w = shape
        box = face.clamp(w, h)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (int(box.center.x), int(box.y + box.h * 0.53))
        axes = (max(4, int(box.w * 0.42)), max(4, int(box.h * 0.43)))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        return cv2.GaussianBlur(mask, (9, 9), 0)

    @staticmethod
    def _inpaint_features(image_bgr: np.ndarray, feature_mask: np.ndarray) -> np.ndarray:
        hard = (feature_mask > 32).astype(np.uint8) * 255
        if np.count_nonzero(hard) == 0:
            return image_bgr.copy()
        radius = max(3, int(min(image_bgr.shape[:2]) * 0.015))
        return cv2.inpaint(image_bgr, hard, radius, cv2.INPAINT_TELEA)

    @staticmethod
    def _hair_like_mask(
        foreground_mask: np.ndarray,
        face_mask: np.ndarray,
        face: BBox,
    ) -> np.ndarray:
        h, w = foreground_mask.shape[:2]
        upper = np.zeros_like(foreground_mask)
        upper[: min(h, int(face.y + face.h * 0.72)), :] = 255
        not_face = cv2.bitwise_not((face_mask > 48).astype(np.uint8) * 255)
        mask = cv2.bitwise_and(foreground_mask, upper)
        mask = cv2.bitwise_and(mask, not_face)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    @staticmethod
    def _hand_like_mask(
        image_bgr: np.ndarray,
        foreground_mask: np.ndarray,
        face: BBox,
    ) -> np.ndarray:
        h, _ = foreground_mask.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Loose skin/peach heuristic for common chibi hands. It is intentionally
        # low-confidence and only feeds debug/rig candidates, not final masks.
        skin = (((hue < 28) | (hue > 165)) & (sat > 15) & (sat < 125) & (val > 120))
        lower = np.zeros_like(foreground_mask, dtype=bool)
        lower[int(face.y + face.h * 0.52) : min(h, int(face.y + face.h * 1.03)), :] = True
        mask = (skin & lower & (foreground_mask > 0)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask

    @staticmethod
    def _mask_bbox_or_full(mask: np.ndarray) -> BBox:
        points = cv2.findNonZero(mask)
        if points is None:
            h, w = mask.shape[:2]
            return BBox(0, 0, w, h)
        x, y, w, h = cv2.boundingRect(points)
        return BBox(x, y, w, h)
