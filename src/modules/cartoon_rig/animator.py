"""Lightweight 2D rig animator for Q-style cartoon characters."""
from __future__ import annotations

import math

import cv2
import numpy as np

from src.modules.cartoon_analyzer.schemas import CartoonFaceAnalysis
from src.modules.cartoon_layer_parser.schemas import CartoonLayerParseResult
from src.modules.meme_animator.schemas import SquashParams

from .builder import CartoonRigBuilder
from .schemas import CartoonRigAsset, RigLayer


class CartoonRigAnimator:
    """Render expression frames by transforming explicit eyes/mouth layers."""

    def __init__(self, builder: CartoonRigBuilder | None = None) -> None:
        self._builder = builder or CartoonRigBuilder()

    def render_sequence(
        self,
        source_bgr: np.ndarray,
        params_sequence: list[SquashParams],
        analysis: CartoonFaceAnalysis | None = None,
        layers: CartoonLayerParseResult | None = None,
    ) -> list[np.ndarray]:
        asset = self._builder.build(source_bgr, analysis=analysis, layers=layers)
        total = max(1, len(params_sequence))
        return [
            self.render_frame(asset, params, frame_index=i, total_frames=total)
            for i, params in enumerate(params_sequence)
        ]

    def render_frame(
        self,
        asset: CartoonRigAsset,
        params: SquashParams,
        frame_index: int = 0,
        total_frames: int = 1,
    ) -> np.ndarray:
        canvas = asset.base_bgr.copy()
        for layer in asset.layers:
            if layer.name == "face_base_clean":
                continue
            transformed = self._transform_layer(layer, params, frame_index, total_frames)
            canvas = self._alpha_composite(canvas, transformed)
        return self._apply_head_motion(canvas, asset, params, frame_index, total_frames)

    def _transform_layer(
        self,
        layer: RigLayer,
        params: SquashParams,
        frame_index: int,
        total_frames: int,
    ) -> np.ndarray:
        sx, sy, dx, dy, angle = self._layer_transform(layer, params, frame_index, total_frames)
        return self._warp_rgba(layer.rgba, layer.pivot.x, layer.pivot.y, sx, sy, angle, dx, dy)

    @staticmethod
    def _layer_transform(
        layer: RigLayer,
        params: SquashParams,
        frame_index: int,
        total_frames: int,
    ) -> tuple[float, float, float, float, float]:
        phase = 2.0 * math.pi * frame_index / max(1, total_frames - 1)
        jitter = math.sin(phase * 3.0) * 0.8
        if layer.name in {"left_eye", "right_eye"}:
            bulge = max(0.0, params.eye_bulge_scale - 1.0)
            squint = params.eye_squint_scale
            sx = 1.0 + 0.20 * bulge
            sy = max(0.18, 1.0 + 0.12 * bulge) * max(0.25, squint)
            dy = -params.brow_raise_offset * max(1.0, layer.bbox.h * 0.12)
            dx = jitter if bulge > 0.6 else 0.0
            return sx, sy, dx, dy, 0.0

        if layer.name == "mouth":
            sx = 1.0 + (params.mouth_width_scale - 1.0) * 0.75
            sy = 1.0 + (params.jaw_drop_scale - 1.0) * 0.95
            dx = params.mouth_corner_offset * layer.bbox.w * 0.08
            dy = (params.jaw_drop_scale - 1.0) * layer.bbox.h * 0.28
            return max(0.3, sx), max(0.25, sy), dx, dy, 0.0

        return 1.0, 1.0, 0.0, 0.0, 0.0

    @staticmethod
    def _warp_rgba(
        rgba: np.ndarray,
        cx: float,
        cy: float,
        sx: float,
        sy: float,
        angle_deg: float,
        dx: float,
        dy: float,
    ) -> np.ndarray:
        h, w = rgba.shape[:2]
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        matrix = np.array(
            [
                [sx * cos_a, -sy * sin_a, cx + dx - sx * cos_a * cx + sy * sin_a * cy],
                [sx * sin_a, sy * cos_a, cy + dy - sx * sin_a * cx - sy * cos_a * cy],
            ],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            rgba,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

    @staticmethod
    def _alpha_composite(base_bgr: np.ndarray, layer_rgba: np.ndarray) -> np.ndarray:
        alpha = layer_rgba[:, :, 3:4].astype(np.float32) / 255.0
        if float(alpha.max()) <= 0.0:
            return base_bgr
        layer_bgr = layer_rgba[:, :, :3].astype(np.float32)
        base = base_bgr.astype(np.float32)
        return (layer_bgr * alpha + base * (1.0 - alpha)).astype(np.uint8)

    @staticmethod
    def _apply_head_motion(
        frame: np.ndarray,
        asset: CartoonRigAsset,
        params: SquashParams,
        frame_index: int,
        total_frames: int,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        center = asset.geometry.head_center
        phase = 2.0 * math.pi * frame_index / max(1, total_frames)
        sx = 1.0 + (1.0 - params.head_squash_scale) * 0.18
        sy = 1.0 + (params.head_stretch_scale - 1.0) * 0.18
        dy = math.sin(phase) * (1.0 - params.head_squash_scale) * h * 0.018
        matrix = CartoonRigAnimator._affine_about(
            center.x,
            center.y,
            sx=max(0.92, min(1.12, sx)),
            sy=max(0.90, min(1.15, sy)),
            angle_deg=params.head_tilt_deg * 0.35,
            dx=0.0,
            dy=dy,
        )
        return cv2.warpAffine(
            frame,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    @staticmethod
    def _affine_about(
        cx: float,
        cy: float,
        sx: float,
        sy: float,
        angle_deg: float,
        dx: float,
        dy: float,
    ) -> np.ndarray:
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        return np.array(
            [
                [sx * cos_a, -sy * sin_a, cx + dx - sx * cos_a * cx + sy * sin_a * cy],
                [sx * sin_a, sy * cos_a, cy + dy - sx * sin_a * cx - sy * cos_a * cy],
            ],
            dtype=np.float32,
        )
