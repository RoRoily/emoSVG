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
        intensity = CartoonRigAnimator._expression_intensity(params)
        settle = math.sin(phase * 2.0) * intensity
        if layer.name in {"left_eye", "right_eye"}:
            bulge = max(0.0, params.eye_bulge_scale - 1.0)
            squint = params.eye_squint_scale
            side = -1.0 if layer.name == "left_eye" else 1.0
            sx = 1.0 + 0.15 * bulge + 0.025 * settle
            sy = max(0.20, 1.0 + 0.08 * bulge + 0.020 * settle) * max(0.28, squint)
            dy = -params.brow_raise_offset * max(1.0, layer.bbox.h * 0.10)
            dy += math.sin(phase * 1.5 + 0.5) * intensity * layer.bbox.h * 0.010
            dx = side * layer.bbox.w * 0.018 * bulge
            angle = side * params.head_tilt_deg * 0.05 * intensity
            return sx, sy, dx, dy, angle

        if layer.name == "mouth":
            jaw = max(0.0, params.jaw_drop_scale - 1.0)
            sx = 1.0 + (params.mouth_width_scale - 1.0) * 0.55 + 0.018 * settle
            sy = 1.0 + jaw * 0.72 + 0.030 * math.sin(phase * 1.5 + 0.9) * intensity
            dx = params.mouth_corner_offset * layer.bbox.w * 0.06
            dy = jaw * layer.bbox.h * 0.18 + math.sin(phase * 1.2 + 1.1) * intensity * layer.bbox.h * 0.012
            return max(0.45, sx), max(0.35, sy), dx, dy, 0.0

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
        intensity = CartoonRigAnimator._expression_intensity(params)
        sx = 1.0 + (1.0 - params.head_squash_scale) * 0.10
        sy = 1.0 + (params.head_stretch_scale - 1.0) * 0.10
        dy = math.sin(phase) * intensity * h * 0.006
        matrix = CartoonRigAnimator._affine_about(
            center.x,
            center.y,
            sx=max(0.92, min(1.12, sx)),
            sy=max(0.90, min(1.15, sy)),
            angle_deg=params.head_tilt_deg * 0.22,
            dx=0.0,
            dy=dy,
        )
        warped = cv2.warpAffine(
            frame,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        alpha = CartoonRigAnimator._foreground_motion_alpha(asset, w, h)
        return (warped.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)).astype(
            np.uint8
        )

    @staticmethod
    def _expression_intensity(params: SquashParams) -> float:
        values = [
            abs(params.eye_bulge_scale - 1.0) / 2.0,
            abs(params.eye_squint_scale - 1.0),
            abs(params.brow_raise_offset),
            abs(params.jaw_drop_scale - 1.0),
            abs(params.mouth_width_scale - 1.0),
            abs(params.mouth_corner_offset),
            abs(params.head_squash_scale - 1.0),
            abs(params.head_stretch_scale - 1.0),
            abs(params.head_tilt_deg) / 18.0,
        ]
        return max(0.0, min(1.0, max(values)))

    @staticmethod
    def _foreground_motion_alpha(asset: CartoonRigAsset, width: int, height: int) -> np.ndarray:
        box = asset.geometry.foreground_bbox
        pad_x = max(2, int(box.w * 0.03))
        pad_y = max(2, int(box.h * 0.03))
        box = box.pad(pad_x, pad_y).clamp(width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[box.y : box.y2, box.x : box.x2] = 255
        blur = max(9, int(min(width, height) * 0.06) | 1)
        alpha = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
        return alpha[:, :, None]

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
