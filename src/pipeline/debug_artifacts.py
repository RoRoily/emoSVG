"""Debug artifact writer for pipeline diagnosis."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.modules.cartoon_analyzer.schemas import CartoonFaceAnalysis
from src.modules.cartoon_layer_parser.schemas import CartoonLayerParseResult
from src.modules.meme_animator.schemas import KeyFrame


class PipelineDebugWriter:
    """Writes stage-0 diagnostic artifacts into a per-request directory."""

    def __init__(self, output_root: Path, source_stem: str, expression: str) -> None:
        ts = int(time.time())
        safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in source_stem)
        self.debug_dir = output_root / "debug" / f"{safe_stem}_{expression}_{ts}"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: dict[str, Any] = {}

    def write_input(self, image_bgr: np.ndarray) -> None:
        cv2.imwrite(str(self.debug_dir / "input.png"), image_bgr)

    def write_analysis_overlay(
        self,
        image_bgr: np.ndarray,
        analysis: CartoonFaceAnalysis,
    ) -> None:
        overlay = image_bgr.copy()
        g = analysis.geometry
        self._draw_box(overlay, g.foreground_bbox, (160, 160, 160), "foreground")
        self._draw_box(overlay, g.face_bbox, (0, 220, 255), "face")
        self._draw_box(overlay, g.left_eye_bbox, (255, 0, 255), "left_eye")
        self._draw_box(overlay, g.right_eye_bbox, (255, 0, 255), "right_eye")
        self._draw_box(overlay, g.mouth_bbox, (0, 80, 255), "mouth")
        for point, color in [
            (g.left_eye_center, (255, 0, 255)),
            (g.right_eye_center, (255, 0, 255)),
            (g.mouth_center, (0, 80, 255)),
            (g.head_center, (0, 220, 255)),
        ]:
            cv2.circle(overlay, (int(point.x), int(point.y)), 3, color, -1)
        cv2.imwrite(str(self.debug_dir / "landmark_overlay.png"), overlay)

    def write_layer_overlay(
        self,
        image_bgr: np.ndarray,
        layers: CartoonLayerParseResult,
    ) -> None:
        palette = [
            (255, 70, 70),
            (70, 255, 70),
            (70, 70, 255),
            (255, 220, 70),
            (255, 70, 220),
            (70, 255, 220),
            (200, 120, 255),
        ]
        overlay = image_bgr.copy().astype(np.float32)
        for idx, layer in enumerate(layers.layers):
            if layer.name == "foreground":
                continue
            color = np.array(palette[idx % len(palette)], dtype=np.float32)
            alpha = (layer.mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
            overlay = overlay * (1.0 - alpha) + color * alpha
        cv2.imwrite(str(self.debug_dir / "masks_overlay.png"), overlay.astype(np.uint8))

    def write_layers(self, layers: CartoonLayerParseResult) -> None:
        layers.save_layers(self.debug_dir / "layers")

    def write_keyframe_contact_sheet(
        self,
        keyframes: list[KeyFrame],
        max_frames: int = 30,
        thumb_size: tuple[int, int] = (128, 128),
    ) -> None:
        frames = [kf.image for kf in keyframes[:max_frames] if isinstance(kf.image, np.ndarray)]
        if not frames:
            return
        tw, th = thumb_size
        cols = min(6, len(frames))
        rows = int(np.ceil(len(frames) / cols))
        sheet = np.full((rows * th, cols * tw, 3), 245, dtype=np.uint8)
        for idx, frame in enumerate(frames):
            thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
            r, c = divmod(idx, cols)
            sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = thumb
            cv2.putText(
                sheet,
                str(keyframes[idx].index),
                (c * tw + 6, r * th + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(self.debug_dir / "keyframe_contact_sheet.jpg"), sheet)

    def update_metrics(self, **kwargs: Any) -> None:
        self._metrics.update(kwargs)

    def write_metrics(self) -> None:
        with open(self.debug_dir / "metrics.json", "w", encoding="utf-8") as fh:
            json.dump(self._metrics, fh, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _draw_box(image: np.ndarray, box, color: tuple[int, int, int], label: str) -> None:
        cv2.rectangle(image, (box.x, box.y), (box.x2, box.y2), color, 2)
        cv2.putText(
            image,
            label,
            (box.x, max(12, box.y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
