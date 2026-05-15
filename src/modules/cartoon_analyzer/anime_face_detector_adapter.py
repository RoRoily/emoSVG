"""Adapter for hysts/anime-face-detector 28-point anime landmarks."""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from .schemas import BBox, CartoonFaceAnalysis, CartoonFaceGeometry, Point2D

logger = logging.getLogger(__name__)


class AnimeFaceDetectorAdapter:
    """Convert anime-face-detector predictions to emoSVG cartoon geometry.

    The upstream model predicts one face bbox and 28 landmarks. The commonly
    used landmark groups are:
      - 11..16: left eye
      - 17..22: right eye
      - 23..27: mouth
    We still sort eye boxes by x coordinate so mirrored/order variants are
    handled safely.
    """

    _LEFT_EYE_IDX = tuple(range(11, 17))
    _RIGHT_EYE_IDX = tuple(range(17, 23))
    _MOUTH_IDX = tuple(range(23, 28))

    def __init__(
        self,
        detector_name: str = "yolov3",
        device: str = "cuda:0",
        score_threshold: float | None = None,
        keypoint_threshold: float | None = None,
    ) -> None:
        self.detector_name = detector_name
        self.device = device
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else float(os.getenv("CARTOON_ANALYZER_FACE_SCORE_THRESHOLD", "0.45"))
        )
        self.keypoint_threshold = (
            keypoint_threshold
            if keypoint_threshold is not None
            else float(os.getenv("CARTOON_ANALYZER_KEYPOINT_SCORE_THRESHOLD", "0.20"))
        )
        self._detector = self._create_detector()

    def analyze_bgr(
        self,
        image_bgr: np.ndarray,
        foreground_mask: np.ndarray,
        foreground_bbox: BBox,
    ) -> CartoonFaceAnalysis:
        h, w = image_bgr.shape[:2]
        predictions = list(self._detector(image_bgr))
        if len(predictions) == 0:
            raise RuntimeError("anime-face-detector returned no faces")

        pred = self._select_prediction(predictions, w, h)
        face_bbox = self._prediction_bbox(pred, w, h)
        keypoints = self._prediction_keypoints(pred)

        if keypoints.shape[0] < 28:
            raise RuntimeError(f"expected at least 28 anime landmarks, got {keypoints.shape[0]}")

        left_eye = self._bbox_from_keypoints(
            keypoints,
            self._LEFT_EYE_IDX,
            face_bbox,
            w,
            h,
            pad_x=0.22,
            pad_y=0.30,
            min_w=0.12,
            min_h=0.10,
        )
        right_eye = self._bbox_from_keypoints(
            keypoints,
            self._RIGHT_EYE_IDX,
            face_bbox,
            w,
            h,
            pad_x=0.22,
            pad_y=0.30,
            min_w=0.12,
            min_h=0.10,
        )
        left_eye, right_eye = sorted([left_eye, right_eye], key=lambda box: box.center.x)

        mouth = self._bbox_from_keypoints(
            keypoints,
            self._MOUTH_IDX,
            face_bbox,
            w,
            h,
            pad_x=0.35,
            pad_y=0.50,
            min_w=0.08,
            min_h=0.06,
        )

        face_score = float(self._prediction_score(pred))
        landmark_score = self._mean_score(
            keypoints,
            self._LEFT_EYE_IDX + self._RIGHT_EYE_IDX + self._MOUTH_IDX,
        )
        confidence = max(0.0, min(0.98, 0.25 + 0.45 * face_score + 0.30 * landmark_score))
        warnings = []
        if face_score < self.score_threshold:
            warnings.append(f"anime face score is low: {face_score:.2f}")
        if landmark_score < self.keypoint_threshold:
            warnings.append(f"anime landmark score is low: {landmark_score:.2f}")

        geometry = CartoonFaceGeometry(
            face_bbox=face_bbox,
            foreground_bbox=foreground_bbox,
            left_eye_bbox=left_eye,
            right_eye_bbox=right_eye,
            mouth_bbox=mouth,
            left_eye_center=left_eye.center,
            right_eye_center=right_eye.center,
            mouth_center=mouth.center,
            head_center=face_bbox.center,
            confidence=confidence,
            warnings=warnings,
        )
        logger.info(
            "Anime landmark geometry detected: confidence=%.2f face=%s eyes=%s/%s mouth=%s",
            confidence,
            face_bbox.to_dict(),
            left_eye.to_dict(),
            right_eye.to_dict(),
            mouth.to_dict(),
        )
        return CartoonFaceAnalysis(
            image_size=(w, h),
            geometry=geometry,
            foreground_mask=foreground_mask,
            backend_used="anime_face_detector",
        )

    def _create_detector(self):
        try:
            from anime_face_detector import create_detector
        except Exception as exc:
            raise RuntimeError(
                "anime-face-detector is not installed. Install it in the current "
                "environment or set CARTOON_ANALYZER_BACKEND=heuristic."
            ) from exc
        return create_detector(self.detector_name, device=self.device)

    def _select_prediction(self, predictions: list[Any], width: int, height: int) -> Any:
        valid = []
        for pred in predictions:
            try:
                box = self._prediction_bbox(pred, width, height)
                score = self._prediction_score(pred)
            except Exception:
                continue
            if score >= self.score_threshold:
                valid.append((pred, score, box.area))
        candidates = valid or [
            (pred, self._prediction_score(pred), self._prediction_bbox(pred, width, height).area)
            for pred in predictions
        ]
        return max(candidates, key=lambda item: (item[1], item[2]))[0]

    @staticmethod
    def _prediction_score(pred: Any) -> float:
        if isinstance(pred, dict):
            bbox = pred.get("bbox")
            if bbox is not None and len(bbox) >= 5:
                return float(bbox[4])
            if "score" in pred:
                return float(pred["score"])
        if hasattr(pred, "get"):
            score = pred.get("score", None)
            if score is not None:
                return float(score)
        return 1.0

    @staticmethod
    def _prediction_bbox(pred: Any, width: int, height: int) -> BBox:
        bbox = pred["bbox"] if isinstance(pred, dict) else pred.bbox
        arr = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if arr.size < 4:
            raise RuntimeError("anime-face-detector bbox has fewer than 4 values")
        x1, y1, x2, y2 = arr[:4]
        return BBox(
            int(round(float(x1))),
            int(round(float(y1))),
            int(round(float(x2 - x1))),
            int(round(float(y2 - y1))),
        ).clamp(width, height)

    @staticmethod
    def _prediction_keypoints(pred: Any) -> np.ndarray:
        if isinstance(pred, dict):
            data = pred.get("keypoints")
        else:
            data = pred.keypoints
        keypoints = np.asarray(data, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            raise RuntimeError(f"unexpected anime landmark shape: {keypoints.shape}")
        if keypoints.shape[1] == 2:
            scores = np.ones((keypoints.shape[0], 1), dtype=np.float32)
            keypoints = np.concatenate([keypoints, scores], axis=1)
        return keypoints[:, :3]

    def _bbox_from_keypoints(
        self,
        keypoints: np.ndarray,
        indices: tuple[int, ...],
        face: BBox,
        image_w: int,
        image_h: int,
        pad_x: float,
        pad_y: float,
        min_w: float,
        min_h: float,
    ) -> BBox:
        points = keypoints[list(indices)]
        confident = points[points[:, 2] >= self.keypoint_threshold]
        if len(confident) < max(2, len(indices) // 3):
            confident = points
        xs = confident[:, 0]
        ys = confident[:, 1]
        x1 = float(np.min(xs))
        y1 = float(np.min(ys))
        x2 = float(np.max(xs))
        y2 = float(np.max(ys))
        raw_w = max(1.0, x2 - x1)
        raw_h = max(1.0, y2 - y1)
        target_w = max(raw_w * (1.0 + pad_x * 2.0), face.w * min_w)
        target_h = max(raw_h * (1.0 + pad_y * 2.0), face.h * min_h)
        center = Point2D(float(np.mean(xs)), float(np.mean(ys)))
        return BBox(
            int(round(center.x - target_w / 2.0)),
            int(round(center.y - target_h / 2.0)),
            int(round(target_w)),
            int(round(target_h)),
        ).clamp(image_w, image_h)

    @staticmethod
    def _mean_score(keypoints: np.ndarray, indices: tuple[int, ...]) -> float:
        scores = keypoints[list(indices), 2]
        return float(np.mean(scores)) if len(scores) else 0.0
