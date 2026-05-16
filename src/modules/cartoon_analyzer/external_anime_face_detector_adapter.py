"""Subprocess adapter for a separate animeFaceDetector environment."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .anime_face_detector_adapter import AnimeFaceDetectorAdapter
from .schemas import BBox, CartoonFaceAnalysis

logger = logging.getLogger(__name__)


class ExternalAnimeFaceDetectorAdapter:
    """Call scripts/anime_face_detect.py in another Python environment."""

    def __init__(
        self,
        python_executable: str | Path | None = None,
        script_path: str | Path | None = None,
        detector_name: str = "yolov3",
        device: str = "cpu",
        timeout_seconds: float | None = None,
    ) -> None:
        self.python_executable = Path(
            python_executable
            or os.getenv("CARTOON_ANALYZER_EXTERNAL_PYTHON", "")
        )
        if not str(self.python_executable):
            raise RuntimeError(
                "CARTOON_ANALYZER_EXTERNAL_PYTHON is not set. Point it to "
                "animeFaceDetector/bin/python."
            )
        if not self.python_executable.exists():
            raise RuntimeError(f"External analyzer Python does not exist: {self.python_executable}")

        default_script = Path(__file__).resolve().parents[3] / "scripts" / "anime_face_detect.py"
        self.script_path = Path(
            script_path
            or os.getenv("CARTOON_ANALYZER_EXTERNAL_SCRIPT", "")
            or default_script
        )
        if not self.script_path.exists():
            raise RuntimeError(f"External analyzer script does not exist: {self.script_path}")

        self.detector_name = detector_name
        self.device = device
        self.timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("CARTOON_ANALYZER_EXTERNAL_TIMEOUT", "180"))
        )
        self._mapper = _ExternalAnimeLandmarkMapper()

    def analyze_bgr(
        self,
        image_bgr: np.ndarray,
        foreground_mask: np.ndarray,
        foreground_bbox: BBox,
    ) -> CartoonFaceAnalysis:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            cv2.imwrite(str(tmp_path), image_bgr)
            payload = self._run_detector(tmp_path)
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass

        faces = payload.get("faces", [])
        if not faces:
            raise RuntimeError("external anime-face-detector returned no faces")
        h, w = image_bgr.shape[:2]
        analysis = self._mapper.predictions_to_analysis(
            faces,
            image_size=(w, h),
            foreground_mask=foreground_mask,
            foreground_bbox=foreground_bbox,
            backend_used="external_anime_face_detector",
        )
        logger.info(
            "External anime landmark geometry detected: confidence=%.2f",
            analysis.geometry.confidence,
        )
        return analysis

    def _run_detector(self, image_path: Path) -> dict[str, Any]:
        cmd = [
            str(self.python_executable),
            str(self.script_path),
            str(image_path),
            "--detector",
            self.detector_name,
            "--device",
            self.device,
        ]
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "external anime-face-detector failed "
                f"(code={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
            )
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"external detector returned invalid JSON: {proc.stdout[:500]}") from exc


class _ExternalAnimeLandmarkMapper(AnimeFaceDetectorAdapter):
    """Reuse the direct adapter's prediction-to-geometry helper methods."""

    def __init__(self) -> None:
        self.detector_name = "external"
        self.device = "external"
        self.score_threshold = float(os.getenv("CARTOON_ANALYZER_FACE_SCORE_THRESHOLD", "0.45"))
        self.keypoint_threshold = float(os.getenv("CARTOON_ANALYZER_KEYPOINT_SCORE_THRESHOLD", "0.20"))

    def _create_detector(self):  # pragma: no cover - never called
        raise NotImplementedError

    def predictions_to_analysis(
        self,
        predictions: list[Any],
        image_size: tuple[int, int],
        foreground_mask: np.ndarray,
        foreground_bbox: BBox,
        backend_used: str,
    ) -> CartoonFaceAnalysis:
        w, h = image_size
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

        from .schemas import CartoonFaceGeometry

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
        return CartoonFaceAnalysis(
            image_size=(w, h),
            geometry=geometry,
            foreground_mask=foreground_mask,
            backend_used=backend_used,
        )
