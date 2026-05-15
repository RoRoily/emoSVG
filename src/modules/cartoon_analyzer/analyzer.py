"""Q-style cartoon face geometry analysis.

The public CartoonFaceAnalyzer prefers a trained anime landmark detector when
available, while keeping the deterministic heuristic analyzer as a fallback for
CPU-only tests and environments where OpenMMLab dependencies are not installed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .schemas import BBox, CartoonFaceAnalysis, CartoonFaceGeometry

logger = logging.getLogger(__name__)


class CartoonFaceAnalyzer:
    """Detect Q-style face, eye, and mouth geometry from a still image.

    Backends:
      - auto: prefer hysts/anime-face-detector, then heuristic fallback.
      - anime_face_detector: require the trained 28-landmark anime detector
        unless fallback is explicitly enabled.
      - heuristic: use deterministic color/geometry rules only.
    """

    BACKEND_AUTO = "auto"
    BACKEND_ANIME_FACE_DETECTOR = "anime_face_detector"
    BACKEND_HEURISTIC = "heuristic"

    def __init__(
        self,
        backend: str | None = None,
        device: str | None = None,
        detector_name: str | None = None,
        allow_fallback: bool | None = None,
    ) -> None:
        self.backend = (backend or os.getenv("CARTOON_ANALYZER_BACKEND") or self.BACKEND_AUTO).lower()
        self.device = device or os.getenv("CARTOON_ANALYZER_DEVICE") or "cuda:0"
        self.detector_name = (
            detector_name
            or os.getenv("CARTOON_ANALYZER_MODEL")
            or "yolov3"
        )
        if allow_fallback is None:
            allow_fallback = os.getenv("CARTOON_ANALYZER_ALLOW_HEURISTIC_FALLBACK", "1") != "0"
        self.allow_fallback = allow_fallback
        self._heuristic = HeuristicCartoonFaceAnalyzer()
        self._trained_backend = None
        self._trained_backend_error: str | None = None

    def analyze(
        self,
        source_image_path: Path,
        resolution: tuple[int, int] | None = None,
    ) -> CartoonFaceAnalysis:
        bgr = self._load_bgr(source_image_path, resolution)
        return self.analyze_bgr(bgr)

    def analyze_bgr(self, image_bgr: np.ndarray) -> CartoonFaceAnalysis:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("CartoonFaceAnalyzer expects an HxWx3 BGR image")

        if self.backend == self.BACKEND_HEURISTIC:
            return self._heuristic.analyze_bgr(image_bgr)

        if self.backend not in {
            self.BACKEND_AUTO,
            self.BACKEND_ANIME_FACE_DETECTOR,
        }:
            raise ValueError(
                "CARTOON_ANALYZER_BACKEND must be one of: "
                "auto, anime_face_detector, heuristic"
            )

        try:
            backend = self._get_trained_backend()
            if backend is None:
                raise RuntimeError(self._trained_backend_error or "trained backend unavailable")
            foreground_mask = self._foreground_mask(image_bgr)
            h, w = image_bgr.shape[:2]
            foreground_bbox = self._mask_bbox(foreground_mask) or BBox(0, 0, w, h)
            return backend.analyze_bgr(
                image_bgr,
                foreground_mask=foreground_mask,
                foreground_bbox=foreground_bbox,
            )
        except Exception as exc:
            if self.backend == self.BACKEND_ANIME_FACE_DETECTOR and not self.allow_fallback:
                raise
            logger.warning(
                "Trained cartoon analyzer unavailable or failed (%s); using heuristic fallback.",
                exc,
            )
            result = self._heuristic.analyze_bgr(image_bgr)
            result.backend_used = "anime_face_detector_unavailable+heuristic_fallback"
            result.geometry.warnings.append(f"trained analyzer fallback: {exc}")
            return result

    def _get_trained_backend(self):
        if self._trained_backend is not None:
            return self._trained_backend
        if self._trained_backend_error is not None:
            return None
        try:
            from .anime_face_detector_adapter import AnimeFaceDetectorAdapter
            self._trained_backend = AnimeFaceDetectorAdapter(
                detector_name=self.detector_name,
                device=self.device,
            )
            logger.info(
                "Using trained cartoon analyzer backend: anime_face_detector(%s) on %s",
                self.detector_name,
                self.device,
            )
        except Exception as exc:
            self._trained_backend_error = str(exc)
            logger.info("anime_face_detector backend is not available: %s", exc)
        return self._trained_backend

    @staticmethod
    def _load_bgr(path: Path, resolution: tuple[int, int] | None) -> np.ndarray:
        return HeuristicCartoonFaceAnalyzer._load_bgr(path, resolution)

    @staticmethod
    def _foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
        return HeuristicCartoonFaceAnalyzer._foreground_mask(image_bgr)

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> BBox | None:
        return HeuristicCartoonFaceAnalyzer._mask_bbox(mask)


class HeuristicCartoonFaceAnalyzer:
    """Deterministic fallback for Q-style face, eye, and mouth geometry."""

    def analyze(
        self,
        source_image_path: Path,
        resolution: tuple[int, int] | None = None,
    ) -> CartoonFaceAnalysis:
        bgr = self._load_bgr(source_image_path, resolution)
        return self.analyze_bgr(bgr)

    def analyze_bgr(self, image_bgr: np.ndarray) -> CartoonFaceAnalysis:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("CartoonFaceAnalyzer expects an HxWx3 BGR image")

        h, w = image_bgr.shape[:2]
        foreground_mask = self._foreground_mask(image_bgr)
        foreground_bbox = self._mask_bbox(foreground_mask) or BBox(0, 0, w, h)
        face_bbox = self._estimate_face_bbox(foreground_bbox, w, h)

        warnings: list[str] = []
        eye_candidates = self._find_eye_candidates(image_bgr, face_bbox)
        if len(eye_candidates) >= 2:
            left_eye, right_eye = self._choose_eye_pair(eye_candidates)
            left_eye = self._expand_eye_bbox(left_eye, face_bbox, w, h)
            right_eye = self._expand_eye_bbox(right_eye, face_bbox, w, h)
            eye_conf = 0.85
        else:
            left_eye, right_eye = self._fallback_eyes(face_bbox)
            eye_conf = 0.35
            warnings.append("eye candidates were weak; used geometric fallback")

        mouth_candidates = self._find_mouth_candidates(image_bgr, face_bbox, left_eye, right_eye)
        if mouth_candidates:
            mouth = max(mouth_candidates, key=lambda item: item[1])[0]
            mouth_conf = 0.8
        else:
            mouth = self._fallback_mouth(face_bbox, left_eye, right_eye)
            mouth_conf = 0.35
            warnings.append("mouth candidates were weak; used geometric fallback")

        confidence = min(0.95, 0.15 + 0.5 * eye_conf + 0.35 * mouth_conf)
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
            "Cartoon geometry detected: confidence=%.2f eyes=%s/%s mouth=%s",
            confidence,
            left_eye.to_dict(),
            right_eye.to_dict(),
            mouth.to_dict(),
        )
        return CartoonFaceAnalysis(
            image_size=(w, h),
            geometry=geometry,
            foreground_mask=foreground_mask,
            backend_used="heuristic_cartoon_geometry",
        )

    @staticmethod
    def _load_bgr(path: Path, resolution: tuple[int, int] | None) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            bgr = img[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
            bgr = bgr.astype(np.uint8)
        else:
            bgr = img[:, :, :3]
        if resolution is not None:
            width, height = resolution
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
        return bgr

    @staticmethod
    def _foreground_mask(image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        corners = np.concatenate(
            [
                image_bgr[: max(1, h // 20), : max(1, w // 20)].reshape(-1, 3),
                image_bgr[: max(1, h // 20), -max(1, w // 20) :].reshape(-1, 3),
                image_bgr[-max(1, h // 20) :, : max(1, w // 20)].reshape(-1, 3),
                image_bgr[-max(1, h // 20) :, -max(1, w // 20) :].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(corners, axis=0)
        diff = np.linalg.norm(image_bgr.astype(np.float32) - bg.astype(np.float32), axis=2)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        mask = ((diff > 18.0) | (gray < 245)).astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    @staticmethod
    def _mask_bbox(mask: np.ndarray) -> BBox | None:
        points = cv2.findNonZero(mask)
        if points is None:
            return None
        x, y, w, h = cv2.boundingRect(points)
        return BBox(x, y, w, h)

    @staticmethod
    def _estimate_face_bbox(fg: BBox, width: int, height: int) -> BBox:
        # Q-style portraits are usually head-dominant. Keep most of the upper
        # foreground and leave lower body/empty margins out when possible.
        pad_x = int(fg.w * 0.04)
        pad_top = int(fg.h * 0.03)
        face_h = int(fg.h * 0.86)
        return BBox(
            fg.x - pad_x,
            fg.y - pad_top,
            fg.w + 2 * pad_x,
            max(1, face_h + pad_top),
        ).clamp(width, height)

    @staticmethod
    def _component_bboxes(mask: np.ndarray, min_area: int) -> list[tuple[BBox, int]]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes: list[tuple[BBox, int]] = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area >= min_area:
                boxes.append((BBox(int(x), int(y), int(w), int(h)), int(area)))
        return boxes

    def _find_eye_candidates(self, image_bgr: np.ndarray, face: BBox) -> list[tuple[BBox, float]]:
        roi = image_bgr[face.y : face.y2, face.x : face.x2]
        if roi.size == 0:
            return []

        roi_h, roi_w = roi.shape[:2]
        upper_limit = int(roi_h * 0.72)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Colored pupils/irises are the strongest anime cue. The dark mask catches
        # black eyes while the y-window suppresses hair and mouth outlines.
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        colored_eye_hue = ((hue >= 70) & (hue <= 175)) | (sat > 80)
        sat_mask = (sat > 45) & (val > 45) & (val < 252) & colored_eye_hue
        dark_mask = gray < 88
        y_mask = np.zeros((roi_h, roi_w), dtype=bool)
        y_mask[int(roi_h * 0.18) : upper_limit, :] = True

        def collect(mask_bool: np.ndarray, score_scale: float) -> list[tuple[BBox, float]]:
            mask = (mask_bool & y_mask).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            candidates: list[tuple[BBox, float]] = []
            min_area = max(8, int(face.area * 0.00045))
            max_area = max(min_area + 1, int(face.area * 0.08))
            max_w = max(8, int(face.w * 0.28))
            max_h = max(8, int(face.h * 0.30))
            for box, area in self._component_bboxes(mask, min_area=min_area):
                aspect = box.w / max(1, box.h)
                if not (0.35 <= aspect <= 2.8):
                    continue
                if area > max_area or box.w > max_w or box.h > max_h:
                    continue
                if box.w < max(4, int(face.w * 0.025)) or box.h < max(4, int(face.h * 0.025)):
                    continue
                global_box = BBox(face.x + box.x, face.y + box.y, box.w, box.h)
                center = global_box.center
                horizontal_center_bias = 1.0 - min(
                    0.8,
                    abs(center.x - face.center.x) / max(1, face.w),
                )
                score = area * score_scale * (0.65 + 0.35 * horizontal_center_bias)
                candidates.append((global_box, float(score)))
            return candidates

        color_candidates = collect(sat_mask, score_scale=1.25)
        if len(color_candidates) >= 2:
            return sorted(color_candidates, key=lambda item: item[1], reverse=True)[:12]

        dark_candidates = collect(dark_mask, score_scale=0.9)
        return sorted(color_candidates + dark_candidates, key=lambda item: item[1], reverse=True)[:12]

    @staticmethod
    def _choose_eye_pair(candidates: list[tuple[BBox, float]]) -> tuple[BBox, BBox]:
        best_pair: tuple[BBox, BBox] | None = None
        best_score = -1.0
        for i, (a, score_a) in enumerate(candidates):
            for b, score_b in candidates[i + 1 :]:
                if abs(a.center.x - b.center.x) < max(a.w, b.w):
                    continue
                y_delta = abs(a.center.y - b.center.y)
                size_delta = abs(a.area - b.area) / max(1, max(a.area, b.area))
                pair_score = score_a + score_b - 6.0 * y_delta - 300.0 * size_delta
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (a, b)
        if best_pair is None:
            ordered = sorted([box for box, _ in candidates[:2]], key=lambda box: box.center.x)
            return ordered[0], ordered[1]
        ordered = sorted(best_pair, key=lambda box: box.center.x)
        return ordered[0], ordered[1]

    @staticmethod
    def _fallback_eyes(face: BBox) -> tuple[BBox, BBox]:
        eye_w = max(6, int(face.w * 0.18))
        eye_h = max(6, int(face.h * 0.15))
        y = int(face.y + face.h * 0.39)
        left_x = int(face.x + face.w * 0.31 - eye_w / 2)
        right_x = int(face.x + face.w * 0.69 - eye_w / 2)
        return BBox(left_x, y, eye_w, eye_h), BBox(right_x, y, eye_w, eye_h)

    @staticmethod
    def _expand_eye_bbox(box: BBox, face: BBox, image_w: int, image_h: int) -> BBox:
        target_w = max(int(box.w * 1.45), int(face.w * 0.14))
        target_h = max(int(box.h * 2.05), int(face.h * 0.18))
        x = int(box.center.x - target_w * 0.50)
        # Colored iris blobs often cover only the lower half of an anime eye.
        y = int(box.center.y - target_h * 0.85)
        return BBox(x, y, target_w, target_h).clamp(image_w, image_h)

    def _find_mouth_candidates(
        self,
        image_bgr: np.ndarray,
        face: BBox,
        left_eye: BBox,
        right_eye: BBox,
    ) -> list[tuple[BBox, float]]:
        roi = image_bgr[face.y : face.y2, face.x : face.x2]
        if roi.size == 0:
            return []
        roi_h, roi_w = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        eye_y = (left_eye.center.y + right_eye.center.y) / 2.0 - face.y
        y1 = int(max(eye_y + roi_h * 0.10, roi_h * 0.45))
        y2 = int(min(roi_h * 0.82, y1 + roi_h * 0.28))
        x1 = int(roi_w * 0.25)
        x2 = int(roi_w * 0.75)

        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        red_or_orange = ((hue < 16) | (hue > 165) | ((hue >= 16) & (hue < 32)))
        color_mask = red_or_orange & (sat > 35) & (val > 70)
        dark_mask = (gray < 120) & (sat > 15)
        window = np.zeros((roi_h, roi_w), dtype=bool)
        window[y1:y2, x1:x2] = True
        mask = ((color_mask | dark_mask) & window).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        candidates: list[tuple[BBox, float]] = []
        min_area = max(5, int(face.area * 0.0004))
        max_area = max(min_area + 1, int(face.area * 0.035))
        for box, area in self._component_bboxes(mask, min_area=min_area):
            aspect = box.w / max(1, box.h)
            if not (0.45 <= aspect <= 3.2):
                continue
            if area > max_area:
                continue
            if box.w < max(8, int(face.w * 0.04)) or box.h < max(5, int(face.h * 0.025)):
                continue
            global_box = BBox(face.x + box.x, face.y + box.y, box.w, box.h)
            if abs(global_box.center.x - face.center.x) > face.w * 0.14:
                continue
            center_bias = 1.0 - min(0.9, abs(global_box.center.x - face.center.x) / max(1, face.w / 2))
            score = area * (0.4 + 0.6 * center_bias)
            candidates.append((global_box, float(score)))
        return sorted(candidates, key=lambda item: item[1], reverse=True)[:6]

    @staticmethod
    def _fallback_mouth(face: BBox, left_eye: BBox, right_eye: BBox) -> BBox:
        mouth_w = max(5, int(face.w * 0.12))
        mouth_h = max(4, int(face.h * 0.08))
        eye_mid_x = (left_eye.center.x + right_eye.center.x) / 2.0
        y = int(face.y + face.h * 0.63)
        x = int(eye_mid_x - mouth_w / 2)
        return BBox(x, y, mouth_w, mouth_h)
