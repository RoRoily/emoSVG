"""Lightweight animation quality metrics for debug and regression tests."""
from __future__ import annotations

import cv2
import numpy as np


def evaluate_animation_frames(
    frames: list[np.ndarray],
    source_bgr: np.ndarray | None = None,
) -> dict[str, float]:
    """Return cheap, model-free metrics for animation regression tracking."""
    if not frames:
        return {"frame_count": 0.0}

    frames = [_match_size(_ensure_bgr(frame), _ensure_bgr(frames[0])) for frame in frames]
    metrics: dict[str, float] = {"frame_count": float(len(frames))}
    if len(frames) >= 2:
        diffs = [
            float(np.mean(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))))
            / 255.0
            for i in range(1, len(frames))
        ]
        metrics["temporal_mean_absdiff"] = float(np.mean(diffs))
        metrics["temporal_max_absdiff"] = float(np.max(diffs))
        if len(diffs) >= 2:
            jitter = [abs(diffs[i] - diffs[i - 1]) for i in range(1, len(diffs))]
            metrics["temporal_jitter_proxy"] = float(np.mean(jitter))
        else:
            metrics["temporal_jitter_proxy"] = 0.0
    else:
        metrics["temporal_mean_absdiff"] = 0.0
        metrics["temporal_max_absdiff"] = 0.0
        metrics["temporal_jitter_proxy"] = 0.0

    ref = _ensure_bgr(source_bgr) if source_bgr is not None else frames[0]
    ref = _match_size(ref, frames[0])
    similarities = []
    for frame in frames:
        matched = _match_size(frame, ref)
        dist = float(np.mean(np.abs(matched.astype(np.float32) - ref.astype(np.float32)))) / 255.0
        similarities.append(max(0.0, 1.0 - dist))
    metrics["source_similarity_mean"] = float(np.mean(similarities))
    metrics["source_similarity_min"] = float(np.min(similarities))
    metrics["motion_energy"] = metrics["temporal_mean_absdiff"] * len(frames)
    return metrics


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"Unsupported image shape for metrics: {image.shape}")


def _match_size(image: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if image.shape[:2] == ref.shape[:2]:
        return image
    h, w = ref.shape[:2]
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
