"""
Image preprocessor for IP feature extraction.

Responsibilities:
- Load image from disk (BGR -> RGB).
- Resize to the target encoder input size.
- Normalise pixel values to the CLIP standard:
    mean = [0.48145466, 0.4578275, 0.40821073]
    std  = [0.26862954, 0.26130258, 0.27577711]
- Return a float32 numpy array ready for the encoder.
"""
from __future__ import annotations

import cv2
import numpy as np

# CLIP normalisation constants
_CLIP_MEAN = np.array([0.48145466, 0.4578275,  0.40821073], dtype=np.float32)
_CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


class ImagePreprocessor:
    """Prepares a source image for CLIP / IP-Adapter encoding."""

    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size  # (width, height)

    def process(self, image_path_or_array, normalize: bool = True) -> np.ndarray:
        """
        Parameters
        ----------
        image_path_or_array:
            Path to an image file, or an existing HxWx3 uint8 BGR numpy array.
        normalize:
            If True, apply CLIP mean/std normalisation.

        Returns
        -------
        float32 numpy array of shape (H, W, 3) in RGB order, values in [0,1]
        (or normalised if normalize=True).
        """
        img = self._load(image_path_or_array)
        img = self._resize(img)
        img = self._to_float_rgb(img)
        if normalize:
            img = self._normalize(img)
        return img

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _load(source) -> np.ndarray:
        from pathlib import Path
        if isinstance(source, (str, Path)):
            arr = cv2.imread(str(source))
            if arr is None:
                raise ValueError(f"Failed to read image: {source}")
            return arr  # BGR uint8
        if isinstance(source, np.ndarray):
            return source.copy()
        raise TypeError(f"Expected path or ndarray, got {type(source)}")

    def _resize(self, img: np.ndarray) -> np.ndarray:
        w, h = self.target_size
        if img.shape[1] == w and img.shape[0] == h:
            return img
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _to_float_rgb(img: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb.astype(np.float32) / 255.0

    @staticmethod
    def _normalize(img: np.ndarray) -> np.ndarray:
        return (img - _CLIP_MEAN) / _CLIP_STD
