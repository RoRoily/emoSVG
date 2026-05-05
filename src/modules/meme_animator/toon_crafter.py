"""
ToonCrafter wrapper for the meme_animator module.

ToonCrafter is a video interpolation model for cartoon/anime content.
In the emoSVG pipeline it serves two roles:

1. Frame interpolation: given a wind-up frame and a peak frame produced by
   LivePortrait, ToonCrafter generates smooth in-between frames that respect
   cartoon-style motion.

2. Standalone animation: when LivePortrait is unavailable, ToonCrafter can
   generate a short animation from a source image and a target expression
   image (requires two reference frames).

Architecture note:
    ToonCrafter is NOT a drop-in replacement for LivePortrait.  It operates on
    pairs of frames (start, end) and fills the gap, whereas LivePortrait drives
    motion from a single source image + coefficient vector.  The two backends
    are complementary:

        LivePortrait  ->  per-frame rendering   (primary)
        ToonCrafter   ->  inter-frame smoothing  (secondary / standalone)

When ToonCrafter weights are absent the wrapper falls back to linear blending
so the rest of the pipeline can be tested without GPU or model downloads.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import AnimationError

logger = logging.getLogger(__name__)

_TOON_CRAFTER_VRAM_GB = 8.0


class ToonCrafterWrapper:
    """
    Thin adapter around ToonCrafter for cartoon-aware frame interpolation.

    Parameters
    ----------
    model_path:
        Directory containing ToonCrafter checkpoint files (model.ckpt + config.yaml).
        If None or the path does not exist, falls back to linear-blend mode.
    num_frames:
        Default number of frames to generate between each (start, end) pair.
    registry:
        ModelRegistry instance to use.
    """

    MODEL_ID = "toon_crafter"

    def __init__(
        self,
        model_path: Path | None = None,
        num_frames: int = 8,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._model_path = Path(model_path) if model_path else None
        self._num_frames = num_frames
        self._registry = registry or ModelRegistry.instance()
        self._use_fallback = self._should_use_fallback()

        if not self._use_fallback:
            self._register_model()
        else:
            logger.warning(
                "ToonCrafter weights not found at %r — using linear-blend fallback.",
                str(self._model_path),
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def interpolate(
        self,
        frame_start: np.ndarray,
        frame_end: np.ndarray,
        num_frames: int | None = None,
    ) -> list[np.ndarray]:
        """
        Generate smooth in-between frames from frame_start to frame_end.

        Parameters
        ----------
        frame_start:  HxWx3 uint8 BGR image (first keyframe).
        frame_end:    HxWx3 uint8 BGR image (last keyframe).
        num_frames:   Override the instance-level num_frames for this call.

        Returns
        -------
        List of HxWx3 uint8 BGR frames including start and end:
            [frame_start, interp_1, ..., interp_N, frame_end]
        """
        n = num_frames if num_frames is not None else self._num_frames

        if self._use_fallback:
            return self._linear_blend_fallback(frame_start, frame_end, n)

        with self._registry.model_context(self.MODEL_ID, offload_after=True) as model:
            return self._toon_crafter_infer(model, frame_start, frame_end, n)

    def smooth_sequence(
        self,
        keyframes: list[np.ndarray],
        frames_between: int = 4,
    ) -> list[np.ndarray]:
        """
        Interpolate between every consecutive pair in a keyframe list.

        Parameters
        ----------
        keyframes:       List of HxWx3 uint8 BGR keyframes.
        frames_between:  Number of interpolated frames to insert between each pair.

        Returns
        -------
        Full smoothed sequence including all original keyframes.
        """
        if len(keyframes) < 2:
            return list(keyframes)

        result: list[np.ndarray] = []
        for i in range(len(keyframes) - 1):
            segment = self.interpolate(keyframes[i], keyframes[i + 1], frames_between)
            result.extend(segment[:-1])  # avoid duplicating the shared boundary frame
        result.append(keyframes[-1])
        return result

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _should_use_fallback(self) -> bool:
        if self._model_path is None:
            return True
        required = ["model.ckpt", "config.yaml"]
        return not all((self._model_path / f).exists() for f in required)

    def _register_model(self) -> None:
        model_path = str(self._model_path)

        def _loader():
            try:
                from tooncrafter.inference import ToonCrafterInference  # type: ignore
                return ToonCrafterInference(model_path)
            except ImportError as exc:
                raise AnimationError(
                    "ToonCrafter package not installed. "
                    "Run: pip install git+https://github.com/ToonCrafter/ToonCrafter.git"
                ) from exc
            except Exception as exc:
                raise AnimationError(
                    f"Failed to load ToonCrafter from {model_path!r}: {exc}"
                ) from exc

        self._registry.register(
            self.MODEL_ID, _loader, estimated_vram_gb=_TOON_CRAFTER_VRAM_GB
        )

    def _toon_crafter_infer(
        self,
        model,
        frame_start: np.ndarray,
        frame_end: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        start_rgb = cv2.cvtColor(frame_start, cv2.COLOR_BGR2RGB)
        end_rgb   = cv2.cvtColor(frame_end,   cv2.COLOR_BGR2RGB)
        try:
            frames_rgb = model.interpolate(
                frame0=start_rgb,
                frame1=end_rgb,
                num_frames=num_frames,
            )
        except Exception as exc:
            raise AnimationError(f"ToonCrafter inference failed: {exc}") from exc

        return [
            cv2.cvtColor(np.array(f, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            for f in frames_rgb
        ]

    # ── Linear-blend fallback (no model required) ─────────────────────────────

    @staticmethod
    def _linear_blend_fallback(
        frame_start: np.ndarray,
        frame_end: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """
        CPU-only fallback: linearly interpolate pixel values between two frames.
        Produces temporally smooth but visually simple transitions.
        """
        if num_frames < 1:
            return [frame_start, frame_end]

        h, w = frame_start.shape[:2]
        if frame_end.shape[:2] != (h, w):
            frame_end = cv2.resize(frame_end, (w, h), interpolation=cv2.INTER_LINEAR)

        start_f = frame_start.astype(np.float32)
        end_f   = frame_end.astype(np.float32)

        frames: list[np.ndarray] = [frame_start]
        for i in range(1, num_frames + 1):
            t = i / (num_frames + 1)
            blended = (1.0 - t) * start_f + t * end_f
            frames.append(np.clip(blended, 0, 255).astype(np.uint8))
        frames.append(frame_end)
        return frames
