"""
ToonCrafter wrapper for the meme_animator module.

ToonCrafter is a video diffusion model for cartoon/anime content.
In the emoSVG pipeline it serves two roles:

1. Primary driver mode (use_toon_crafter_as_driver=True):
   Given a neutral source frame and a peak expression frame produced by a
   single LivePortrait render, ToonCrafter's video diffusion backbone generates
   the *entire* animation sequence — wind-up, attack, hold, and settle — as a
   temporally coherent cartoon video.  This produces true Squash-and-Stretch
   motion because the diffusion model interpolates in image space rather than
   warping keypoint coordinates.

   Call: generate_from_boundaries(source_frame, peak_frame, num_frames)

2. Interpolation mode (legacy, use_toon_crafter=True without driver flag):
   Given a sequence of LivePortrait keyframes, ToonCrafter inserts
   frames_between interpolated frames between each consecutive pair for
   smoother cartoon motion.

   Call: smooth_sequence(keyframes, frames_between)

Architecture note:
    In driver mode LivePortrait renders only ONE frame (the peak expression).
    ToonCrafter then owns the full temporal generation.  In interpolation mode
    LivePortrait renders the full keyframe sequence and ToonCrafter fills gaps.

When ToonCrafter weights are absent both modes fall back to CPU-only
implementations so the pipeline can be tested without GPU or model downloads.
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

    def generate_from_boundaries(
        self,
        source_frame: np.ndarray,
        peak_frame: np.ndarray,
        num_frames: int = 16,
    ) -> list[np.ndarray]:
        """
        Primary driver mode: generate a full animation sequence from a neutral
        source frame to a peak expression frame.

        ToonCrafter's video diffusion backbone treats source_frame and
        peak_frame as the first and last frames of a video clip and generates
        num_frames temporally coherent in-between frames.  The result is a
        complete wind-up → attack → hold arc in a single diffusion pass.

        Parameters
        ----------
        source_frame: HxWx3 uint8 BGR — neutral expression (animation start).
        peak_frame:   HxWx3 uint8 BGR — peak expression (animation end).
        num_frames:   Total frames in the output sequence including boundaries.
                      Must be >= 2.  Typical values: 12–24.

        Returns
        -------
        List of num_frames HxWx3 uint8 BGR frames:
            [source_frame, generated_1, ..., generated_N, peak_frame]
        """
        if num_frames < 2:
            return [source_frame, peak_frame]

        if self._use_fallback:
            return self._driver_fallback(source_frame, peak_frame, num_frames)

        with self._registry.model_context(self.MODEL_ID, offload_after=True) as model:
            return self._toon_crafter_driver(model, source_frame, peak_frame, num_frames)

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

    def _toon_crafter_driver(
        self,
        model,
        source_frame: np.ndarray,
        peak_frame: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """
        Real ToonCrafter inference in driver mode.

        Calls model.generate(frame0, frame1, num_frames) which uses the video
        diffusion backbone to produce a temporally coherent sequence.  The API
        mirrors ToonCrafter's official inference interface where frame0/frame1
        are the conditioning boundary frames.
        """
        source_rgb = cv2.cvtColor(source_frame, cv2.COLOR_BGR2RGB)
        peak_rgb   = cv2.cvtColor(peak_frame,   cv2.COLOR_BGR2RGB)
        try:
            # ToonCrafter's generate() API: boundary-conditioned video generation.
            # num_frames includes the two boundary frames.
            frames_rgb = model.generate(
                frame0=source_rgb,
                frame1=peak_rgb,
                num_frames=num_frames,
            )
        except AttributeError:
            # Older ToonCrafter versions expose interpolate() only — fall back
            # to calling interpolate with (num_frames - 2) inner frames.
            inner = max(num_frames - 2, 1)
            frames_rgb = model.interpolate(
                frame0=source_rgb,
                frame1=peak_rgb,
                num_frames=inner,
            )
        except Exception as exc:
            raise AnimationError(f"ToonCrafter driver inference failed: {exc}") from exc

        return [
            cv2.cvtColor(np.array(f, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            for f in frames_rgb
        ]

    @staticmethod
    def _driver_fallback(
        source_frame: np.ndarray,
        peak_frame: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        """
        CPU-only fallback for driver mode.

        Produces a smooth ease-in/ease-out sequence from source to peak using
        a cosine schedule, which approximates the anticipation → attack → hold
        arc better than a linear blend.
        """
        import math

        h, w = source_frame.shape[:2]
        if peak_frame.shape[:2] != (h, w):
            peak_frame = cv2.resize(peak_frame, (w, h), interpolation=cv2.INTER_LINEAR)

        src_f  = source_frame.astype(np.float32)
        peak_f = peak_frame.astype(np.float32)

        frames: list[np.ndarray] = []
        for i in range(num_frames):
            # Cosine ease-in-out: slow start, fast middle, slow end
            t_linear = i / max(num_frames - 1, 1)
            t = (1.0 - math.cos(t_linear * math.pi)) / 2.0
            blended = (1.0 - t) * src_f + t * peak_f
            frames.append(np.clip(blended, 0, 255).astype(np.uint8))
        return frames

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
