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

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import logging
from pathlib import Path

import cv2
import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import AnimationError

logger = logging.getLogger(__name__)

_TOON_CRAFTER_VRAM_GB = 8.0
_VIDEO_EXTENSIONS = (".mp4", ".gif", ".webp", ".avi", ".mov")


class _OfficialToonCrafterSubprocess:
    """Adapter that calls ToonCrafter's official inference.py entry point."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.ckpt_path = model_path / "model.ckpt"
        self.config_path = model_path / "config.yaml"
        self.repo_path = self._resolve_repo_path()
        self.script_path = self.repo_path / "scripts" / "evaluation" / "inference.py"
        if not self.script_path.exists():
            raise AnimationError(
                "ToonCrafter official inference script not found. "
                "Set TOON_CRAFTER_REPO_PATH to the official ToonCrafter source directory."
            )

    def interpolate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        target_count = max(num_frames + 2, 2)
        return self._run_pairs([(frame0, frame1)], target_count=target_count)[0]

    def generate(
        self,
        frame0: np.ndarray,
        frame1: np.ndarray,
        num_frames: int,
    ) -> list[np.ndarray]:
        target_count = max(num_frames, 2)
        return self._run_pairs([(frame0, frame1)], target_count=target_count)[0]

    def interpolate_pairs(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        frames_between: int,
    ) -> list[list[np.ndarray]]:
        target_count = max(frames_between + 2, 2)
        return self._run_pairs(pairs, target_count=target_count)

    def _run_pairs(
        self,
        pairs: list[tuple[np.ndarray, np.ndarray]],
        target_count: int,
    ) -> list[list[np.ndarray]]:
        if not pairs:
            return []

        keep_tmp = os.getenv("TOON_CRAFTER_KEEP_TMP", "0") == "1"
        tmp_root = Path(tempfile.mkdtemp(prefix="emosvg_tooncrafter_"))
        prompt_dir = tmp_root / "prompts"
        output_dir = tmp_root / "outputs"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            prompt = os.getenv("TOON_CRAFTER_PROMPT", "a cartoon character animation")
            with (prompt_dir / "prompts.txt").open("w", encoding="utf-8") as fh:
                for _ in pairs:
                    fh.write(prompt + "\n")

            for idx, (frame0, frame1) in enumerate(pairs):
                frame0, frame1 = self._match_pair_sizes(frame0, frame1)
                cv2.imwrite(str(prompt_dir / f"pair_{idx:04d}_0.png"), frame0)
                cv2.imwrite(str(prompt_dir / f"pair_{idx:04d}_1.png"), frame1)

            h, w = pairs[0][0].shape[:2]
            height = self._aligned_size(int(os.getenv("TOON_CRAFTER_HEIGHT", str(h))))
            width = self._aligned_size(int(os.getenv("TOON_CRAFTER_WIDTH", str(w))))
            video_length = self._video_length(target_count)

            cmd = self._build_command(
                prompt_dir=prompt_dir,
                output_dir=output_dir,
                height=height,
                width=width,
                video_length=video_length,
            )
            logger.info(
                "Running official ToonCrafter: %d pair(s), %dx%d, %d frames, %s DDIM steps.",
                len(pairs),
                width,
                height,
                video_length,
                os.getenv("TOON_CRAFTER_DDIM_STEPS", "25"),
            )
            start = time.monotonic()
            proc = subprocess.run(
                cmd,
                cwd=str(self.repo_path),
                env=self._subprocess_env(),
                text=True,
                capture_output=True,
                timeout=float(os.getenv("TOON_CRAFTER_TIMEOUT_SEC", "1800")),
            )
            elapsed = time.monotonic() - start
            if proc.stdout:
                logger.info("ToonCrafter stdout:\n%s", proc.stdout.strip())
            if proc.stderr:
                logger.warning("ToonCrafter stderr:\n%s", proc.stderr.strip())
            if proc.returncode != 0:
                raise AnimationError(
                    f"Official ToonCrafter inference failed with code {proc.returncode}."
                )

            segments = []
            for idx, (frame0, frame1) in enumerate(pairs):
                video = self._find_output_video(output_dir, idx)
                frames = self._read_video_frames(video)
                frames = self._resample_frames(frames, target_count)
                h, w = frame0.shape[:2]
                frames = [
                    cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR)
                    if f.shape[:2] != (h, w)
                    else f
                    for f in frames
                ]
                frames[0] = frame0.copy()
                frames[-1] = frame1.copy()
                segments.append(frames)

            logger.info("Official ToonCrafter completed in %.2f s.", elapsed)
            return segments
        finally:
            if keep_tmp:
                logger.info("Keeping ToonCrafter temp directory: %s", tmp_root)
            else:
                shutil.rmtree(tmp_root, ignore_errors=True)

    def _build_command(
        self,
        prompt_dir: Path,
        output_dir: Path,
        height: int,
        width: int,
        video_length: int,
    ) -> list[str]:
        cmd = [
            os.getenv("TOON_CRAFTER_PYTHON", sys.executable),
            str(self.script_path),
            "--ckpt_path",
            str(self.ckpt_path),
            "--config",
            str(self.config_path),
            "--prompt_dir",
            str(prompt_dir),
            "--savedir",
            str(output_dir),
            "--height",
            str(height),
            "--width",
            str(width),
            "--video_length",
            str(video_length),
            "--interp",
            "--ddim_steps",
            os.getenv("TOON_CRAFTER_DDIM_STEPS", "25"),
            "--bs",
            "1",
        ]
        if os.getenv("TOON_CRAFTER_PERFRAME_AE", "1") != "0":
            cmd.append("--perframe_ae")
        extra_args = os.getenv("TOON_CRAFTER_EXTRA_ARGS")
        if extra_args:
            cmd.extend(shlex.split(extra_args))
        return cmd

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        parts = [str(self.repo_path)]
        if pythonpath:
            parts.append(pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(parts)
        return env

    @staticmethod
    def _resolve_repo_path() -> Path:
        raw = os.getenv("TOON_CRAFTER_REPO_PATH")
        if raw:
            return Path(raw).expanduser().resolve()
        try:
            import lvdm  # type: ignore
        except ImportError as exc:
            raise AnimationError(
                "ToonCrafter source is not importable. "
                "Add the official ToonCrafter directory to Python path or set "
                "TOON_CRAFTER_REPO_PATH."
            ) from exc
        return Path(lvdm.__file__).resolve().parents[1]

    @staticmethod
    def _aligned_size(value: int) -> int:
        return max(16, (value // 16) * 16)

    @staticmethod
    def _video_length(target_count: int) -> int:
        raw = int(os.getenv("TOON_CRAFTER_VIDEO_LENGTH", "16"))
        length = max(raw, target_count, 2)
        return length if length % 2 == 0 else length + 1

    @staticmethod
    def _match_pair_sizes(
        frame0: np.ndarray,
        frame1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h, w = frame0.shape[:2]
        if frame1.shape[:2] != (h, w):
            frame1 = cv2.resize(frame1, (w, h), interpolation=cv2.INTER_LINEAR)
        return frame0, frame1

    @staticmethod
    def _find_output_video(output_dir: Path, pair_index: int) -> Path:
        stem = f"pair_{pair_index:04d}_0"
        candidates = [
            p for p in output_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in _VIDEO_EXTENSIONS
            and p.stem.startswith(stem)
        ]
        if not candidates:
            all_candidates = [
                p for p in output_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
            ]
            all_candidates.sort(key=lambda p: str(p))
            if pair_index < len(all_candidates):
                return all_candidates[pair_index]
            raise AnimationError(
                f"ToonCrafter produced no video for pair {pair_index} in {output_dir}."
            )
        candidates.sort(key=lambda p: (p.stat().st_mtime, str(p)))
        return candidates[-1]

    @staticmethod
    def _read_video_frames(path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        if not frames:
            raise AnimationError(f"Failed to read ToonCrafter video output: {path}")
        return frames

    @staticmethod
    def _resample_frames(frames: list[np.ndarray], target_count: int) -> list[np.ndarray]:
        if len(frames) == target_count:
            return frames
        if target_count <= 1:
            return [frames[0]]
        indices = np.linspace(0, len(frames) - 1, target_count)
        return [frames[int(round(i))].copy() for i in indices]


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

        if not self._use_fallback:
            pairs = [
                (keyframes[i], keyframes[i + 1])
                for i in range(len(keyframes) - 1)
            ]
            with self._registry.model_context(self.MODEL_ID, offload_after=True) as model:
                if hasattr(model, "interpolate_pairs"):
                    segments = model.interpolate_pairs(pairs, frames_between=frames_between)
                    result: list[np.ndarray] = []
                    for segment in segments:
                        result.extend(segment[:-1])
                    result.append(keyframes[-1])
                    return result

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
        def _loader():
            try:
                assert self._model_path is not None
                return _OfficialToonCrafterSubprocess(self._model_path)
            except Exception as exc:
                raise AnimationError(
                    f"Failed to initialise ToonCrafter official adapter: {exc}"
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
        try:
            frames = model.interpolate(
                frame0=frame_start,
                frame1=frame_end,
                num_frames=num_frames,
            )
        except Exception as exc:
            raise AnimationError(f"ToonCrafter inference failed: {exc}") from exc

        return [np.array(f, dtype=np.uint8) for f in frames]

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
        try:
            # ToonCrafter's generate() API: boundary-conditioned video generation.
            # num_frames includes the two boundary frames.
            frames = model.generate(
                frame0=source_frame,
                frame1=peak_frame,
                num_frames=num_frames,
            )
        except AttributeError:
            # Older ToonCrafter versions expose interpolate() only — fall back
            # to calling interpolate with (num_frames - 2) inner frames.
            inner = max(num_frames - 2, 1)
            frames = model.interpolate(
                frame0=source_frame,
                frame1=peak_frame,
                num_frames=inner,
            )
        except Exception as exc:
            raise AnimationError(f"ToonCrafter driver inference failed: {exc}") from exc

        return [np.array(f, dtype=np.uint8) for f in frames]

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
