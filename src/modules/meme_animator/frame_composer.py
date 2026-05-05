"""
Frame composer: assembles rendered frames into a GIF, MP4, or WebP animation.

Responsibilities:
- Resize frames to the requested resolution.
- Write output file via imageio (GIF/WebP) or OpenCV (MP4).
- Return the output path and basic metadata.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameComposer:
    """Assembles a list of BGR frames into an animation file."""

    def compose(
        self,
        frames: list[np.ndarray],
        output_path: Path,
        fps: int,
        resolution: tuple[int, int],
        output_format: str,
    ) -> Path:
        """
        Parameters
        ----------
        frames:        List of HxWx3 uint8 BGR frames.
        output_path:   Destination file path (extension must match output_format).
        fps:           Frames per second.
        resolution:    (width, height) target resolution.
        output_format: "gif" | "mp4" | "webp"

        Returns
        -------
        Resolved output path.
        """
        if not frames:
            raise ValueError("frames list is empty — nothing to compose")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        resized = [self._resize(f, resolution) for f in frames]

        if output_format == "gif":
            self._write_gif(resized, output_path, fps)
        elif output_format == "mp4":
            self._write_mp4(resized, output_path, fps)
        elif output_format == "webp":
            self._write_webp(resized, output_path, fps)
        else:
            raise ValueError(f"Unsupported output_format: {output_format!r}")

        logger.info("Animation written to %s (%d frames @ %d fps)", output_path, len(frames), fps)
        return output_path

    # ── Format writers ────────────────────────────────────────────────────

    @staticmethod
    def _write_gif(frames: list[np.ndarray], path: Path, fps: int) -> None:
        try:
            import imageio
        except ImportError as exc:
            raise ImportError("imageio is required for GIF output: pip install imageio") from exc

        duration_ms = int(1000 / fps)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        imageio.mimsave(str(path), rgb_frames, format="GIF", duration=duration_ms, loop=0)

    @staticmethod
    def _write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()

    @staticmethod
    def _write_webp(frames: list[np.ndarray], path: Path, fps: int) -> None:
        try:
            import imageio
        except ImportError as exc:
            raise ImportError("imageio is required for WebP output: pip install imageio") from exc

        duration_ms = int(1000 / fps)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        imageio.mimsave(str(path), rgb_frames, format="WEBP", duration=duration_ms, loop=0)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _resize(frame: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
        w, h = resolution
        if frame.shape[1] == w and frame.shape[0] == h:
            return frame
        return cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
