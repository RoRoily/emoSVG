"""
TripoSR wrapper for single-image 3D reconstruction.

When TripoSR weights are unavailable, falls back to a minimal trimesh
primitive (a UV-sphere) so the rest of the pipeline can be tested
without GPU or model downloads.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import MissingModelWeightsError, ReconstructionError
from .mesh_processor import MeshProcessor
from .exporter import MeshExporter
from .schemas import ReconstructionRequest, ReconstructionResult, MeshStats

logger = logging.getLogger(__name__)

_TRIPOSR_VRAM_GB = 6.0


class Reconstructor3D:
    """
    Orchestrates: background removal -> TripoSR inference -> mesh post-processing -> export.

    Parameters
    ----------
    model_id:       HuggingFace model ID or local path for TripoSR weights.
    output_dir:     Root directory for exported mesh files.
    registry:       ModelRegistry singleton.
    """

    MODEL_ID = "triposr"

    def __init__(
        self,
        model_id: str = "stabilityai/TripoSR",
        output_dir: Path = Path("outputs/meshes"),
        registry: Optional[ModelRegistry] = None,
    ) -> None:
        self._model_id_or_path = model_id
        self._output_dir = output_dir
        self._registry = registry or ModelRegistry.instance()
        self._processor = MeshProcessor()
        self._exporter = MeshExporter()
        self._use_fallback = not self._triposr_available()

        if not self._use_fallback:
            self._register_model()
        else:
            logger.warning(
                "TripoSR not available — using sphere fallback (testing only)."
            )

    # ── Public API ────────────────────────────────────────────────────────

    def reconstruct(self, request: ReconstructionRequest) -> ReconstructionResult:
        import time
        t0 = time.monotonic()
        logger.info("Reconstructor3D.reconstruct: %s", request.source_image_path.name)

        # 1. Load, strip background, and recenter foreground
        image_np = self._load_image(
            request.source_image_path,
            request.remove_background,
            request.foreground_ratio,
        )

        # 2. Run inference
        if self._use_fallback:
            mesh = self._sphere_fallback()
            backend = "triposr_fallback"
        else:
            mesh = self._triposr_infer(image_np, request.mc_resolution)
            backend = "triposr"

        # 3. Post-process mesh
        mesh = self._processor.process(mesh)

        # 4. Export
        self._output_dir.mkdir(parents=True, exist_ok=True)
        stem = request.source_image_path.stem
        output_paths = self._exporter.export(
            mesh, self._output_dir, stem, request.export_formats
        )

        stats = self._processor.compute_stats(mesh)
        elapsed = time.monotonic() - t0
        logger.info("Reconstruction done in %.2f s -> %s", elapsed, list(output_paths.values()))

        return ReconstructionResult(
            output_paths=output_paths,
            mesh_stats=stats,
            backend_used=backend,
        )

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/reconstruction_3d.yaml",
        registry: Optional[ModelRegistry] = None,
    ) -> "Reconstructor3D":
        from src.core import load_config
        cfg = load_config(config_path)
        return cls(
            model_id=cfg.get("triposr", {}).get("model_id", "stabilityai/TripoSR"),
            output_dir=Path(cfg.get("export", {}).get("output_dir", "outputs/meshes")),
            registry=registry,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _triposr_available(self) -> bool:
        try:
            import tsr  # noqa: F401
            return True
        except ImportError:
            return False

    def _register_model(self) -> None:
        model_id = self._model_id_or_path

        def _loader():
            try:
                from tsr.system import TSR  # type: ignore
                model = TSR.from_pretrained(
                    model_id,
                    config_name="config.yaml",
                    weight_name="model.ckpt",
                )
                model.renderer.set_chunk_size(8192)
                return model
            except Exception as exc:
                raise ReconstructionError(
                    f"Failed to load TripoSR from {model_id!r}: {exc}"
                ) from exc

        self._registry.register(
            self.MODEL_ID, _loader, estimated_vram_gb=_TRIPOSR_VRAM_GB
        )

    def _triposr_infer(self, image_np: np.ndarray, mc_resolution: int):
        import torch
        with self._registry.model_context(self.MODEL_ID, offload_after=True) as model:
            try:
                with torch.no_grad():
                    scene_codes = model([image_np], device=self._registry.device_manager.device)
                    meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)
                return meshes[0]
            except Exception as exc:
                raise ReconstructionError(f"TripoSR inference failed: {exc}") from exc

    @staticmethod
    def _load_image(
        path: Path,
        remove_bg: bool,
        foreground_ratio: float = 0.85,
    ) -> np.ndarray:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ReconstructionError(f"Failed to read image: {path}")

        if remove_bg:
            channels = img.shape[2] if img.ndim == 3 else 1
            if channels == 3:
                try:
                    from rembg import remove as rembg_remove  # type: ignore
                    import PIL.Image
                    pil = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    pil_no_bg = rembg_remove(pil)
                    img = np.array(pil_no_bg)  # RGBA
                except ImportError:
                    logger.warning("rembg not installed — skipping background removal.")

        # Normalise to float32 RGBA [0, 1] first so we can use alpha for cropping
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # Crop and recenter the foreground subject using the alpha channel.
        # TripoSR expects the subject to fill ~foreground_ratio of the frame.
        if img.ndim == 3 and img.shape[2] == 4:
            img = Reconstructor3D._recenter_foreground(img, foreground_ratio)

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]  # drop alpha after recentering
        return img

    @staticmethod
    def _recenter_foreground(rgba: np.ndarray, foreground_ratio: float) -> np.ndarray:
        """
        Crop the tight bounding box of the alpha foreground, then pad it back
        to a square so the subject fills foreground_ratio of the output frame.

        Parameters
        ----------
        rgba:             HxWx4 float32 image in [0, 1].
        foreground_ratio: Fraction of the output frame the subject should fill.

        Returns
        -------
        HxWx4 float32 image with the subject centred and scaled.
        """
        import cv2
        alpha = rgba[..., 3]
        mask = (alpha > 0.05).astype(np.uint8)

        if mask.sum() == 0:
            # No foreground detected — return as-is
            return rgba

        # Tight bounding box of the foreground
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        # Expand to a square centred on the foreground bbox
        fg_h = r_max - r_min + 1
        fg_w = c_max - c_min + 1
        side = max(fg_h, fg_w)

        # Output canvas size: side / foreground_ratio so the subject fills the ratio
        canvas_side = int(round(side / foreground_ratio))

        # Centre of the foreground bbox
        cy = (r_min + r_max) / 2.0
        cx = (c_min + c_max) / 2.0

        # Source crop window (may extend outside the original image)
        half = canvas_side / 2.0
        src_r0 = cy - half
        src_c0 = cx - half

        # Build output canvas (white background for RGB, 0 alpha for padding)
        out = np.ones((canvas_side, canvas_side, 4), dtype=np.float32)
        out[..., 3] = 0.0  # transparent padding

        # Compute the overlap between the source image and the crop window
        h, w = rgba.shape[:2]
        dst_r0 = max(0, -int(src_r0))
        dst_c0 = max(0, -int(src_c0))
        src_r_start = max(0, int(src_r0))
        src_c_start = max(0, int(src_c0))
        src_r_end = min(h, int(src_r0) + canvas_side)
        src_c_end = min(w, int(src_c0) + canvas_side)
        copy_h = src_r_end - src_r_start
        copy_w = src_c_end - src_c_start

        if copy_h > 0 and copy_w > 0:
            out[dst_r0:dst_r0 + copy_h, dst_c0:dst_c0 + copy_w] = (
                rgba[src_r_start:src_r_end, src_c_start:src_c_end]
            )

        return out

    @staticmethod
    def _sphere_fallback():
        """Return a trimesh UV-sphere as a stand-in mesh."""
        import trimesh
        return trimesh.creation.uv_sphere(radius=1.0, count=[32, 32])
