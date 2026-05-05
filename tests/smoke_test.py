"""
Smoke tests: end-to-end pipeline runs in fallback mode (no GPU, no weights).

Tests:
  1. All 5 expressions produce a valid animation GIF.
  2. LivePortrait output (peak keyframe) feeds into TripoSR → mesh exported.
  3. Full pipeline: animation + 3D + SVG in a single execute() call.
  4. Foreground crop/recenter produces a correctly-shaped float32 array.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.model_registry import ModelRegistry
from src.modules.meme_animator.animator import MemeAnimator
from src.modules.meme_animator.schemas import AnimationRequest, MemeExpression
from src.modules.reconstructor_3d.reconstructor import Reconstructor3D
from src.modules.reconstructor_3d.schemas import ExportFormat, ReconstructionRequest
from src.pipeline.full_pipeline import FullPipeline, FullPipelineRequest


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


@pytest.fixture()
def char_image(tmp_path: Path) -> Path:
    """256×256 test character image with a simple two-tone face."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    # Skin-tone face circle
    cv2.circle(img, (128, 128), 90, (180, 140, 100), -1)
    # Eyes
    cv2.circle(img, (100, 110), 18, (30, 30, 30), -1)
    cv2.circle(img, (156, 110), 18, (30, 30, 30), -1)
    # Mouth
    cv2.ellipse(img, (128, 160), (35, 18), 0, 0, 180, (30, 30, 30), -1)
    p = tmp_path / "character.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture()
def mock_clip_bundle():
    """Minimal CLIP mock so IPExtractor skips HuggingFace downloads."""
    import torch
    from unittest.mock import MagicMock
    proc = MagicMock()
    proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    out = MagicMock()
    out.image_embeds = torch.zeros(1, 768)
    model = MagicMock()
    model.return_value = out
    model.to = MagicMock(return_value=model)
    return (proc, model)


def _inject_clip(pipeline: FullPipeline, bundle) -> None:
    from src.core.model_registry import ModelState
    from src.modules.ip_extractor.extractor import IPExtractor
    entry = pipeline._registry._entries.get(IPExtractor.MODEL_ID)
    if entry:
        entry.module = bundle
        entry.state = ModelState.ON_CPU


# ── Smoke 1: all expressions produce a valid animation ───────────────────────

class TestAllExpressionsAnimate:
    @pytest.mark.parametrize("expr", [
        MemeExpression.SHOCK,
        MemeExpression.LAUGH,
        MemeExpression.RAGE,
        MemeExpression.CRY,
        MemeExpression.SMUG,
    ])
    def test_expression_produces_gif(self, expr, char_image, tmp_path):
        animator = MemeAnimator(output_dir=tmp_path)
        req = AnimationRequest(
            source_image_path=char_image,
            expression=expr,
            output_format="gif",
            fps=12,
            resolution=(64, 64),
        )
        result = animator.generate(req)

        assert result.output_path.exists(), f"No output for {expr}"
        assert result.output_path.stat().st_size > 0, f"Empty GIF for {expr}"
        assert result.frame_count > 0, f"Zero frames for {expr}"
        assert result.backend_used in ("live_portrait", "live_portrait_fallback")


# ── Smoke 2: LivePortrait peak keyframe → TripoSR mesh ───────────────────────

class TestLivePortraitToTripoSR:
    def test_peak_keyframe_feeds_triposr(self, char_image, tmp_path):
        """
        Run the animator, extract the peak keyframe, then pass it to
        Reconstructor3D.  Both run in fallback mode — no GPU required.
        """
        # Step A: animate
        animator = MemeAnimator(output_dir=tmp_path / "anim")
        anim_req = AnimationRequest(
            source_image_path=char_image,
            expression=MemeExpression.SHOCK,
            output_format="gif",
            fps=12,
            resolution=(64, 64),
        )
        anim_result = animator.generate(anim_req)
        assert anim_result.frame_count > 0

        # Step B: pick the peak keyframe (middle of the sequence)
        assert anim_result.keyframes, "Animator produced no keyframes"
        peak_kf = anim_result.keyframes[len(anim_result.keyframes) // 2]
        peak_img: np.ndarray = peak_kf.image  # HxWx3 uint8 BGR

        # Persist to disk so Reconstructor3D can read it
        kf_path = tmp_path / "peak_keyframe.png"
        cv2.imwrite(str(kf_path), peak_img)
        assert kf_path.exists()

        # Step C: reconstruct 3D from the peak keyframe
        rec = Reconstructor3D(output_dir=tmp_path / "meshes")
        rec_req = ReconstructionRequest(
            source_image_path=kf_path,
            export_formats=[ExportFormat.OBJ, ExportFormat.GLB],
            mc_resolution=64,
            remove_background=False,  # no rembg needed in smoke test
        )
        rec_result = rec.reconstruct(rec_req)

        assert rec_result.backend_used == "triposr_fallback"
        assert rec_result.mesh_stats.vertex_count > 0
        assert rec_result.mesh_stats.face_count > 0
        assert rec_result.output_paths["obj"].exists()
        assert rec_result.output_paths["glb"].exists()

    def test_mesh_bounding_box_nonzero(self, char_image, tmp_path):
        """Reconstructed mesh must have a non-degenerate bounding box."""
        animator = MemeAnimator(output_dir=tmp_path / "anim")
        anim_result = animator.generate(AnimationRequest(
            source_image_path=char_image,
            expression=MemeExpression.LAUGH,
            fps=12,
            resolution=(64, 64),
        ))
        peak_kf = anim_result.keyframes[len(anim_result.keyframes) // 2]
        kf_path = tmp_path / "peak.png"
        cv2.imwrite(str(kf_path), peak_kf.image)

        rec = Reconstructor3D(output_dir=tmp_path / "meshes")
        result = rec.reconstruct(ReconstructionRequest(
            source_image_path=kf_path,
            mc_resolution=64,
            remove_background=False,
        ))
        assert all(v > 0 for v in result.mesh_stats.bounding_box)


# ── Smoke 3: full pipeline (animation + 3D + SVG) ────────────────────────────

class TestFullPipelineSmoke:
    def test_full_pipeline_animation_only(self, char_image, tmp_path, mock_clip_bundle):
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req = FullPipelineRequest(
            source_image_path=char_image,
            expression=MemeExpression.SHOCK,
            output_format="gif",
            fps=12,
            resolution=(64, 64),
            run_3d=False,
            run_svg=False,
        )
        result = pipeline.execute(req)

        assert result.animation.output_path.exists()
        assert result.animation.frame_count > 0
        assert result.reconstruction is None
        assert result.vectorization is None
        assert result.elapsed_seconds > 0

    def test_full_pipeline_with_3d_uses_peak_keyframe(
        self, char_image, tmp_path, mock_clip_bundle
    ):
        """
        Default (use_source_for_3d_svg=False): 3D reconstruction uses the
        peak animation keyframe, not the original source image.
        """
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req = FullPipelineRequest(
            source_image_path=char_image,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            run_3d=True,
            run_svg=False,
            use_source_for_3d_svg=False,
        )
        result = pipeline.execute(req)

        assert result.reconstruction is not None
        assert result.reconstruction.mesh_stats.face_count > 0
        # A tmp keyframe file must have been written
        tmp_kf_dir = tmp_path / "_tmp_keyframes"
        assert tmp_kf_dir.exists()
        assert any(tmp_kf_dir.iterdir())

    def test_full_pipeline_all_modules(self, char_image, tmp_path, mock_clip_bundle):
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req = FullPipelineRequest(
            source_image_path=char_image,
            expression=MemeExpression.LAUGH,
            fps=12,
            resolution=(64, 64),
            run_3d=True,
            run_svg=True,
        )
        result = pipeline.execute(req)

        assert result.animation.output_path.exists()
        assert result.reconstruction is not None
        assert result.vectorization is not None
        assert result.vectorization.output_path.exists()


# ── Smoke 4: foreground crop/recenter ────────────────────────────────────────

class TestForegroundRecenter:
    def test_recenter_produces_square_output(self):
        """_recenter_foreground must return a square canvas."""
        rgba = np.zeros((200, 300, 4), dtype=np.float32)
        # Place a small foreground blob off-centre
        rgba[50:120, 80:180, :3] = 0.8
        rgba[50:120, 80:180, 3] = 1.0

        result = Reconstructor3D._recenter_foreground(rgba, foreground_ratio=0.85)
        assert result.shape[0] == result.shape[1], "Output must be square"
        assert result.shape[2] == 4

    def test_recenter_subject_fills_ratio(self):
        """After recentering, the foreground bbox should fill ~foreground_ratio."""
        rgba = np.zeros((256, 256, 4), dtype=np.float32)
        rgba[60:196, 60:196, :3] = 0.9
        rgba[60:196, 60:196, 3] = 1.0

        ratio = 0.80
        result = Reconstructor3D._recenter_foreground(rgba, foreground_ratio=ratio)
        canvas_side = result.shape[0]

        # Measure the foreground in the output
        alpha_out = result[..., 3]
        rows = np.any(alpha_out > 0.05, axis=1)
        cols = np.any(alpha_out > 0.05, axis=0)
        fg_h = np.sum(rows)
        fg_w = np.sum(cols)
        actual_ratio = max(fg_h, fg_w) / canvas_side
        assert abs(actual_ratio - ratio) < 0.15, (
            f"Expected ratio ~{ratio}, got {actual_ratio:.2f}"
        )

    def test_recenter_no_foreground_returns_unchanged(self):
        """Fully transparent image should be returned as-is."""
        rgba = np.zeros((128, 128, 4), dtype=np.float32)
        result = Reconstructor3D._recenter_foreground(rgba, foreground_ratio=0.85)
        assert result.shape == rgba.shape

    def test_load_image_rgb_output(self, tmp_path):
        """_load_image must return a float32 HxWx3 array (no alpha channel)."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        p = tmp_path / "test.png"
        cv2.imwrite(str(p), img)

        result = Reconstructor3D._load_image(p, remove_bg=False)
        assert result.dtype == np.float32
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.max() <= 1.0


# ── Smoke 5: ToonCrafter inter-frame smoothing ───────────────────────────────

class TestToonCrafterSmoke:
    """
    End-to-end smoke tests for ToonCrafter smoothing wired into the pipeline.
    All tests run in fallback mode (no GPU, no weights).
    """

    def test_toon_crafter_increases_frame_count(self, char_image, tmp_path):
        """Enabling ToonCrafter should produce more frames than without it."""
        animator = MemeAnimator(output_dir=tmp_path / "anim")

        result_plain = animator.generate(AnimationRequest(
            source_image_path=char_image,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=False,
        ))
        result_smooth = animator.generate(AnimationRequest(
            source_image_path=char_image,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=3,
        ))

        assert result_smooth.frame_count > result_plain.frame_count
        assert "toon_crafter" in result_smooth.backend_used

    def test_toon_crafter_output_is_valid_gif(self, char_image, tmp_path):
        """GIF produced with ToonCrafter smoothing must be a valid file."""
        animator = MemeAnimator(output_dir=tmp_path / "anim")
        result = animator.generate(AnimationRequest(
            source_image_path=char_image,
            expression=MemeExpression.LAUGH,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=2,
        ))
        assert result.output_path.exists()
        assert result.output_path.stat().st_size > 0

    def test_full_pipeline_with_toon_crafter(self, char_image, tmp_path, mock_clip_bundle):
        """Full pipeline should accept use_toon_crafter and produce more frames."""
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req_plain = FullPipelineRequest(
            source_image_path=char_image,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            run_3d=False,
            run_svg=False,
            use_toon_crafter=False,
        )
        req_smooth = FullPipelineRequest(
            source_image_path=char_image,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            run_3d=False,
            run_svg=False,
            use_toon_crafter=True,
            frames_between=2,
        )
        result_plain = pipeline.execute(req_plain)
        result_smooth = pipeline.execute(req_smooth)

        assert result_smooth.animation.frame_count > result_plain.animation.frame_count
        assert "toon_crafter" in result_smooth.animation.backend_used

    def test_toon_crafter_fallback_label(self, char_image, tmp_path):
        """Without ToonCrafter weights, backend label should say fallback."""
        animator = MemeAnimator(toon_crafter_path=None, output_dir=tmp_path / "anim")
        result = animator.generate(AnimationRequest(
            source_image_path=char_image,
            expression=MemeExpression.RAGE,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=2,
        ))
        assert "toon_crafter_fallback" in result.backend_used
