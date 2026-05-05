"""
Unit tests for meme_animator sub-modules.
No GPU, no model weights required — LivePortrait runs in affine-warp fallback.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.model_registry import ModelRegistry
from src.modules.meme_animator.motion_designer import MotionDesigner, _PRESETS, _NEUTRAL
from src.modules.meme_animator.schemas import (
    AnimationRequest, MemeExpression, SquashParams,
)
from src.modules.meme_animator.live_portrait import LivePortraitWrapper
from src.modules.meme_animator.frame_composer import FrameComposer
from src.modules.meme_animator.animator import MemeAnimator


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    """Write a 256x256 solid-colour PNG and return its path."""
    img = np.full((256, 256, 3), (100, 150, 200), dtype=np.uint8)
    p = tmp_path / "sample.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture()
def sample_bgr() -> np.ndarray:
    return np.full((256, 256, 3), (100, 150, 200), dtype=np.uint8)


# ── SquashParams validation ───────────────────────────────────────────────

class TestSquashParams:
    def test_defaults_are_neutral(self):
        p = SquashParams()
        assert p.eye_bulge_scale == 1.0
        assert p.head_tilt_deg == 0.0

    def test_out_of_range_raises(self):
        with pytest.raises(Exception):
            SquashParams(eye_bulge_scale=99.0)

    def test_valid_custom_params(self):
        p = SquashParams(eye_bulge_scale=2.5, jaw_drop_scale=1.8)
        assert p.eye_bulge_scale == pytest.approx(2.5)


# ── AnimationRequest validation ───────────────────────────────────────────

class TestAnimationRequest:
    def test_valid_request(self, sample_image_path):
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
        )
        assert req.fps == 24

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(Exception):
            AnimationRequest(source_image_path=tmp_path / "nonexistent.png")

    def test_custom_expression_requires_params(self, sample_image_path):
        with pytest.raises(Exception):
            AnimationRequest(
                source_image_path=sample_image_path,
                expression=MemeExpression.CUSTOM,
                custom_params=None,
            )

    def test_custom_expression_with_params_ok(self, sample_image_path):
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.CUSTOM,
            custom_params=SquashParams(eye_bulge_scale=2.0),
        )
        assert req.expression == MemeExpression.CUSTOM


# ── MotionDesigner ────────────────────────────────────────────────────────

class TestMotionDesigner:
    def test_all_presets_produce_frames(self):
        md = MotionDesigner()
        for expr in MemeExpression:
            if expr == MemeExpression.CUSTOM:
                continue
            frames = md.design(expr)
            assert len(frames) > 0, f"No frames for {expr}"

    def test_frame_count_formula(self):
        md = MotionDesigner()
        peak = _PRESETS[MemeExpression.SHOCK]
        frames = md.design(MemeExpression.SHOCK)
        expected = (
            md.WIND_UP_FRAMES + md.ATTACK_FRAMES
            + peak.hold_frames + peak.bounce_frames + md.SETTLE_FRAMES
        )
        assert len(frames) == expected

    def test_first_frame_close_to_neutral(self):
        md = MotionDesigner()
        frames = md.design(MemeExpression.SHOCK)
        first = frames[0]
        # Wind-up frame should be close to neutral (not at peak)
        assert abs(first.eye_bulge_scale - 1.0) < 0.5

    def test_peak_frame_has_max_bulge(self):
        md = MotionDesigner()
        frames = md.design(MemeExpression.SHOCK)
        peak_idx = md.WIND_UP_FRAMES + md.ATTACK_FRAMES - 1
        assert frames[peak_idx].eye_bulge_scale > 2.0

    def test_last_frame_close_to_neutral(self):
        md = MotionDesigner()
        frames = md.design(MemeExpression.SHOCK)
        last = frames[-1]
        assert abs(last.eye_bulge_scale - 1.0) < 0.3
        assert abs(last.jaw_drop_scale - 1.0) < 0.3

    def test_custom_expression(self):
        md = MotionDesigner()
        custom = SquashParams(eye_bulge_scale=3.0, jaw_drop_scale=2.5)
        frames = md.design(MemeExpression.CUSTOM, custom_params=custom)
        assert len(frames) > 0

    def test_custom_without_params_raises(self):
        md = MotionDesigner()
        with pytest.raises(ValueError):
            md.design(MemeExpression.CUSTOM)

    def test_wind_up_is_counter_pose(self):
        md = MotionDesigner()
        peak = _PRESETS[MemeExpression.SHOCK]
        wind_up = md._wind_up(peak)
        # Wind-up eye_bulge should be slightly BELOW neutral (counter-pose)
        assert wind_up.eye_bulge_scale < 1.0

    def test_all_frame_params_in_valid_range(self):
        md = MotionDesigner()
        for expr in MemeExpression:
            if expr == MemeExpression.CUSTOM:
                continue
            for frame in md.design(expr):
                # Pydantic validators enforce ranges — re-instantiation must not raise
                SquashParams(**frame.model_dump())


# ── LivePortraitWrapper (fallback mode) ───────────────────────────────────

class TestLivePortraitFallback:
    def test_uses_fallback_when_no_weights(self):
        wrapper = LivePortraitWrapper(model_path=None)
        assert wrapper._use_fallback is True

    def test_render_frame_returns_correct_shape(self, sample_bgr):
        wrapper = LivePortraitWrapper(model_path=None)
        params = SquashParams(head_squash_scale=0.85, head_stretch_scale=1.15)
        result = wrapper.render_frame(sample_bgr, params)
        assert result.shape == sample_bgr.shape
        assert result.dtype == np.uint8

    def test_render_sequence_length(self, sample_bgr):
        wrapper = LivePortraitWrapper(model_path=None)
        md = MotionDesigner()
        param_seq = md.design(MemeExpression.SHOCK)
        frames = wrapper.render_sequence(sample_bgr, param_seq)
        assert len(frames) == len(param_seq)

    def test_neutral_params_minimal_change(self, sample_bgr):
        wrapper = LivePortraitWrapper(model_path=None)
        neutral = SquashParams()
        result = wrapper.render_frame(sample_bgr, neutral)
        # With neutral params the image should be nearly identical
        diff = np.abs(result.astype(int) - sample_bgr.astype(int)).mean()
        assert diff < 5.0

    def test_bulge_params_changes_image(self):
        # Use a gradient image so affine warp produces measurable pixel changes.
        rng = np.random.default_rng(42)
        textured = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
        wrapper = LivePortraitWrapper(model_path=None)
        bulge = SquashParams(eye_bulge_scale=2.8, head_squash_scale=0.7)
        result = wrapper.render_frame(textured, bulge)
        diff = np.abs(result.astype(int) - textured.astype(int)).mean()
        assert diff > 0.5  # must produce visible change on a non-uniform image


# ── FrameComposer ─────────────────────────────────────────────────────────

class TestFrameComposer:
    def _make_frames(self, n: int = 5) -> list[np.ndarray]:
        return [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(n)]

    def test_gif_output_created(self, tmp_path):
        composer = FrameComposer()
        out = tmp_path / "test.gif"
        composer.compose(self._make_frames(), out, fps=12, resolution=(64, 64), output_format="gif")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_mp4_output_created(self, tmp_path):
        composer = FrameComposer()
        out = tmp_path / "test.mp4"
        composer.compose(self._make_frames(), out, fps=12, resolution=(64, 64), output_format="mp4")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_frames_raises(self, tmp_path):
        composer = FrameComposer()
        with pytest.raises(ValueError):
            composer.compose([], tmp_path / "out.gif", fps=12, resolution=(64, 64), output_format="gif")

    def test_resize_applied(self, tmp_path):
        composer = FrameComposer()
        frames = [np.zeros((128, 128, 3), dtype=np.uint8)]
        out = tmp_path / "resized.gif"
        composer.compose(frames, out, fps=12, resolution=(64, 64), output_format="gif")
        assert out.exists()

    def test_unsupported_format_raises(self, tmp_path):
        composer = FrameComposer()
        with pytest.raises(ValueError):
            composer.compose(
                self._make_frames(), tmp_path / "out.xyz",
                fps=12, resolution=(64, 64), output_format="xyz",
            )

    def test_webp_output_created(self, tmp_path):
        composer = FrameComposer()
        out = tmp_path / "test.webp"
        composer.compose(self._make_frames(), out, fps=12, resolution=(64, 64), output_format="webp")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_webp_is_valid_animation(self, tmp_path):
        """WebP output should be readable back as a multi-frame image."""
        import imageio
        composer = FrameComposer()
        out = tmp_path / "anim.webp"
        frames = self._make_frames(n=4)
        composer.compose(frames, out, fps=8, resolution=(32, 32), output_format="webp")
        # imageio should be able to read it back without error
        reader = imageio.get_reader(str(out))
        read_frames = list(reader)
        assert len(read_frames) >= 1  # at least one frame readable


# ── MemeAnimator end-to-end (fallback mode) ───────────────────────────────

class TestMemeAnimatorE2E:
    def test_generate_gif(self, sample_image_path, tmp_path):
        animator = MemeAnimator(
            live_portrait_path=None,
            output_dir=tmp_path,
        )
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            output_format="gif",
            fps=12,
            resolution=(64, 64),
        )
        result = animator.generate(req)
        assert result.output_path.exists()
        assert result.frame_count > 0
        assert result.backend_used == "live_portrait_fallback"

    def test_generate_all_expressions(self, sample_image_path, tmp_path):
        animator = MemeAnimator(live_portrait_path=None, output_dir=tmp_path)
        for expr in MemeExpression:
            if expr == MemeExpression.CUSTOM:
                continue
            req = AnimationRequest(
                source_image_path=sample_image_path,
                expression=expr,
                output_format="gif",
                fps=12,
                resolution=(64, 64),
            )
            result = animator.generate(req)
            assert result.output_path.exists(), f"No output for {expr}"

    def test_result_metadata(self, sample_image_path, tmp_path):
        animator = MemeAnimator(live_portrait_path=None, output_dir=tmp_path)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.LAUGH,
            fps=24,
            resolution=(64, 64),
        )
        result = animator.generate(req)
        assert result.fps == 24
        assert result.duration_ms == pytest.approx(result.frame_count * (1000 / 24))
        assert len(result.keyframes) > 0


# ── ToonCrafterWrapper (fallback mode) ────────────────────────────────────────

class TestToonCrafterFallback:
    def test_uses_fallback_without_weights(self):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None)
        assert wrapper._use_fallback is True

    def test_interpolate_returns_correct_count(self, sample_bgr):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None, num_frames=4)
        end = np.full_like(sample_bgr, 128)
        frames = wrapper.interpolate(sample_bgr, end, num_frames=4)
        # start + 4 interp + end = 6
        assert len(frames) == 6

    def test_interpolate_frames_are_bgr_uint8(self, sample_bgr):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None)
        end = np.full_like(sample_bgr, 200)
        frames = wrapper.interpolate(sample_bgr, end, num_frames=3)
        for f in frames:
            assert f.dtype == np.uint8
            assert f.shape == sample_bgr.shape

    def test_interpolate_first_last_match_inputs(self, sample_bgr):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None)
        end = np.full_like(sample_bgr, 200)
        frames = wrapper.interpolate(sample_bgr, end, num_frames=3)
        assert np.array_equal(frames[0], sample_bgr)
        assert np.array_equal(frames[-1], end)

    def test_interpolate_values_monotone(self, sample_bgr):
        """Middle frames should be between start and end pixel values."""
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        # Use uniform images so mean is well-defined
        start = np.full((64, 64, 3), 50, dtype=np.uint8)
        end   = np.full((64, 64, 3), 200, dtype=np.uint8)
        wrapper = ToonCrafterWrapper(model_path=None)
        frames = wrapper.interpolate(start, end, num_frames=5)
        means = [f.mean() for f in frames]
        # Each frame mean should be >= previous (monotone increasing)
        for i in range(1, len(means)):
            assert means[i] >= means[i - 1] - 1.0  # allow tiny float rounding

    def test_interpolate_num_frames_zero(self, sample_bgr):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None)
        end = np.full_like(sample_bgr, 100)
        frames = wrapper.interpolate(sample_bgr, end, num_frames=0)
        assert len(frames) == 2  # just start and end

    def test_smooth_sequence_length(self, sample_bgr):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None)
        kf1 = sample_bgr.copy()
        kf2 = np.full_like(sample_bgr, 100)
        kf3 = np.full_like(sample_bgr, 200)
        result = wrapper.smooth_sequence([kf1, kf2, kf3], frames_between=3)
        # 2 gaps × (3 interp + 1 boundary) + final frame = 2×4 + 1 = 9
        assert len(result) == 9

    def test_smooth_sequence_single_frame(self, sample_bgr):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None)
        result = wrapper.smooth_sequence([sample_bgr])
        assert len(result) == 1
        assert np.array_equal(result[0], sample_bgr)

    def test_smooth_sequence_mismatched_sizes(self):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        wrapper = ToonCrafterWrapper(model_path=None)
        kf1 = np.zeros((64, 64, 3), dtype=np.uint8)
        kf2 = np.zeros((128, 128, 3), dtype=np.uint8)  # different size
        # Should not raise — end frame is resized to match start
        frames = wrapper.interpolate(kf1, kf2, num_frames=2)
        assert all(f.shape == kf1.shape for f in frames)

    def test_weights_present_uses_real_backend(self, tmp_path):
        from src.modules.meme_animator.toon_crafter import ToonCrafterWrapper
        # Create fake checkpoint files
        (tmp_path / "model.ckpt").write_bytes(b"fake")
        (tmp_path / "config.yaml").write_text("fake: true")
        wrapper = ToonCrafterWrapper(model_path=tmp_path)
        assert wrapper._use_fallback is False


# ── LivePortrait real inference path (mocked pipeline) ───────────────────────

class TestLivePortraitRealPath:
    """
    Tests for the real inference path in LivePortraitWrapper.
    All tests mock the official LivePortrait pipeline — no weights required.
    """

    # ── Fixtures ──────────────────────────────────────────────────────────────

    @pytest.fixture()
    def mock_pipeline(self):
        """
        Mock that mimics the LivePortrait pipeline API:
          - get_kp_info(tensor) → kp_info dict
          - execute_portraits(...) → {"out": tensor}
        """
        import torch
        from unittest.mock import MagicMock

        kp_info = {
            "pitch": torch.zeros(1, 1),
            "yaw":   torch.zeros(1, 1),
            "roll":  torch.zeros(1, 1),
            "t":     torch.zeros(1, 3),
            "exp":   torch.zeros(1, 63),
            "scale": torch.ones(1, 1),
            "kp":    torch.zeros(1, 21, 3),
        }

        mock = MagicMock()
        mock.get_kp_info.return_value = kp_info
        # execute_portraits returns a dict with "out" key: (1,3,H,W) in [0,1]
        mock.execute_portraits.return_value = {
            "out": torch.rand(1, 3, 64, 64)
        }
        # Make parameters() return an iterator with one dummy param so
        # _preprocess_source can detect the device
        dummy_param = torch.nn.Parameter(torch.zeros(1))
        mock.parameters.return_value = iter([dummy_param])
        return mock

    @pytest.fixture()
    def wrapper_with_mock(self, mock_pipeline, tmp_path):
        """
        LivePortraitWrapper in real-path mode with mock pipeline injected.
        Creates fake weight files so _should_use_fallback() returns False,
        then injects the mock into the registry.
        """
        from src.core.model_registry import ModelState

        # Create the expected weight directory structure
        weights = tmp_path / "pretrained_weights"
        for rel in [
            "liveportrait/base_models/appearance_feature_extractor.pth",
            "liveportrait/base_models/motion_extractor.pth",
            "liveportrait/base_models/warping_module.pth",
            "liveportrait/base_models/spade_generator.pth",
            "liveportrait/retargeting_models/stitching_retargeting_module.pth",
        ]:
            p = weights / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake")

        wrapper = LivePortraitWrapper(model_path=tmp_path)
        assert wrapper._use_fallback is False, "Should use real path with fake weights"

        # Inject mock pipeline into registry (bypass actual loading)
        entry = wrapper._registry._entries.get(LivePortraitWrapper.MODEL_ID)
        assert entry is not None
        entry.module = mock_pipeline
        entry.state = ModelState.ON_CPU

        return wrapper

    # ── MEME_MOTION_TEMPLATES tests ───────────────────────────────────────────

    def test_all_required_templates_exist(self):
        from src.modules.meme_animator.live_portrait import MEME_MOTION_TEMPLATES
        required = {"SHOCK_EXTREME", "LAUGH_EXTREME", "RAGE_EXTREME", "CRY_EXTREME", "SMUG_EXTREME"}
        assert required.issubset(set(MEME_MOTION_TEMPLATES.keys()))

    def test_template_exp_delta_shape(self):
        from src.modules.meme_animator.live_portrait import MEME_MOTION_TEMPLATES
        for name, tmpl in MEME_MOTION_TEMPLATES.items():
            assert "exp_delta" in tmpl, f"Missing exp_delta in {name}"
            assert tmpl["exp_delta"].shape == (63,), f"Wrong shape in {name}"
            assert tmpl["exp_delta"].dtype == np.float32, f"Wrong dtype in {name}"

    def test_template_has_required_keys(self):
        from src.modules.meme_animator.live_portrait import MEME_MOTION_TEMPLATES
        for name, tmpl in MEME_MOTION_TEMPLATES.items():
            assert "exp_delta" in tmpl,    f"Missing exp_delta in {name}"
            assert "scale_delta" in tmpl,  f"Missing scale_delta in {name}"
            assert "roll_delta" in tmpl,   f"Missing roll_delta in {name}"

    def test_shock_template_has_eye_bulge(self):
        from src.modules.meme_animator.live_portrait import (
            MEME_MOTION_TEMPLATES, _EYE_LEFT_Z, _EYE_RIGHT_Z
        )
        shock = MEME_MOTION_TEMPLATES["SHOCK_EXTREME"]["exp_delta"]
        eye_dims = _EYE_LEFT_Z + _EYE_RIGHT_Z
        assert any(abs(shock[i]) > 0.3 for i in eye_dims), \
            "SHOCK_EXTREME should have significant eye-bulge in z-dims"

    def test_shock_template_has_jaw_drop(self):
        from src.modules.meme_animator.live_portrait import (
            MEME_MOTION_TEMPLATES, _JAW_Y
        )
        shock = MEME_MOTION_TEMPLATES["SHOCK_EXTREME"]["exp_delta"]
        assert any(abs(shock[i]) > 0.5 for i in _JAW_Y), \
            "SHOCK_EXTREME should have significant jaw drop in y-dims"

    def test_laugh_template_has_eye_squint(self):
        from src.modules.meme_animator.live_portrait import (
            MEME_MOTION_TEMPLATES, _EYE_LEFT_Y, _EYE_RIGHT_Y
        )
        laugh = MEME_MOTION_TEMPLATES["LAUGH_EXTREME"]["exp_delta"]
        eye_dims = _EYE_LEFT_Y + _EYE_RIGHT_Y
        assert any(abs(laugh[i]) > 0.2 for i in eye_dims), \
            "LAUGH_EXTREME should have eye squint in y-dims"

    def test_rage_template_has_furrowed_brow(self):
        from src.modules.meme_animator.live_portrait import (
            MEME_MOTION_TEMPLATES, _BROW_Y
        )
        rage = MEME_MOTION_TEMPLATES["RAGE_EXTREME"]["exp_delta"]
        assert any(abs(rage[i]) > 0.3 for i in _BROW_Y), \
            "RAGE_EXTREME should have furrowed brow in y-dims"

    def test_templates_are_distinct(self):
        from src.modules.meme_animator.live_portrait import MEME_MOTION_TEMPLATES
        names = list(MEME_MOTION_TEMPLATES.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = MEME_MOTION_TEMPLATES[names[i]]["exp_delta"]
                b = MEME_MOTION_TEMPLATES[names[j]]["exp_delta"]
                assert not np.allclose(a, b), \
                    f"Templates {names[i]} and {names[j]} are identical"

    # ── _squash_params_to_exp_delta tests ─────────────────────────────────────

    def test_neutral_params_produce_near_zero_delta(self):
        neutral = SquashParams()  # all defaults = neutral
        delta = LivePortraitWrapper._squash_params_to_exp_delta(neutral)
        assert delta.shape == (63,)
        assert np.abs(delta).max() < 0.05, \
            "Neutral SquashParams should produce near-zero exp delta"

    def test_eye_bulge_affects_correct_dims(self):
        from src.modules.meme_animator.live_portrait import _EYE_LEFT_Z, _EYE_RIGHT_Z
        params = SquashParams(eye_bulge_scale=2.8)
        delta = LivePortraitWrapper._squash_params_to_exp_delta(params)
        eye_dims = set(_EYE_LEFT_Z + _EYE_RIGHT_Z)
        other_dims = [i for i in range(63) if i not in eye_dims]
        # Eye dims should be non-zero
        assert any(abs(delta[i]) > 0.1 for i in eye_dims)
        # Non-eye dims should be near zero (only eye_bulge changed)
        assert all(abs(delta[i]) < 0.05 for i in other_dims)

    def test_jaw_drop_affects_correct_dims(self):
        from src.modules.meme_animator.live_portrait import _JAW_Y
        params = SquashParams(jaw_drop_scale=2.2)
        delta = LivePortraitWrapper._squash_params_to_exp_delta(params)
        assert any(abs(delta[i]) > 0.2 for i in _JAW_Y)

    def test_brow_raise_affects_correct_dims(self):
        from src.modules.meme_animator.live_portrait import _BROW_Y
        params = SquashParams(brow_raise_offset=0.9)
        delta = LivePortraitWrapper._squash_params_to_exp_delta(params)
        assert any(abs(delta[i]) > 0.1 for i in _BROW_Y)

    def test_mouth_width_affects_correct_dims(self):
        from src.modules.meme_animator.live_portrait import _MOUTH_X
        params = SquashParams(mouth_width_scale=2.0)
        delta = LivePortraitWrapper._squash_params_to_exp_delta(params)
        assert any(abs(delta[i]) > 0.1 for i in _MOUTH_X)

    def test_delta_dtype_is_float32(self):
        params = SquashParams(eye_bulge_scale=2.0, jaw_drop_scale=1.8)
        delta = LivePortraitWrapper._squash_params_to_exp_delta(params)
        assert delta.dtype == np.float32

    def test_delta_shape_is_63(self):
        params = SquashParams()
        delta = LivePortraitWrapper._squash_params_to_exp_delta(params)
        assert delta.shape == (63,)

    # ── _build_driving_info tests ─────────────────────────────────────────────

    def test_build_driving_info_modifies_exp(self):
        import torch
        kp_source = {
            "pitch": torch.zeros(1, 1),
            "yaw":   torch.zeros(1, 1),
            "roll":  torch.zeros(1, 1),
            "t":     torch.zeros(1, 3),
            "exp":   torch.zeros(1, 63),
            "scale": torch.ones(1, 1),
            "kp":    torch.zeros(1, 21, 3),
        }
        params = SquashParams(eye_bulge_scale=2.8, jaw_drop_scale=2.0)
        x_d_info = LivePortraitWrapper._build_driving_info(kp_source, params)
        # exp should be modified
        assert not torch.allclose(x_d_info["exp"], kp_source["exp"])

    def test_build_driving_info_does_not_mutate_source(self):
        import torch
        kp_source = {
            "pitch": torch.zeros(1, 1),
            "yaw":   torch.zeros(1, 1),
            "roll":  torch.zeros(1, 1),
            "t":     torch.zeros(1, 3),
            "exp":   torch.zeros(1, 63),
            "scale": torch.ones(1, 1),
            "kp":    torch.zeros(1, 21, 3),
        }
        original_exp = kp_source["exp"].clone()
        params = SquashParams(eye_bulge_scale=2.8)
        LivePortraitWrapper._build_driving_info(kp_source, params)
        # Source should be unchanged
        assert torch.allclose(kp_source["exp"], original_exp)

    def test_build_driving_info_roll_modified_by_tilt(self):
        import torch
        kp_source = {
            "pitch": torch.zeros(1, 1),
            "yaw":   torch.zeros(1, 1),
            "roll":  torch.zeros(1, 1),
            "t":     torch.zeros(1, 3),
            "exp":   torch.zeros(1, 63),
            "scale": torch.ones(1, 1),
            "kp":    torch.zeros(1, 21, 3),
        }
        params = SquashParams(head_tilt_deg=15.0)
        x_d_info = LivePortraitWrapper._build_driving_info(kp_source, params)
        assert not torch.allclose(x_d_info["roll"], kp_source["roll"])

    def test_build_driving_info_neutral_params_minimal_change(self):
        import torch
        kp_source = {
            "pitch": torch.zeros(1, 1),
            "yaw":   torch.zeros(1, 1),
            "roll":  torch.zeros(1, 1),
            "t":     torch.zeros(1, 3),
            "exp":   torch.zeros(1, 63),
            "scale": torch.ones(1, 1),
            "kp":    torch.zeros(1, 21, 3),
        }
        params = SquashParams()  # neutral
        x_d_info = LivePortraitWrapper._build_driving_info(kp_source, params)
        # With neutral params, exp delta should be near zero
        assert torch.abs(x_d_info["exp"]).max().item() < 0.05

    # ── Real inference path end-to-end (mocked pipeline) ─────────────────────

    def test_render_frame_calls_get_kp_info(self, wrapper_with_mock, mock_pipeline, sample_bgr):
        wrapper_with_mock.render_frame(sample_bgr, SquashParams(eye_bulge_scale=2.5))
        mock_pipeline.get_kp_info.assert_called_once()

    def test_render_frame_calls_execute_portraits(self, wrapper_with_mock, mock_pipeline, sample_bgr):
        wrapper_with_mock.render_frame(sample_bgr, SquashParams(jaw_drop_scale=2.0))
        mock_pipeline.execute_portraits.assert_called_once()

    def test_render_frame_returns_bgr_uint8(self, wrapper_with_mock, sample_bgr):
        result = wrapper_with_mock.render_frame(sample_bgr, SquashParams(eye_bulge_scale=2.5))
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3

    def test_render_sequence_calls_get_kp_info_once(self, wrapper_with_mock, mock_pipeline, sample_bgr):
        """Source keypoints should be extracted once and reused for all frames."""
        params_seq = [SquashParams(eye_bulge_scale=1.0 + i * 0.3) for i in range(5)]
        wrapper_with_mock.render_sequence(sample_bgr, params_seq)
        # get_kp_info called exactly once (source KP cached for the batch)
        assert mock_pipeline.get_kp_info.call_count == 1

    def test_render_sequence_calls_execute_portraits_per_frame(
        self, wrapper_with_mock, mock_pipeline, sample_bgr
    ):
        params_seq = [SquashParams() for _ in range(4)]
        wrapper_with_mock.render_sequence(sample_bgr, params_seq)
        assert mock_pipeline.execute_portraits.call_count == 4

    def test_render_sequence_returns_correct_count(self, wrapper_with_mock, sample_bgr):
        params_seq = [SquashParams(eye_bulge_scale=1.0 + i * 0.2) for i in range(6)]
        frames = wrapper_with_mock.render_sequence(sample_bgr, params_seq)
        assert len(frames) == 6

    def test_tensor_to_bgr_handles_minus1_to_1_range(self):
        import torch
        # Tensor in [-1, 1] range
        t = torch.full((1, 3, 32, 32), -0.5)
        result = LivePortraitWrapper._tensor_to_bgr(t)
        assert result.dtype == np.uint8
        assert result.shape == (32, 32, 3)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_tensor_to_bgr_handles_0_to_1_range(self):
        import torch
        t = torch.full((1, 3, 32, 32), 0.7)
        result = LivePortraitWrapper._tensor_to_bgr(t)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_weights_present_disables_fallback(self, tmp_path):
        """Fake weight files should cause _use_fallback to be False."""
        weights = tmp_path / "pretrained_weights"
        for rel in [
            "liveportrait/base_models/appearance_feature_extractor.pth",
            "liveportrait/base_models/motion_extractor.pth",
            "liveportrait/base_models/warping_module.pth",
            "liveportrait/base_models/spade_generator.pth",
            "liveportrait/retargeting_models/stitching_retargeting_module.pth",
        ]:
            p = weights / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake")
        wrapper = LivePortraitWrapper(model_path=tmp_path)
        assert wrapper._use_fallback is False

    def test_missing_one_weight_uses_fallback(self, tmp_path):
        """If any required weight is missing, fallback should be used."""
        weights = tmp_path / "pretrained_weights"
        # Only create 4 of the 5 required files
        for rel in [
            "liveportrait/base_models/appearance_feature_extractor.pth",
            "liveportrait/base_models/motion_extractor.pth",
            "liveportrait/base_models/warping_module.pth",
            "liveportrait/base_models/spade_generator.pth",
            # stitching_retargeting_module.pth intentionally missing
        ]:
            p = weights / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake")
        wrapper = LivePortraitWrapper(model_path=tmp_path)
        assert wrapper._use_fallback is True


# ── ToonCrafter wired into MemeAnimator ──────────────────────────────────────

class TestMemeAnimatorToonCrafter:
    """
    Tests for ToonCrafter inter-frame smoothing wired into MemeAnimator.
    All tests run in fallback mode (no GPU, no weights).
    """

    def test_use_toon_crafter_false_by_default(self, sample_image_path, tmp_path):
        """Default request should not invoke ToonCrafter."""
        animator = MemeAnimator(output_dir=tmp_path)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
        )
        result = animator.generate(req)
        assert "toon_crafter" not in result.backend_used

    def test_use_toon_crafter_true_adds_backend_label(self, sample_image_path, tmp_path):
        """When use_toon_crafter=True, backend_used should mention toon_crafter."""
        animator = MemeAnimator(output_dir=tmp_path)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=2,
        )
        result = animator.generate(req)
        assert "toon_crafter" in result.backend_used

    def test_toon_crafter_increases_frame_count(self, sample_image_path, tmp_path):
        """Smoothing should produce more frames than the raw LivePortrait sequence."""
        animator = MemeAnimator(output_dir=tmp_path)

        req_no_tc = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=False,
        )
        req_with_tc = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=3,
        )
        result_no_tc = animator.generate(req_no_tc)
        result_with_tc = animator.generate(req_with_tc)

        assert result_with_tc.frame_count > result_no_tc.frame_count

    def test_toon_crafter_output_file_exists(self, sample_image_path, tmp_path):
        """Output GIF should be created even when ToonCrafter is enabled."""
        animator = MemeAnimator(output_dir=tmp_path)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.LAUGH,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=2,
        )
        result = animator.generate(req)
        assert result.output_path.exists()
        assert result.output_path.stat().st_size > 0

    def test_toon_crafter_fallback_label_when_no_weights(self, sample_image_path, tmp_path):
        """Without ToonCrafter weights, fallback label should appear."""
        animator = MemeAnimator(toon_crafter_path=None, output_dir=tmp_path)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=2,
        )
        result = animator.generate(req)
        assert "toon_crafter_fallback" in result.backend_used

    def test_frames_between_field_validated(self, sample_image_path):
        """frames_between must be in [1, 16]."""
        with pytest.raises(Exception):
            AnimationRequest(
                source_image_path=sample_image_path,
                frames_between=0,
            )
        with pytest.raises(Exception):
            AnimationRequest(
                source_image_path=sample_image_path,
                frames_between=17,
            )

    def test_keyframe_params_index_safe_after_smoothing(self, sample_image_path, tmp_path):
        """KeyFrame.params should not raise IndexError after ToonCrafter expands frames."""
        animator = MemeAnimator(output_dir=tmp_path)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.RAGE,
            fps=12,
            resolution=(64, 64),
            use_toon_crafter=True,
            frames_between=4,
        )
        result = animator.generate(req)
        # All keyframes should have valid params
        for kf in result.keyframes:
            assert kf.params is not None


# ── IP features passed to LivePortraitWrapper ────────────────────────────────

class TestIPFeaturesWiring:
    """
    Tests for ip_image_embeds flowing from AnimationRequest → LivePortraitWrapper.
    All tests run in fallback mode — IP embeds are accepted but not used by the
    affine-warp fallback (no crash expected).
    """

    def test_ip_embeds_none_by_default(self, sample_image_path):
        req = AnimationRequest(source_image_path=sample_image_path)
        assert req.ip_image_embeds is None

    def test_ip_embeds_accepted_in_request(self, sample_image_path):
        embeds = np.zeros((1, 768), dtype=np.float32)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            ip_image_embeds=embeds,
        )
        assert req.ip_image_embeds is not None

    def test_generate_with_ip_embeds_does_not_crash(self, sample_image_path, tmp_path):
        """Passing ip_image_embeds should not raise in fallback mode."""
        animator = MemeAnimator(output_dir=tmp_path)
        embeds = np.random.randn(1, 768).astype(np.float32)
        req = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            ip_image_embeds=embeds,
        )
        result = animator.generate(req)
        assert result.frame_count > 0
        assert result.output_path.exists()

    def test_generate_with_ip_embeds_same_frame_count(self, sample_image_path, tmp_path):
        """IP embeds should not change the number of frames produced."""
        animator = MemeAnimator(output_dir=tmp_path)
        req_no_ip = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
        )
        req_with_ip = AnimationRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            ip_image_embeds=np.zeros((1, 768), dtype=np.float32),
        )
        r1 = animator.generate(req_no_ip)
        r2 = animator.generate(req_with_ip)
        assert r1.frame_count == r2.frame_count

    def test_prepare_ip_embeds_returns_none_for_none(self):
        """_prepare_ip_embeds(None, ...) should return None."""
        from unittest.mock import MagicMock
        result = LivePortraitWrapper._prepare_ip_embeds(None, MagicMock())
        assert result is None

    def test_prepare_ip_embeds_converts_numpy_to_tensor(self):
        """_prepare_ip_embeds should return a torch tensor with correct shape."""
        import torch
        from unittest.mock import MagicMock
        embeds = np.zeros((1, 768), dtype=np.float32)
        mock_pipeline = MagicMock()
        mock_pipeline.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        result = LivePortraitWrapper._prepare_ip_embeds(embeds, mock_pipeline)
        assert result is not None
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 768)

    def test_prepare_ip_embeds_adds_batch_dim_for_1d(self):
        """1-D embedding should be unsqueezed to (1, D)."""
        import torch
        from unittest.mock import MagicMock
        embeds = np.zeros(768, dtype=np.float32)
        mock_pipeline = MagicMock()
        mock_pipeline.parameters.return_value = iter([torch.nn.Parameter(torch.zeros(1))])
        result = LivePortraitWrapper._prepare_ip_embeds(embeds, mock_pipeline)
        assert result.shape == (1, 768)

    def test_render_sequence_with_ip_embeds_real_path(
        self, tmp_path, sample_bgr
    ):
        """Real-path render_sequence should pass ip_tensor to _infer_with_kp."""
        import torch
        from unittest.mock import MagicMock, patch
        from src.core.model_registry import ModelState

        # Create fake weight files
        weights = tmp_path / "pretrained_weights"
        for rel in [
            "liveportrait/base_models/appearance_feature_extractor.pth",
            "liveportrait/base_models/motion_extractor.pth",
            "liveportrait/base_models/warping_module.pth",
            "liveportrait/base_models/spade_generator.pth",
            "liveportrait/retargeting_models/stitching_retargeting_module.pth",
        ]:
            p = weights / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"fake")

        kp_info = {
            "pitch": torch.zeros(1, 1), "yaw": torch.zeros(1, 1),
            "roll":  torch.zeros(1, 1), "t":   torch.zeros(1, 3),
            "exp":   torch.zeros(1, 63), "scale": torch.ones(1, 1),
            "kp":    torch.zeros(1, 21, 3),
        }
        mock_pipeline = MagicMock()
        mock_pipeline.get_kp_info.return_value = kp_info
        mock_pipeline.execute_portraits.return_value = {"out": torch.rand(1, 3, 64, 64)}
        mock_pipeline.parameters.return_value = iter(
            [torch.nn.Parameter(torch.zeros(1))]
        )

        wrapper = LivePortraitWrapper(model_path=tmp_path)
        entry = wrapper._registry._entries.get(LivePortraitWrapper.MODEL_ID)
        entry.module = mock_pipeline
        entry.state = ModelState.ON_CPU

        embeds = np.zeros((1, 768), dtype=np.float32)
        wrapper.render_sequence(sample_bgr, [SquashParams()], ip_image_embeds=embeds)

        # execute_portraits should have been called with ip_adapter_embeds kwarg
        call_kwargs = mock_pipeline.execute_portraits.call_args.kwargs
        assert "ip_adapter_embeds" in call_kwargs
        assert call_kwargs["ip_adapter_embeds"] is not None
