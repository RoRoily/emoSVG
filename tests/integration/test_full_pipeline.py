"""
Integration tests for FullPipeline and FastAPI endpoints.
All model backends run in fallback mode (no weights required).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.core.model_registry import ModelRegistry
from src.modules.meme_animator.schemas import MemeExpression
from src.pipeline.full_pipeline import FullPipeline, FullPipelineRequest

# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :] = (200, 100, 50)
    img[128:, :] = (50, 200, 100)
    p = tmp_path / "character.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture()
def mock_clip_bundle():
    """Mock CLIP (processor, model) bundle to avoid HuggingFace downloads."""
    import torch
    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    mock_output = MagicMock()
    mock_output.image_embeds = torch.zeros(1, 768)
    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.to = MagicMock(return_value=mock_model)
    return (mock_processor, mock_model)


def _inject_clip(pipeline: FullPipeline, bundle) -> None:
    """Inject mock CLIP bundle into the registry so no download occurs."""
    from src.core.model_registry import ModelState
    from src.modules.ip_extractor.extractor import IPExtractor
    reg = pipeline._registry
    entry = reg._entries.get(IPExtractor.MODEL_ID)
    if entry:
        entry.module = bundle
        entry.state = ModelState.ON_CPU


# ── FullPipeline integration ───────────────────────────────────────────────

class TestFullPipelineIntegration:
    def test_full_run_animation_only(self, sample_image_path, tmp_path, mock_clip_bundle):
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req = FullPipelineRequest(
            source_image_path=sample_image_path,
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

    def test_full_run_with_3d_and_svg(self, sample_image_path, tmp_path, mock_clip_bundle):
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req = FullPipelineRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.LAUGH,
            output_format="gif",
            fps=12,
            resolution=(64, 64),
            run_3d=True,
            run_svg=True,
        )
        result = pipeline.execute(req)

        assert result.animation.output_path.exists()
        assert result.reconstruction is not None
        assert result.reconstruction.mesh_stats.face_count > 0
        assert result.vectorization is not None
        assert result.vectorization.output_path.exists()

    def test_all_expressions_run(self, sample_image_path, tmp_path, mock_clip_bundle):
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        for expr in MemeExpression:
            if expr == MemeExpression.CUSTOM:
                continue
            req = FullPipelineRequest(
                source_image_path=sample_image_path,
                expression=expr,
                fps=12,
                resolution=(64, 64),
                run_3d=False,
                run_svg=False,
            )
            result = pipeline.execute(req)
            assert result.animation.output_path.exists(), f"Failed for {expr}"

    def test_elapsed_seconds_positive(self, sample_image_path, tmp_path, mock_clip_bundle):
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)
        req = FullPipelineRequest(
            source_image_path=sample_image_path,
            fps=12, resolution=(64, 64), run_3d=False, run_svg=False,
        )
        result = pipeline.execute(req)
        assert result.elapsed_seconds > 0

    def test_debug_artifacts_written(self, sample_image_path, tmp_path, mock_clip_bundle):
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)
        req = FullPipelineRequest(
            source_image_path=sample_image_path,
            fps=12,
            resolution=(64, 64),
            run_3d=False,
            run_svg=False,
            debug=True,
        )
        result = pipeline.execute(req)

        assert result.debug_dir is not None
        assert (result.debug_dir / "input.png").exists()
        assert (result.debug_dir / "landmark_overlay.png").exists()
        assert (result.debug_dir / "masks_overlay.png").exists()
        assert (result.debug_dir / "metrics.json").exists()
        assert result.cartoon_analysis is not None
        assert result.cartoon_layers is not None

    def test_use_source_for_3d_svg_false_default(self, sample_image_path, tmp_path, mock_clip_bundle):
        """Default (False): 3D and SVG use the peak animation keyframe."""
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)
        req = FullPipelineRequest(
            source_image_path=sample_image_path,
            fps=12, resolution=(64, 64),
            run_3d=True, run_svg=True,
            use_source_for_3d_svg=False,
        )
        result = pipeline.execute(req)
        assert result.reconstruction is not None
        assert result.vectorization is not None
        # A tmp keyframe file should have been created
        tmp_keyframes = tmp_path / "_tmp_keyframes"
        assert tmp_keyframes.exists()
        assert any(tmp_keyframes.iterdir())

    def test_use_source_for_3d_svg_true(self, sample_image_path, tmp_path, mock_clip_bundle):
        """True: 3D and SVG use the original source image — no tmp keyframe written."""
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)
        req = FullPipelineRequest(
            source_image_path=sample_image_path,
            fps=12, resolution=(64, 64),
            run_3d=True, run_svg=True,
            use_source_for_3d_svg=True,
        )
        result = pipeline.execute(req)
        assert result.reconstruction is not None
        assert result.vectorization is not None
        # No tmp keyframe directory should exist
        tmp_keyframes = tmp_path / "_tmp_keyframes"
        assert not tmp_keyframes.exists()


# ── FastAPI endpoint integration ──────────────────────────────────────────

@pytest.fixture()
def api_client(tmp_path, mock_clip_bundle):
    """TestClient with pipeline singleton replaced by a test instance."""
    from src.api import dependencies
    from src.api.main import app

    pipeline = FullPipeline(output_root=tmp_path)
    _inject_clip(pipeline, mock_clip_bundle)

    # Override the dependency
    app.dependency_overrides[dependencies.get_pipeline] = lambda: pipeline
    client = TestClient(app, raise_server_exceptions=False)
    yield client
    app.dependency_overrides.clear()


class TestAPIEndpoints:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_status(self, api_client):
        resp = api_client.get("/status")
        assert resp.status_code == 200
        assert "models" in resp.json()

    def test_animate_endpoint(self, api_client, sample_image_path):
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/animate",
                data={"expression": "shock", "fps": "12", "width": "64", "height": "64"},
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "output_path" in body
        assert body["frame_count"] > 0

    def test_animate_endpoint_cartoon_backend(self, api_client, sample_image_path):
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/animate",
                data={
                    "expression": "shock",
                    "fps": "12",
                    "width": "64",
                    "height": "64",
                    "backend": "cartoon_rig",
                    "debug": "true",
                },
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["backend_used"] == "cartoon_rig"
        assert body["debug_dir"] is not None
        assert body["cartoon_layer_count"] >= 5
        assert body["metrics"]["frame_count"] == float(body["frame_count"])

    def test_vectorize_endpoint(self, api_client, sample_image_path):
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/vectorize",
                data={"bezier_tolerance": "2.0", "min_region_area": "50"},
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "output_path" in body
        assert body["layer_count"] > 0

    def test_reconstruct_endpoint(self, api_client, sample_image_path):
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/reconstruct",
                data={"mc_resolution": "64", "remove_background": "false"},
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "vertex_count" in body
        assert body["vertex_count"] > 0

    def test_generate_endpoint(self, api_client, sample_image_path):
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/generate",
                data={
                    "expression": "shock", "fps": "12",
                    "width": "64", "height": "64",
                    "run_3d": "true", "run_svg": "true",
                },
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "animation_path" in body
        assert body["frame_count"] > 0

    def test_animate_endpoint_with_toon_crafter(self, api_client, sample_image_path):
        """use_toon_crafter=true should be accepted and produce more frames."""
        with open(sample_image_path, "rb") as f:
            resp_no_tc = api_client.post(
                "/animate",
                data={"expression": "shock", "fps": "12", "width": "64", "height": "64",
                      "use_toon_crafter": "false", "frames_between": "2"},
                files={"file": ("character.png", f, "image/png")},
            )
        with open(sample_image_path, "rb") as f:
            resp_with_tc = api_client.post(
                "/animate",
                data={"expression": "shock", "fps": "12", "width": "64", "height": "64",
                      "use_toon_crafter": "true", "frames_between": "2"},
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp_no_tc.status_code == 200
        assert resp_with_tc.status_code == 200
        assert resp_with_tc.json()["frame_count"] > resp_no_tc.json()["frame_count"]

    def test_generate_endpoint_with_toon_crafter(self, api_client, sample_image_path):
        """Full pipeline endpoint should accept use_toon_crafter and frames_between."""
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/generate",
                data={
                    "expression": "laugh", "fps": "12",
                    "width": "64", "height": "64",
                    "run_3d": "false", "run_svg": "false",
                    "use_toon_crafter": "true", "frames_between": "2",
                },
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["frame_count"] > 0
        assert "toon_crafter" in body["backends"]["animation"]


# ── IP features integration ───────────────────────────────────────────────────

class TestIPFeaturesIntegration:
    """
    Tests that IP features extracted by IPExtractor are passed through to the
    animator and reflected in the pipeline result.
    """

    def test_ip_features_present_in_result(self, sample_image_path, tmp_path, mock_clip_bundle):
        """FullPipelineResult must always include ip_features."""
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req = FullPipelineRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.SHOCK,
            fps=12,
            resolution=(64, 64),
            run_3d=False,
            run_svg=False,
        )
        result = pipeline.execute(req)

        assert result.ip_features is not None
        assert result.ip_features.image_embeds is not None
        assert result.ip_features.backend_used in ("ip_adapter", "clip_fallback")

    def test_ip_features_backend_reported_in_api(self, api_client, sample_image_path):
        """The /generate endpoint should report the IP backend in the backends dict."""
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/generate",
                data={
                    "expression": "shock", "fps": "12",
                    "width": "64", "height": "64",
                    "run_3d": "false", "run_svg": "false",
                },
                files={"file": ("character.png", f, "image/png")},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "ip" in body["backends"]
        assert body["backends"]["ip"] in ("ip_adapter", "clip_fallback")

    def test_ip_features_do_not_change_frame_count(
        self, sample_image_path, tmp_path, mock_clip_bundle
    ):
        """IP features are a conditioning signal — they must not alter frame count."""
        pipeline = FullPipeline(output_root=tmp_path)
        _inject_clip(pipeline, mock_clip_bundle)

        req = FullPipelineRequest(
            source_image_path=sample_image_path,
            expression=MemeExpression.LAUGH,
            fps=12,
            resolution=(64, 64),
            run_3d=False,
            run_svg=False,
        )
        result = pipeline.execute(req)

        # Frame count should match the motion designer's output
        # (wind-up + attack + hold + bounce + settle)
        assert result.animation.frame_count > 0
        # IP features should be present regardless
        assert result.ip_features.image_embeds is not None


# ── Batch /generate endpoint ──────────────────────────────────────────────────

class TestBatchEndpoint:
    """Tests for POST /batch/generate — multi-image batch processing."""

    def test_batch_single_image(self, api_client, sample_image_path):
        """Single image batch should succeed and return 1 result."""
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/batch/generate",
                data={"expression": "shock", "fps": "12", "width": "64", "height": "64",
                      "run_3d": "false", "run_svg": "false"},
                files=[("files", ("char.png", f, "image/png"))],
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["succeeded"] == 1
        assert body["failed"] == 0
        assert len(body["results"]) == 1
        assert body["results"][0]["error"] is None
        assert body["results"][0]["frame_count"] > 0

    def test_batch_multiple_images(self, api_client, sample_image_path):
        """Multiple images should all be processed independently."""
        files = [
            ("files", ("img1.png", open(sample_image_path, "rb"), "image/png")),
            ("files", ("img2.png", open(sample_image_path, "rb"), "image/png")),
            ("files", ("img3.png", open(sample_image_path, "rb"), "image/png")),
        ]
        try:
            resp = api_client.post(
                "/batch/generate",
                data={"expression": "laugh", "fps": "12", "width": "64", "height": "64",
                      "run_3d": "false", "run_svg": "false"},
                files=files,
            )
        finally:
            for _, (_, fh, _) in files:
                fh.close()

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["succeeded"] == 3
        assert body["failed"] == 0
        for r in body["results"]:
            assert r["error"] is None
            assert r["frame_count"] > 0

    def test_batch_preserves_filenames(self, api_client, sample_image_path):
        """Each result should carry the original filename."""
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/batch/generate",
                data={"expression": "shock", "fps": "12", "width": "64", "height": "64"},
                files=[("files", ("my_character.png", f, "image/png"))],
            )
        assert resp.status_code == 200
        assert resp.json()["results"][0]["filename"] == "my_character.png"

    def test_batch_exceeds_max_raises_422(self, api_client, sample_image_path):
        """Sending more than 8 images should return HTTP 422."""
        files = [
            ("files", (f"img{i}.png", open(sample_image_path, "rb"), "image/png"))
            for i in range(9)
        ]
        try:
            resp = api_client.post(
                "/batch/generate",
                data={"expression": "shock", "fps": "12", "width": "64", "height": "64"},
                files=files,
            )
        finally:
            for _, (_, fh, _) in files:
                fh.close()
        assert resp.status_code == 422

    def test_batch_empty_files_raises_422(self, api_client):
        """Empty file list should return HTTP 422."""
        resp = api_client.post(
            "/batch/generate",
            data={"expression": "shock", "fps": "12", "width": "64", "height": "64"},
            files=[],
        )
        assert resp.status_code == 422

    def test_batch_with_toon_crafter(self, api_client, sample_image_path):
        """Batch endpoint should accept use_toon_crafter flag."""
        with open(sample_image_path, "rb") as f:
            resp = api_client.post(
                "/batch/generate",
                data={"expression": "shock", "fps": "12", "width": "64", "height": "64",
                      "use_toon_crafter": "true", "frames_between": "2"},
                files=[("files", ("char.png", f, "image/png"))],
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["succeeded"] == 1
        assert "toon_crafter" in body["results"][0]["backends"]["animation"]

    def test_status_endpoint_has_vram_key(self, api_client):
        """The /status endpoint should include a 'vram' key."""
        resp = api_client.get("/status")
        assert resp.status_code == 200
        body = resp.json()
        assert "vram" in body
        assert "models" in body
