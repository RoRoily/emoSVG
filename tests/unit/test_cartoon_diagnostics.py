from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.modules.cartoon_analyzer import CartoonFaceAnalyzer
from src.modules.cartoon_evaluator import evaluate_animation_frames
from src.modules.cartoon_layer_parser import CartoonLayerParser
from src.modules.cartoon_rig import CartoonRigAnimator, CartoonRigBuilder
from src.modules.meme_animator.animator import MemeAnimator
from src.modules.meme_animator.motion_designer import MotionDesigner
from src.modules.meme_animator.schemas import AnimationBackend, AnimationRequest, MemeExpression
from src.pipeline.debug_artifacts import PipelineDebugWriter


def _make_chibi_face(path: Path) -> Path:
    img = np.full((160, 160, 3), 255, dtype=np.uint8)
    cv2.circle(img, (80, 78), 62, (232, 225, 214), -1)
    cv2.ellipse(img, (55, 68), (15, 20), 0, 0, 360, (170, 110, 220), -1)
    cv2.ellipse(img, (105, 68), (15, 20), 0, 0, 360, (170, 110, 220), -1)
    cv2.circle(img, (50, 62), 4, (255, 255, 255), -1)
    cv2.circle(img, (100, 62), 4, (255, 255, 255), -1)
    cv2.ellipse(img, (80, 103), (10, 13), 0, 0, 180, (75, 65, 180), 2)
    cv2.imwrite(str(path), img)
    return path


def test_cartoon_face_analyzer_detects_eye_order(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    result = CartoonFaceAnalyzer().analyze(image_path)
    geom = result.geometry

    assert geom.left_eye_center.x < geom.right_eye_center.x
    assert geom.mouth_center.y > geom.left_eye_center.y
    assert geom.confidence > 0.3


def test_cartoon_layer_parser_outputs_rig_parts(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    analyzer = CartoonFaceAnalyzer()
    analysis = analyzer.analyze(image_path)
    layers = CartoonLayerParser(analyzer=analyzer).parse(image_path, analysis=analysis)

    names = {layer.name for layer in layers.layers}
    assert {"foreground", "face_base_clean", "left_eye", "right_eye", "mouth"} <= names
    for layer in layers.layers:
        assert layer.rgba.shape[:2] == (160, 160)
        assert layer.rgba.shape[2] == 4
        assert layer.mask.dtype == np.uint8


def test_pipeline_debug_writer_creates_artifacts(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    bgr = cv2.imread(str(image_path))
    analyzer = CartoonFaceAnalyzer()
    analysis = analyzer.analyze(image_path)
    layers = CartoonLayerParser(analyzer=analyzer).parse(image_path, analysis=analysis)

    writer = PipelineDebugWriter(tmp_path, "chibi", "shock")
    writer.write_input(bgr)
    writer.write_analysis_overlay(bgr, analysis)
    writer.write_layer_overlay(bgr, layers)
    writer.write_layers(layers)
    writer.update_metrics(cartoon_analysis=analysis.to_dict(), cartoon_layers=layers.to_dict())
    writer.write_metrics()

    assert (writer.debug_dir / "input.png").exists()
    assert (writer.debug_dir / "landmark_overlay.png").exists()
    assert (writer.debug_dir / "masks_overlay.png").exists()
    assert (writer.debug_dir / "layers" / "left_eye.png").exists()
    assert (writer.debug_dir / "metrics.json").exists()


def test_cartoon_rig_animator_produces_motion(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    bgr = cv2.imread(str(image_path))
    analyzer = CartoonFaceAnalyzer()
    analysis = analyzer.analyze(image_path)
    layers = CartoonLayerParser(analyzer=analyzer).parse(image_path, analysis=analysis)
    params = MotionDesigner().design(MemeExpression.SHOCK)

    frames = CartoonRigAnimator().render_sequence(
        bgr,
        params,
        analysis=analysis,
        layers=layers,
    )

    assert len(frames) == len(params)
    assert frames[0].shape == bgr.shape
    assert not np.array_equal(frames[0], frames[len(frames) // 2])


def test_cartoon_rig_builder_has_editable_parts(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    bgr = cv2.imread(str(image_path))
    analyzer = CartoonFaceAnalyzer()
    analysis = analyzer.analyze(image_path)
    layers = CartoonLayerParser(analyzer=analyzer).parse(image_path, analysis=analysis)

    asset = CartoonRigBuilder().build(bgr, analysis=analysis, layers=layers)

    assert asset.get_layer("left_eye") is not None
    assert asset.get_layer("right_eye") is not None
    assert asset.get_layer("mouth") is not None


def test_evaluate_animation_frames_returns_metrics(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    bgr = cv2.imread(str(image_path))
    shifted = np.roll(bgr, shift=2, axis=1)

    metrics = evaluate_animation_frames([bgr, shifted, bgr], source_bgr=bgr)

    assert metrics["frame_count"] == 3.0
    assert metrics["temporal_mean_absdiff"] >= 0.0
    assert 0.0 <= metrics["source_similarity_min"] <= 1.0


def test_meme_animator_cartoon_backend(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    animator = MemeAnimator(output_dir=tmp_path)

    result = animator.generate(
        AnimationRequest(
            source_image_path=image_path,
            expression=MemeExpression.SHOCK,
            animation_backend=AnimationBackend.CARTOON_RIG,
            output_format="gif",
            fps=12,
            resolution=(128, 128),
        )
    )

    assert result.output_path.exists()
    assert result.backend_used == "cartoon_rig"
    assert result.metrics["frame_count"] == float(result.frame_count)


def test_meme_animator_cartoon_backend_with_interpolator_fallback(tmp_path: Path):
    image_path = _make_chibi_face(tmp_path / "chibi.png")
    animator = MemeAnimator(output_dir=tmp_path, toon_crafter_path=None)

    result = animator.generate(
        AnimationRequest(
            source_image_path=image_path,
            expression=MemeExpression.SHOCK,
            animation_backend=AnimationBackend.CARTOON_RIG,
            output_format="gif",
            fps=12,
            resolution=(128, 128),
            use_toon_crafter=True,
            frames_between=1,
        )
    )

    assert result.output_path.exists()
    assert "cartoon_rig+toon_crafter_fallback[rig_post]" == result.backend_used
    assert result.frame_count > len(MotionDesigner().design(MemeExpression.SHOCK))
