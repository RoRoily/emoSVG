from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np

from src.modules.cartoon_analyzer import (
    AnimeFaceDetectorAdapter,
    CartoonFaceAnalyzer,
    ExternalAnimeFaceDetectorAdapter,
)
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
    result = CartoonFaceAnalyzer(backend="heuristic").analyze(image_path)
    geom = result.geometry

    assert geom.left_eye_center.x < geom.right_eye_center.x
    assert geom.mouth_center.y > geom.left_eye_center.y
    assert geom.confidence > 0.3


def test_anime_face_detector_adapter_maps_landmarks(monkeypatch):
    keypoints = np.zeros((28, 3), dtype=np.float32)
    keypoints[:, 0] = 80
    keypoints[:, 1] = 80
    keypoints[:, 2] = 0.9
    keypoints[11:17, :2] = np.array(
        [[45, 62], [52, 58], [62, 61], [64, 72], [54, 78], [45, 72]],
        dtype=np.float32,
    )
    keypoints[17:23, :2] = np.array(
        [[96, 62], [104, 58], [114, 61], [116, 72], [106, 78], [96, 72]],
        dtype=np.float32,
    )
    keypoints[23:28, :2] = np.array(
        [[73, 108], [80, 104], [88, 108], [86, 116], [76, 116]],
        dtype=np.float32,
    )

    def fake_create_detector(detector_name, device):
        def detector(_image):
            return [{"bbox": np.array([20, 15, 140, 145, 0.95]), "keypoints": keypoints}]

        return detector

    monkeypatch.setitem(
        sys.modules,
        "anime_face_detector",
        types.SimpleNamespace(create_detector=fake_create_detector),
    )

    image = np.full((160, 160, 3), 255, dtype=np.uint8)
    mask = np.full((160, 160), 255, dtype=np.uint8)
    result = AnimeFaceDetectorAdapter(device="cpu").analyze_bgr(
        image,
        foreground_mask=mask,
        foreground_bbox=CartoonFaceAnalyzer._mask_bbox(mask),
    )
    geom = result.geometry

    assert result.backend_used == "anime_face_detector"
    assert geom.left_eye_center.x < geom.right_eye_center.x
    assert geom.mouth_center.y > geom.left_eye_center.y
    assert geom.confidence > 0.8


def test_external_anime_face_detector_adapter_maps_json(tmp_path: Path):
    fake_script = tmp_path / "fake_anime_face_detect.py"
    fake_script.write_text(
        """
import json

keypoints = [[80, 80, 0.9] for _ in range(28)]
for idx, point in zip(range(11, 17), [[45, 62], [52, 58], [62, 61], [64, 72], [54, 78], [45, 72]]):
    keypoints[idx][:2] = point
for idx, point in zip(range(17, 23), [[96, 62], [104, 58], [114, 61], [116, 72], [106, 78], [96, 72]]):
    keypoints[idx][:2] = point
for idx, point in zip(range(23, 28), [[73, 108], [80, 104], [88, 108], [86, 116], [76, 116]]):
    keypoints[idx][:2] = point
print(json.dumps({"faces": [{"bbox": [20, 15, 140, 145, 0.95], "keypoints": keypoints}]}))
""".strip(),
        encoding="utf-8",
    )

    image = np.full((160, 160, 3), 255, dtype=np.uint8)
    mask = np.full((160, 160), 255, dtype=np.uint8)
    result = ExternalAnimeFaceDetectorAdapter(
        python_executable=sys.executable,
        script_path=fake_script,
        device="cpu",
        timeout_seconds=10,
    ).analyze_bgr(
        image,
        foreground_mask=mask,
        foreground_bbox=CartoonFaceAnalyzer._mask_bbox(mask),
    )
    geom = result.geometry

    assert result.backend_used == "external_anime_face_detector"
    assert geom.left_eye_center.x < geom.right_eye_center.x
    assert geom.mouth_center.y > geom.left_eye_center.y
    assert geom.confidence > 0.8


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
