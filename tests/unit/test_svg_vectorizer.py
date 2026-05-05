"""
Unit tests for svg_vectorizer module.
No GPU, no SAM weights — uses contour-detection fallback throughout.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.model_registry import ModelRegistry
from src.modules.svg_vectorizer.schemas import VectorizationRequest, SVGLayer
from src.modules.svg_vectorizer.segmentor import Segmentor
from src.modules.svg_vectorizer.bezier_fitter import BezierFitter
from src.modules.svg_vectorizer.svg_builder import SVGBuilder
from src.modules.svg_vectorizer.vectorizer import SVGVectorizer


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    """256x256 image with two distinct colour regions."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :] = (200, 100, 50)   # top half: blue-ish
    img[128:, :] = (50, 200, 100)   # bottom half: green-ish
    p = tmp_path / "sample.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture()
def sample_bgr() -> np.ndarray:
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:128, :] = (200, 100, 50)
    img[128:, :] = (50, 200, 100)
    return img


@pytest.fixture()
def circle_mask() -> np.ndarray:
    """256x256 boolean mask with a filled circle."""
    mask = np.zeros((256, 256), dtype=bool)
    cv2.circle(
        mask.view(np.uint8), (128, 128), 80, 1, -1
    )
    return mask.astype(bool)


# ── Schema validation ─────────────────────────────────────────────────────

class TestVectorizationRequest:
    def test_valid_request(self, sample_image_path):
        req = VectorizationRequest(source_image_path=sample_image_path)
        assert req.bezier_tolerance == pytest.approx(1.5)
        assert req.layer_naming == "semantic"

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(Exception):
            VectorizationRequest(source_image_path=tmp_path / "nope.png")

    def test_invalid_layer_naming_raises(self, sample_image_path):
        with pytest.raises(Exception):
            VectorizationRequest(
                source_image_path=sample_image_path,
                layer_naming="invalid",
            )

    def test_custom_output_path(self, sample_image_path, tmp_path):
        out = tmp_path / "out.svg"
        req = VectorizationRequest(
            source_image_path=sample_image_path,
            output_path=out,
        )
        assert req.output_path == out


# ── Segmentor (contour fallback) ──────────────────────────────────────────

class TestSegmentor:
    def test_uses_fallback_without_sam(self):
        seg = Segmentor(sam_checkpoint=None)
        assert seg._use_fallback is True

    def test_segment_returns_masks(self, sample_bgr):
        seg = Segmentor(sam_checkpoint=None)
        masks = seg.segment(sample_bgr, min_area=100)
        assert len(masks) > 0

    def test_masks_are_boolean(self, sample_bgr):
        seg = Segmentor(sam_checkpoint=None)
        masks = seg.segment(sample_bgr, min_area=100)
        for m in masks:
            assert m.dtype == bool

    def test_masks_match_image_shape(self, sample_bgr):
        seg = Segmentor(sam_checkpoint=None)
        masks = seg.segment(sample_bgr, min_area=100)
        h, w = sample_bgr.shape[:2]
        for m in masks:
            assert m.shape == (h, w)

    def test_masks_sorted_by_area_descending(self, sample_bgr):
        seg = Segmentor(sam_checkpoint=None)
        masks = seg.segment(sample_bgr, min_area=100)
        areas = [m.sum() for m in masks]
        assert areas == sorted(areas, reverse=True)

    def test_min_area_filter(self, sample_bgr):
        seg = Segmentor(sam_checkpoint=None)
        masks_strict = seg.segment(sample_bgr, min_area=5000)
        masks_loose  = seg.segment(sample_bgr, min_area=10)
        assert len(masks_strict) <= len(masks_loose)

    def test_small_image_does_not_crash(self):
        seg = Segmentor(sam_checkpoint=None)
        tiny = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        masks = seg.segment(tiny, min_area=1)
        assert isinstance(masks, list)


# ── BezierFitter ─────────────────────────────────────────────────────────

class TestBezierFitter:
    def test_circle_mask_produces_paths(self, circle_mask):
        fitter = BezierFitter(tolerance=2.0)
        paths = fitter.mask_to_paths(circle_mask)
        assert len(paths) > 0

    def test_path_starts_with_M(self, circle_mask):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        for p in paths:
            assert p.startswith("M ")

    def test_path_ends_with_Z(self, circle_mask):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        for p in paths:
            assert p.endswith("Z")

    def test_path_contains_C_commands(self, circle_mask):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        assert any("C " in p for p in paths)

    def test_empty_mask_returns_no_paths(self):
        fitter = BezierFitter()
        empty = np.zeros((64, 64), dtype=bool)
        paths = fitter.mask_to_paths(empty)
        assert paths == []

    def test_precision_respected(self, circle_mask):
        fitter = BezierFitter(precision=0)
        paths = fitter.mask_to_paths(circle_mask)
        # With precision=0 there should be no decimal points in coordinates
        for p in paths:
            # Strip command letters and check numeric parts
            nums = [tok for tok in p.replace(",", " ").split()
                    if tok not in ("M", "C", "Z")]
            for n in nums:
                assert "." not in n, f"Expected integer coord, got {n!r}"

    def test_max_nodes_downsamples(self):
        # Create a large contour mask
        mask = np.zeros((512, 512), dtype=bool)
        cv2.circle(mask.view(np.uint8), (256, 256), 200, 1, -1)
        mask = mask.astype(bool)
        fitter = BezierFitter(max_nodes=50, tolerance=3.0)
        paths = fitter.mask_to_paths(mask)
        assert len(paths) > 0

    def test_tiny_contour_skipped(self):
        fitter = BezierFitter()
        mask = np.zeros((64, 64), dtype=bool)
        mask[30:33, 30:33] = True   # 3x3 square — too small for Bezier
        paths = fitter.mask_to_paths(mask)
        # Should either produce no paths or very short ones — must not crash
        assert isinstance(paths, list)


# ── SVGBuilder ────────────────────────────────────────────────────────────

class TestSVGBuilder:
    def test_build_creates_svg_file(self, sample_bgr, circle_mask, tmp_path):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        builder = SVGBuilder()
        out = tmp_path / "out.svg"
        layers = builder.build(sample_bgr, [circle_mask], [paths], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_svg_contains_path_element(self, sample_bgr, circle_mask, tmp_path):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        builder = SVGBuilder()
        out = tmp_path / "out.svg"
        builder.build(sample_bgr, [circle_mask], [paths], out)
        content = out.read_text(encoding="utf-8")
        assert "<path" in content

    def test_svg_contains_group_per_layer(self, sample_bgr, circle_mask, tmp_path):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        builder = SVGBuilder()
        out = tmp_path / "out.svg"
        layers = builder.build(sample_bgr, [circle_mask], [paths], out)
        content = out.read_text(encoding="utf-8")
        assert "layer-0" in content

    def test_returns_svg_layer_objects(self, sample_bgr, circle_mask, tmp_path):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        builder = SVGBuilder()
        out = tmp_path / "out.svg"
        layers = builder.build(sample_bgr, [circle_mask], [paths], out)
        assert len(layers) > 0
        assert isinstance(layers[0], SVGLayer)

    def test_fill_color_is_valid_hex(self, sample_bgr, circle_mask, tmp_path):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        builder = SVGBuilder()
        out = tmp_path / "out.svg"
        layers = builder.build(sample_bgr, [circle_mask], [paths], out)
        for layer in layers:
            assert layer.fill_color.startswith("#")
            assert len(layer.fill_color) == 7

    def test_index_layer_naming(self, sample_bgr, circle_mask, tmp_path):
        fitter = BezierFitter()
        paths = fitter.mask_to_paths(circle_mask)
        builder = SVGBuilder(layer_naming="index")
        out = tmp_path / "out.svg"
        layers = builder.build(sample_bgr, [circle_mask], [paths], out)
        assert layers[0].label == "layer-0"

    def test_empty_paths_layer_skipped(self, sample_bgr, circle_mask, tmp_path):
        builder = SVGBuilder()
        out = tmp_path / "out.svg"
        # Pass empty path list for the mask
        layers = builder.build(sample_bgr, [circle_mask], [[]], out)
        assert len(layers) == 0


# ── SVGVectorizer end-to-end ──────────────────────────────────────────────

class TestSVGVectorizerE2E:
    def test_vectorize_produces_svg(self, sample_image_path, tmp_path):
        vec = SVGVectorizer(sam_checkpoint=None, output_dir=tmp_path)
        req = VectorizationRequest(source_image_path=sample_image_path)
        result = vec.vectorize(req)
        assert result.output_path.exists()
        assert result.output_path.suffix == ".svg"

    def test_result_has_layers(self, sample_image_path, tmp_path):
        vec = SVGVectorizer(sam_checkpoint=None, output_dir=tmp_path)
        req = VectorizationRequest(source_image_path=sample_image_path)
        result = vec.vectorize(req)
        assert result.layer_count > 0
        assert result.total_paths > 0

    def test_backend_is_fallback(self, sample_image_path, tmp_path):
        vec = SVGVectorizer(sam_checkpoint=None, output_dir=tmp_path)
        req = VectorizationRequest(source_image_path=sample_image_path)
        result = vec.vectorize(req)
        assert result.backend_used == "contour_fallback"

    def test_custom_output_path(self, sample_image_path, tmp_path):
        vec = SVGVectorizer(sam_checkpoint=None, output_dir=tmp_path)
        out = tmp_path / "custom_output.svg"
        req = VectorizationRequest(
            source_image_path=sample_image_path,
            output_path=out,
        )
        result = vec.vectorize(req)
        assert result.output_path == out
        assert out.exists()

    def test_svg_is_valid_xml(self, sample_image_path, tmp_path):
        import xml.etree.ElementTree as ET
        vec = SVGVectorizer(sam_checkpoint=None, output_dir=tmp_path)
        req = VectorizationRequest(source_image_path=sample_image_path)
        result = vec.vectorize(req)
        # Should not raise
        tree = ET.parse(str(result.output_path))
        root = tree.getroot()
        assert "svg" in root.tag.lower()

    def test_index_layer_naming(self, sample_image_path, tmp_path):
        vec = SVGVectorizer(sam_checkpoint=None, output_dir=tmp_path)
        req = VectorizationRequest(
            source_image_path=sample_image_path,
            layer_naming="index",
        )
        result = vec.vectorize(req)
        assert all(l.label.startswith("layer-") for l in result.layers)
