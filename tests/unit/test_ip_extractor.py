"""
Unit tests for ip_extractor module.
No GPU, no IP-Adapter weights — uses HuggingFace CLIP fallback.
CLIP model download is mocked so tests run offline.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.core.model_registry import ModelRegistry
from src.modules.ip_extractor.schemas import ExtractionRequest, IPFeatures
from src.modules.ip_extractor.preprocessor import ImagePreprocessor, _CLIP_MEAN, _CLIP_STD
from src.modules.ip_extractor.extractor import IPExtractor


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    p = tmp_path / "sample.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture()
def sample_bgr() -> np.ndarray:
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


# ── Mock CLIP bundle ──────────────────────────────────────────────────────

def _make_mock_clip_bundle():
    """Return a (processor, model) mock that mimics HuggingFace CLIP output."""
    import torch

    mock_processor = MagicMock()
    mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    mock_output = MagicMock()
    mock_output.image_embeds = torch.zeros(1, 768)

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.to = MagicMock(return_value=mock_model)

    return (mock_processor, mock_model)


# ── Schema validation ─────────────────────────────────────────────────────

class TestExtractionRequest:
    def test_valid_request(self, sample_image_path):
        req = ExtractionRequest(source_image_path=sample_image_path)
        assert req.target_size == (224, 224)
        assert req.normalize is True

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(Exception):
            ExtractionRequest(source_image_path=tmp_path / "nope.png")

    def test_custom_target_size(self, sample_image_path):
        req = ExtractionRequest(
            source_image_path=sample_image_path,
            target_size=(336, 336),
        )
        assert req.target_size == (336, 336)


# ── ImagePreprocessor ─────────────────────────────────────────────────────

class TestImagePreprocessor:
    def test_output_shape(self, sample_bgr):
        proc = ImagePreprocessor(target_size=(224, 224))
        result = proc.process(sample_bgr, normalize=False)
        assert result.shape == (224, 224, 3)

    def test_output_dtype_float32(self, sample_bgr):
        proc = ImagePreprocessor()
        result = proc.process(sample_bgr, normalize=False)
        assert result.dtype == np.float32

    def test_output_range_without_normalize(self, sample_bgr):
        proc = ImagePreprocessor()
        result = proc.process(sample_bgr, normalize=False)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_shifts_values(self, sample_bgr):
        proc = ImagePreprocessor()
        unnorm = proc.process(sample_bgr, normalize=False)
        normed = proc.process(sample_bgr, normalize=True)
        # Normalised values can go below 0 or above 1
        assert not np.allclose(unnorm, normed)

    def test_output_is_rgb_not_bgr(self, sample_image_path):
        # Write a known-colour image: pure red in BGR = (0, 0, 255)
        img_bgr = np.zeros((64, 64, 3), dtype=np.uint8)
        img_bgr[:, :, 2] = 255   # red channel in BGR
        proc = ImagePreprocessor(target_size=(64, 64))
        result = proc.process(img_bgr, normalize=False)
        # In RGB the red channel is index 0
        assert result[:, :, 0].mean() > 0.9

    def test_process_from_path(self, sample_image_path):
        proc = ImagePreprocessor(target_size=(224, 224))
        result = proc.process(sample_image_path, normalize=False)
        assert result.shape == (224, 224, 3)

    def test_invalid_input_raises(self):
        proc = ImagePreprocessor()
        with pytest.raises(TypeError):
            proc.process(12345)

    def test_resize_to_target(self, sample_bgr):
        proc = ImagePreprocessor(target_size=(64, 64))
        result = proc.process(sample_bgr, normalize=False)
        assert result.shape == (64, 64, 3)


# ── IPExtractor (mocked CLIP) ─────────────────────────────────────────────

class TestIPExtractor:
    def _make_extractor(self) -> IPExtractor:
        """Create an extractor with mocked CLIP bundle injected into registry."""
        extractor = IPExtractor(ip_adapter_path=None)
        # Replace the registered loader with a mock bundle
        bundle = _make_mock_clip_bundle()
        reg = extractor._registry
        reg._entries[IPExtractor.MODEL_ID].module = bundle
        from src.core.model_registry import ModelState
        reg._entries[IPExtractor.MODEL_ID].state = ModelState.ON_CPU
        return extractor

    def test_uses_fallback_without_weights(self):
        extractor = IPExtractor(ip_adapter_path=None)
        assert extractor._use_fallback is True

    def test_extract_returns_ip_features(self, sample_image_path):
        extractor = self._make_extractor()
        req = ExtractionRequest(source_image_path=sample_image_path)
        result = extractor.extract(req)
        assert isinstance(result, IPFeatures)

    def test_image_embeds_is_numpy(self, sample_image_path):
        extractor = self._make_extractor()
        req = ExtractionRequest(source_image_path=sample_image_path)
        result = extractor.extract(req)
        assert isinstance(result.image_embeds, np.ndarray)

    def test_preprocessed_shape(self, sample_image_path):
        extractor = self._make_extractor()
        req = ExtractionRequest(source_image_path=sample_image_path)
        result = extractor.extract(req)
        assert result.preprocessed.shape == (224, 224, 3)

    def test_backend_used_fallback(self, sample_image_path):
        extractor = self._make_extractor()
        req = ExtractionRequest(source_image_path=sample_image_path)
        result = extractor.extract(req)
        assert result.backend_used == "clip_fallback"

    def test_face_embeds_none_without_insightface(self, sample_image_path):
        extractor = self._make_extractor()
        req = ExtractionRequest(source_image_path=sample_image_path)
        result = extractor.extract(req)
        # insightface not installed in test env — face_embeds should be None
        assert result.face_embeds is None

    def test_custom_target_size(self, sample_image_path):
        extractor = self._make_extractor()
        req = ExtractionRequest(
            source_image_path=sample_image_path,
            target_size=(336, 336),
        )
        result = extractor.extract(req)
        assert result.preprocessed.shape == (336, 336, 3)

    def test_ip_adapter_path_not_found_uses_fallback(self, tmp_path):
        extractor = IPExtractor(ip_adapter_path=tmp_path / "nonexistent")
        assert extractor._use_fallback is True

    def test_ip_adapter_path_with_encoder_dir(self, tmp_path):
        encoder_dir = tmp_path / "image_encoder"
        encoder_dir.mkdir()
        (encoder_dir / "config.json").write_text("{}")
        extractor = IPExtractor(ip_adapter_path=tmp_path)
        assert extractor._use_fallback is False
