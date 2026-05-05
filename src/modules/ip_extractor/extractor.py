"""
IP character feature extractor.

Primary backend: IP-Adapter (CLIP ViT-H/14 image encoder).
Fallback: lightweight CLIP ViT-B/32 via HuggingFace transformers
          (no custom weights required — downloads automatically).

The extractor produces two complementary embeddings:
1. image_embeds  — full CLIP patch tokens (spatial detail, style).
2. face_embeds   — InsightFace identity vector (optional, face-specific).

Both are stored as numpy arrays so they can be passed to diffusers
IP-Adapter pipelines without holding GPU memory between calls.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import IPExtractionError

from .preprocessor import ImagePreprocessor
from .schemas import ExtractionRequest, IPFeatures

logger = logging.getLogger(__name__)

_CLIP_VRAM_GB   = 2.5
_CLIP_MODEL_ID  = "openai/clip-vit-large-patch14"   # ViT-L/14 — same as IP-Adapter


class IPExtractor:
    """
    Extracts IP character features from a source image.

    Parameters
    ----------
    ip_adapter_path:
        Directory containing IP-Adapter weights (image_encoder/ subfolder).
        If None or missing, falls back to HuggingFace CLIP download.
    registry:
        ModelRegistry singleton.
    """

    MODEL_ID = "clip_image_encoder"

    def __init__(
        self,
        ip_adapter_path: Path | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._ip_adapter_path = Path(ip_adapter_path) if ip_adapter_path else None
        self._registry = registry or ModelRegistry.instance()
        self._preprocessor = ImagePreprocessor()
        self._use_fallback = not self._ip_adapter_available()

        self._register_model()
        if self._use_fallback:
            logger.info(
                "IP-Adapter weights not found — using HuggingFace CLIP fallback."
            )

    # ── Public API ────────────────────────────────────────────────────────

    def extract(self, request: ExtractionRequest) -> IPFeatures:
        """
        Extract IP character features from the source image.

        Returns
        -------
        IPFeatures with image_embeds (numpy) and optional face_embeds.
        """
        self._preprocessor.target_size = request.target_size
        preprocessed = self._preprocessor.process(
            request.source_image_path, normalize=request.normalize
        )

        image_embeds = self._encode_image(preprocessed)
        face_embeds  = self._encode_face(request.source_image_path)

        return IPFeatures(
            image_embeds=image_embeds,
            face_embeds=face_embeds,
            preprocessed=preprocessed,
            backend_used="clip_fallback" if self._use_fallback else "ip_adapter",
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _ip_adapter_available(self) -> bool:
        if self._ip_adapter_path is None:
            return False
        encoder_dir = self._ip_adapter_path / "image_encoder"
        return encoder_dir.exists() and any(encoder_dir.iterdir())

    def _register_model(self) -> None:
        if self._use_fallback:
            model_id = _CLIP_MODEL_ID

            def _loader():
                try:
                    from transformers import (  # type: ignore
                        CLIPImageProcessor,
                        CLIPVisionModelWithProjection,
                    )
                    processor = CLIPImageProcessor.from_pretrained(model_id)
                    model = CLIPVisionModelWithProjection.from_pretrained(model_id)
                    return (processor, model)
                except Exception as exc:
                    raise IPExtractionError(
                        f"Failed to load CLIP from HuggingFace: {exc}"
                    ) from exc
        else:
            encoder_path = str(self._ip_adapter_path / "image_encoder")

            def _loader():
                try:
                    from transformers import (  # type: ignore
                        CLIPImageProcessor,
                        CLIPVisionModelWithProjection,
                    )
                    processor = CLIPImageProcessor.from_pretrained(encoder_path)
                    model = CLIPVisionModelWithProjection.from_pretrained(encoder_path)
                    return (processor, model)
                except Exception as exc:
                    raise IPExtractionError(
                        f"Failed to load IP-Adapter encoder from {encoder_path}: {exc}"
                    ) from exc

        self._registry.register(
            self.MODEL_ID, _loader, estimated_vram_gb=_CLIP_VRAM_GB
        )

    def _encode_image(self, preprocessed: np.ndarray) -> np.ndarray:
        """Run CLIP image encoder and return patch embeddings as numpy."""
        import torch

        with self._registry.model_context(self.MODEL_ID, offload_after=True) as bundle:
            processor, model = bundle
            try:
                # processor expects uint8 PIL or numpy; de-normalise first
                img_uint8 = self._denormalize(preprocessed)
                inputs = processor(images=img_uint8, return_tensors="pt")
                device = self._registry.device_manager.device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model = model.to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                # image_embeds: (1, hidden_size)
                embeds = outputs.image_embeds.cpu().numpy()
                return embeds
            except Exception as exc:
                raise IPExtractionError(f"CLIP encoding failed: {exc}") from exc

    @staticmethod
    def _denormalize(img: np.ndarray) -> np.ndarray:
        """Reverse CLIP normalisation back to uint8 for the HF processor."""
        from .preprocessor import _CLIP_MEAN, _CLIP_STD
        restored = img * _CLIP_STD + _CLIP_MEAN
        restored = np.clip(restored * 255.0, 0, 255).astype(np.uint8)
        return restored

    @staticmethod
    def _encode_face(image_path: Path) -> np.ndarray | None:
        """
        Extract InsightFace identity embedding.
        Returns None if insightface is not installed or no face is detected.
        """
        try:
            import cv2
            import insightface  # type: ignore
            app = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            faces = app.get(img)
            if not faces:
                logger.debug("No face detected in %s", image_path.name)
                return None
            # Return the embedding of the largest detected face
            largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return largest.embedding[np.newaxis, :]   # (1, 512)
        except ImportError:
            logger.debug("insightface not installed — skipping face embedding.")
            return None
        except Exception as exc:
            logger.warning("Face encoding failed: %s", exc)
            return None
