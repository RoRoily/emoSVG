"""Data contracts for lightweight cartoon layer parsing."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.modules.cartoon_analyzer.schemas import BBox


@dataclass
class CartoonLayer:
    name: str
    rgba: np.ndarray
    mask: np.ndarray
    bbox: BBox
    confidence: float
    role: str = "part"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bbox": self.bbox.to_dict(),
            "confidence": float(self.confidence),
            "role": self.role,
            "nonzero_alpha": int(np.count_nonzero(self.mask)),
        }


@dataclass
class CartoonLayerParseResult:
    layers: list[CartoonLayer]
    backend_used: str = "geometry_prompted_masks"
    warnings: list[str] = field(default_factory=list)

    def get(self, name: str) -> CartoonLayer | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def to_dict(self) -> dict:
        return {
            "backend_used": self.backend_used,
            "layers": [layer.to_dict() for layer in self.layers],
            "warnings": list(self.warnings),
        }

    def save_layers(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for layer in self.layers:
            cv2.imwrite(str(output_dir / f"{layer.name}.png"), layer.rgba)
            cv2.imwrite(str(output_dir / f"{layer.name}_mask.png"), layer.mask)
