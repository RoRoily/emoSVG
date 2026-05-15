"""Data contracts for lightweight 2D cartoon rigging."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.modules.cartoon_analyzer.schemas import BBox, CartoonFaceGeometry, Point2D


@dataclass
class RigLayer:
    name: str
    rgba: np.ndarray
    bbox: BBox
    pivot: Point2D
    role: str
    z_index: int


@dataclass
class CartoonRigAsset:
    canvas_size: tuple[int, int]
    base_bgr: np.ndarray
    geometry: CartoonFaceGeometry
    layers: list[RigLayer]

    def get_layer(self, name: str) -> RigLayer | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None
