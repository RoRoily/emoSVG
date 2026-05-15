"""Data contracts for cartoon character geometry analysis."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y)}


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    @property
    def center(self) -> Point2D:
        return Point2D(self.x + self.w / 2.0, self.y + self.h / 2.0)

    def clamp(self, width: int, height: int) -> BBox:
        x1 = max(0, min(width - 1, self.x))
        y1 = max(0, min(height - 1, self.y))
        x2 = max(x1 + 1, min(width, self.x2))
        y2 = max(y1 + 1, min(height, self.y2))
        return BBox(x1, y1, x2 - x1, y2 - y1)

    def pad(self, px: int, py: int | None = None) -> BBox:
        py = px if py is None else py
        return BBox(self.x - px, self.y - py, self.w + 2 * px, self.h + 2 * py)

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class CartoonFaceGeometry:
    face_bbox: BBox
    foreground_bbox: BBox
    left_eye_bbox: BBox
    right_eye_bbox: BBox
    mouth_bbox: BBox
    left_eye_center: Point2D
    right_eye_center: Point2D
    mouth_center: Point2D
    head_center: Point2D
    confidence: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "face_bbox": self.face_bbox.to_dict(),
            "foreground_bbox": self.foreground_bbox.to_dict(),
            "left_eye_bbox": self.left_eye_bbox.to_dict(),
            "right_eye_bbox": self.right_eye_bbox.to_dict(),
            "mouth_bbox": self.mouth_bbox.to_dict(),
            "left_eye_center": self.left_eye_center.to_dict(),
            "right_eye_center": self.right_eye_center.to_dict(),
            "mouth_center": self.mouth_center.to_dict(),
            "head_center": self.head_center.to_dict(),
            "confidence": float(self.confidence),
            "warnings": list(self.warnings),
        }


@dataclass
class CartoonFaceAnalysis:
    image_size: tuple[int, int]
    geometry: CartoonFaceGeometry
    foreground_mask: object
    backend_used: str = "heuristic_cartoon_geometry"

    def to_dict(self) -> dict:
        return {
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "geometry": self.geometry.to_dict(),
            "backend_used": self.backend_used,
        }
