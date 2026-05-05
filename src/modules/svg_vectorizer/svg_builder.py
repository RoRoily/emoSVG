"""
Hierarchical SVG document builder.

Assembles per-layer path data into a well-structured SVG file with:
- A <defs> block for reusable styles.
- One <g id="layer-N"> group per segmented region.
- Deterministic fill colours derived from the region's mean pixel colour.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import svgwrite

from .schemas import SVGLayer

logger = logging.getLogger(__name__)


class SVGBuilder:
    """
    Builds and writes an SVG file from a list of (mask, path_data_list) pairs.

    Parameters
    ----------
    precision:      Decimal places for SVG coordinates.
    layer_naming:   "semantic" uses colour-based labels; "index" uses layer-N.
    """

    def __init__(self, precision: int = 2, layer_naming: str = "semantic") -> None:
        self.precision = precision
        self.layer_naming = layer_naming

    def build(
        self,
        image_bgr: np.ndarray,
        masks: list[np.ndarray],
        paths_per_mask: list[list[str]],
        output_path: Path,
    ) -> list[SVGLayer]:
        """
        Parameters
        ----------
        image_bgr:       Original source image (used for colour sampling).
        masks:           List of boolean HxW masks (one per layer).
        paths_per_mask:  Parallel list of SVG path data strings per mask.
        output_path:     Destination .svg file path.

        Returns
        -------
        List of SVGLayer metadata objects.
        """
        h, w = image_bgr.shape[:2]
        dwg = svgwrite.Drawing(
            str(output_path),
            size=(f"{w}px", f"{h}px"),
            viewBox=f"0 0 {w} {h}",
        )
        dwg.attribs["xmlns"] = "http://www.w3.org/2000/svg"

        layers: list[SVGLayer] = []

        for idx, (mask, path_data_list) in enumerate(zip(masks, paths_per_mask)):
            if not path_data_list:
                continue

            fill_hex = self._sample_fill(image_bgr, mask)
            label = self._make_label(image_bgr, mask, idx)
            layer_id = f"layer-{idx}"

            group = dwg.g(id=layer_id, fill=fill_hex, opacity="1")
            for path_data in path_data_list:
                group.add(dwg.path(d=path_data))
            dwg.add(group)

            layers.append(SVGLayer(
                layer_id=layer_id,
                label=label,
                path_count=len(path_data_list),
                fill_color=fill_hex,
                opacity=1.0,
            ))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dwg.save(pretty=True)
        logger.info("SVG written to %s (%d layers)", output_path, len(layers))
        return layers

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _sample_fill(image_bgr: np.ndarray, mask: np.ndarray) -> str:
        """Compute the mean BGR colour of the masked region and return as CSS hex."""
        if mask.sum() == 0:
            return "#888888"
        region = image_bgr[mask]
        mean_bgr = region.mean(axis=0)
        b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
        return f"#{r:02x}{g:02x}{b:02x}"

    def _make_label(
        self,
        image_bgr: np.ndarray,
        mask: np.ndarray,
        idx: int,
    ) -> str:
        if self.layer_naming == "index":
            return f"layer-{idx}"
        # Semantic: classify by hue bucket
        fill = self._sample_fill(image_bgr, mask).lstrip("#")
        r, g, b = int(fill[0:2], 16), int(fill[2:4], 16), int(fill[4:6], 16)
        pixel = np.array([[[b, g, r]]], dtype=np.uint8)
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
        if v < 40:
            return "shadow"
        if s < 30:
            return "highlight" if v > 200 else "midtone"
        hue_labels = [
            (15,  "red"), (30,  "orange"), (45,  "yellow"),
            (75,  "yellow-green"), (105, "green"), (135, "cyan-green"),
            (150, "cyan"), (165, "blue-cyan"), (195, "blue"),
            (225, "blue-violet"), (255, "violet"), (285, "magenta"),
            (315, "red-magenta"), (360, "red"),
        ]
        for threshold, name in hue_labels:
            if h * 2 <= threshold:
                return name
        return f"region-{idx}"
