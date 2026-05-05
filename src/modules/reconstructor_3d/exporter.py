"""
Mesh exporter: writes trimesh objects to .obj and/or .glb files.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .schemas import ExportFormat

if TYPE_CHECKING:
    import trimesh as trimesh_type

logger = logging.getLogger(__name__)


class MeshExporter:
    """Exports a trimesh.Trimesh to one or more file formats."""

    def export(
        self,
        mesh: trimesh_type.Trimesh,
        output_dir: Path,
        stem: str,
        formats: list[ExportFormat],
    ) -> dict[str, Path]:
        """
        Parameters
        ----------
        mesh:       Processed trimesh.Trimesh.
        output_dir: Directory to write files into.
        stem:       Base filename without extension.
        formats:    List of ExportFormat values to write.

        Returns
        -------
        Dict mapping format string to output Path.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}

        for fmt in formats:
            path = output_dir / f"{stem}.{fmt.value}"
            try:
                mesh.export(str(path))
                results[fmt.value] = path
                logger.info("Exported mesh to %s", path)
            except Exception as exc:
                logger.error("Failed to export %s: %s", path, exc)
                raise

        return results
