"""
Mesh post-processing: cleaning, smoothing, and stats computation.
Uses trimesh for all geometry operations.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .schemas import MeshStats

if TYPE_CHECKING:
    import trimesh as trimesh_type

logger = logging.getLogger(__name__)


class MeshProcessor:
    """
    Applies a configurable post-processing chain to a raw trimesh.Trimesh.

    Steps (all optional):
    1. Remove degenerate / duplicate faces.
    2. Remove small disconnected components.
    3. Laplacian smoothing.
    4. Face-count simplification (if pymeshlab is available).
    """

    def __init__(
        self,
        remove_small_components: bool = True,
        smooth_iterations: int = 2,
        target_face_count: int = 50_000,
    ) -> None:
        self.remove_small_components = remove_small_components
        self.smooth_iterations = smooth_iterations
        self.target_face_count = target_face_count

    def process(self, mesh) -> trimesh_type.Trimesh:
        import trimesh

        if not isinstance(mesh, trimesh.Trimesh):
            # TripoSR may return a Scene; extract the largest geometry.
            if isinstance(mesh, trimesh.Scene):
                geometries = list(mesh.geometry.values())
                if not geometries:
                    raise ValueError("TripoSR returned an empty Scene.")
                mesh = max(geometries, key=lambda g: len(g.faces))
            else:
                raise TypeError(f"Expected trimesh.Trimesh or Scene, got {type(mesh)}")

        logger.info(
            "Mesh pre-process: %d vertices, %d faces", len(mesh.vertices), len(mesh.faces)
        )

        # Step 1 — remove degenerate geometry (trimesh 4.x API)
        # nondegenerate_faces() returns a boolean mask; apply it to rebuild the mesh.
        valid_mask = mesh.nondegenerate_faces()
        if not valid_mask.all():
            mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces[valid_mask],
                process=False,
            )
        mesh.remove_unreferenced_vertices()

        # Step 2 — keep only the largest connected component
        if self.remove_small_components:
            components = mesh.split(only_watertight=False)
            if components:
                mesh = max(components, key=lambda c: len(c.faces))

        # Step 3 — Laplacian smoothing
        if self.smooth_iterations > 0:
            trimesh.smoothing.filter_laplacian(mesh, iterations=self.smooth_iterations)

        # Step 4 — simplification (requires pymeshlab)
        if len(mesh.faces) > self.target_face_count:
            mesh = self._simplify(mesh)

        logger.info(
            "Mesh post-process: %d vertices, %d faces (watertight=%s)",
            len(mesh.vertices), len(mesh.faces), mesh.is_watertight,
        )
        return mesh

    def compute_stats(self, mesh) -> MeshStats:
        bounds = mesh.bounding_box.extents if hasattr(mesh, "bounding_box") else (0.0, 0.0, 0.0)
        return MeshStats(
            vertex_count=len(mesh.vertices),
            face_count=len(mesh.faces),
            is_watertight=bool(mesh.is_watertight),
            bounding_box=tuple(float(x) for x in bounds),  # type: ignore[arg-type]
        )

    def _simplify(self, mesh) -> trimesh_type.Trimesh:
        try:
            import pymeshlab  # type: ignore
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(
                vertex_matrix=mesh.vertices,
                face_matrix=mesh.faces,
            ))
            ms.meshing_decimation_quadric_edge_collapse(targetfacenum=self.target_face_count)
            m = ms.current_mesh()
            import trimesh
            return trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
        except ImportError:
            logger.warning("pymeshlab not installed — skipping mesh simplification.")
            return mesh
        except Exception as exc:
            logger.warning("Mesh simplification failed (%s) — using original.", exc)
            return mesh
