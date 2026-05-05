"""
Unit tests for reconstructor_3d module.
No GPU, no TripoSR weights — uses sphere fallback throughout.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

from src.core.model_registry import ModelRegistry
from src.modules.reconstructor_3d.schemas import (
    ExportFormat, ReconstructionRequest, MeshStats,
)
from src.modules.reconstructor_3d.mesh_processor import MeshProcessor
from src.modules.reconstructor_3d.exporter import MeshExporter
from src.modules.reconstructor_3d.reconstructor import Reconstructor3D


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()


@pytest.fixture()
def sample_image_path(tmp_path: Path) -> Path:
    img = np.full((256, 256, 3), (120, 160, 200), dtype=np.uint8)
    p = tmp_path / "sample.png"
    cv2.imwrite(str(p), img)
    return p


@pytest.fixture()
def sphere_mesh() -> trimesh.Trimesh:
    return trimesh.creation.uv_sphere(radius=1.0, count=[16, 16])


# ── Schema validation ─────────────────────────────────────────────────────

class TestReconstructionRequest:
    def test_valid_request(self, sample_image_path):
        req = ReconstructionRequest(source_image_path=sample_image_path)
        assert req.mc_resolution == 256
        assert ExportFormat.OBJ in req.export_formats

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(Exception):
            ReconstructionRequest(source_image_path=tmp_path / "nope.png")

    def test_custom_formats(self, sample_image_path):
        req = ReconstructionRequest(
            source_image_path=sample_image_path,
            export_formats=[ExportFormat.GLB],
        )
        assert req.export_formats == [ExportFormat.GLB]

    def test_mc_resolution_bounds(self, sample_image_path):
        with pytest.raises(Exception):
            ReconstructionRequest(
                source_image_path=sample_image_path,
                mc_resolution=9999,
            )


# ── MeshProcessor ─────────────────────────────────────────────────────────

class TestMeshProcessor:
    def test_process_sphere_returns_trimesh(self, sphere_mesh):
        proc = MeshProcessor()
        result = proc.process(sphere_mesh)
        assert isinstance(result, trimesh.Trimesh)

    def test_process_preserves_geometry(self, sphere_mesh):
        proc = MeshProcessor(smooth_iterations=0, remove_small_components=False)
        result = proc.process(sphere_mesh)
        assert len(result.vertices) > 0
        assert len(result.faces) > 0

    def test_process_scene_extracts_largest(self):
        proc = MeshProcessor()
        small = trimesh.creation.uv_sphere(radius=0.1, count=[4, 4])
        large = trimesh.creation.uv_sphere(radius=2.0, count=[16, 16])
        scene = trimesh.Scene(geometry={"small": small, "large": large})
        result = proc.process(scene)
        assert len(result.faces) >= len(large.faces)

    def test_process_empty_scene_raises(self):
        proc = MeshProcessor()
        scene = trimesh.Scene()
        with pytest.raises((ValueError, Exception)):
            proc.process(scene)

    def test_compute_stats(self, sphere_mesh):
        proc = MeshProcessor()
        stats = proc.compute_stats(sphere_mesh)
        assert isinstance(stats, MeshStats)
        assert stats.vertex_count > 0
        assert stats.face_count > 0
        assert len(stats.bounding_box) == 3

    def test_stats_bounding_box_positive(self, sphere_mesh):
        proc = MeshProcessor()
        stats = proc.compute_stats(sphere_mesh)
        assert all(v > 0 for v in stats.bounding_box)

    def test_smoothing_does_not_crash(self, sphere_mesh):
        proc = MeshProcessor(smooth_iterations=3)
        result = proc.process(sphere_mesh)
        assert len(result.vertices) > 0

    def test_wrong_type_raises(self):
        proc = MeshProcessor()
        with pytest.raises(TypeError):
            proc.process("not a mesh")


# ── MeshExporter ──────────────────────────────────────────────────────────

class TestMeshExporter:
    def test_export_obj(self, sphere_mesh, tmp_path):
        exporter = MeshExporter()
        paths = exporter.export(sphere_mesh, tmp_path, "test", [ExportFormat.OBJ])
        assert "obj" in paths
        assert paths["obj"].exists()
        assert paths["obj"].stat().st_size > 0

    def test_export_glb(self, sphere_mesh, tmp_path):
        exporter = MeshExporter()
        paths = exporter.export(sphere_mesh, tmp_path, "test", [ExportFormat.GLB])
        assert "glb" in paths
        assert paths["glb"].exists()

    def test_export_both_formats(self, sphere_mesh, tmp_path):
        exporter = MeshExporter()
        paths = exporter.export(
            sphere_mesh, tmp_path, "test",
            [ExportFormat.OBJ, ExportFormat.GLB],
        )
        assert len(paths) == 2
        for p in paths.values():
            assert p.exists()

    def test_export_creates_output_dir(self, sphere_mesh, tmp_path):
        exporter = MeshExporter()
        nested = tmp_path / "deep" / "nested"
        paths = exporter.export(sphere_mesh, nested, "mesh", [ExportFormat.OBJ])
        assert paths["obj"].exists()

    def test_stem_used_in_filename(self, sphere_mesh, tmp_path):
        exporter = MeshExporter()
        paths = exporter.export(sphere_mesh, tmp_path, "my_character", [ExportFormat.OBJ])
        assert "my_character" in paths["obj"].name


# ── Reconstructor3D end-to-end (fallback mode) ────────────────────────────

class TestReconstructor3DE2E:
    def test_reconstruct_returns_result(self, sample_image_path, tmp_path):
        rec = Reconstructor3D(output_dir=tmp_path)
        req = ReconstructionRequest(source_image_path=sample_image_path)
        result = rec.reconstruct(req)
        assert result.backend_used == "triposr_fallback"
        assert result.mesh_stats.vertex_count > 0

    def test_output_files_created(self, sample_image_path, tmp_path):
        rec = Reconstructor3D(output_dir=tmp_path)
        req = ReconstructionRequest(
            source_image_path=sample_image_path,
            export_formats=[ExportFormat.OBJ, ExportFormat.GLB],
        )
        result = rec.reconstruct(req)
        for fmt, path in result.output_paths.items():
            assert path.exists(), f"Missing output for format {fmt}"

    def test_obj_only_export(self, sample_image_path, tmp_path):
        rec = Reconstructor3D(output_dir=tmp_path)
        req = ReconstructionRequest(
            source_image_path=sample_image_path,
            export_formats=[ExportFormat.OBJ],
        )
        result = rec.reconstruct(req)
        assert "obj" in result.output_paths
        assert "glb" not in result.output_paths

    def test_mesh_stats_populated(self, sample_image_path, tmp_path):
        rec = Reconstructor3D(output_dir=tmp_path)
        req = ReconstructionRequest(source_image_path=sample_image_path)
        result = rec.reconstruct(req)
        stats = result.mesh_stats
        assert stats.face_count > 0
        assert stats.vertex_count > 0
        assert len(stats.bounding_box) == 3

    def test_uses_fallback_without_triposr(self, sample_image_path, tmp_path):
        rec = Reconstructor3D(output_dir=tmp_path)
        assert rec._use_fallback is True
