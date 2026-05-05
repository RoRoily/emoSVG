"""POST /reconstruct — 3D reconstruction endpoint."""
from __future__ import annotations

import os, shutil, tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import get_pipeline
from src.core.exceptions import EmoSVGError
from src.modules.reconstructor_3d.reconstructor import Reconstructor3D
from src.modules.reconstructor_3d.schemas import ReconstructionRequest

router = APIRouter(prefix="/reconstruct", tags=["reconstruct"])


class ReconstructResponse(BaseModel):
    output_paths: dict[str, str]
    vertex_count: int
    face_count: int
    is_watertight: bool
    backend_used: str


@router.post("", response_model=ReconstructResponse)
async def reconstruct(
    file: UploadFile = File(...),
    mc_resolution: int = Form(256),
    remove_background: bool = Form(True),
) -> ReconstructResponse:
    """Reconstruct a 3D mesh from a single character image."""
    suffix = Path(file.filename or "upload.png").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        rec = Reconstructor3D()
        result = rec.reconstruct(ReconstructionRequest(
            source_image_path=tmp_path,
            mc_resolution=mc_resolution,
            remove_background=remove_background,
        ))
    except EmoSVGError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return ReconstructResponse(
        output_paths={k: str(v) for k, v in result.output_paths.items()},
        vertex_count=result.mesh_stats.vertex_count,
        face_count=result.mesh_stats.face_count,
        is_watertight=result.mesh_stats.is_watertight,
        backend_used=result.backend_used,
    )
