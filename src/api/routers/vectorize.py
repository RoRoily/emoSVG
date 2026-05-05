"""POST /vectorize — SVG vectorization endpoint."""
from __future__ import annotations

import os, shutil, tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.dependencies import get_pipeline
from src.core.exceptions import EmoSVGError
from src.modules.svg_vectorizer.vectorizer import SVGVectorizer
from src.modules.svg_vectorizer.schemas import VectorizationRequest

router = APIRouter(prefix="/vectorize", tags=["vectorize"])


class VectorizeResponse(BaseModel):
    output_path: str
    layer_count: int
    total_paths: int
    backend_used: str


@router.post("", response_model=VectorizeResponse)
async def vectorize(
    file: UploadFile = File(...),
    bezier_tolerance: float = Form(1.5),
    min_region_area: int = Form(100),
    layer_naming: str = Form("semantic"),
) -> VectorizeResponse:
    """Convert a raster image to a layered SVG vector file."""
    suffix = Path(file.filename or "upload.png").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        vec = SVGVectorizer()
        result = vec.vectorize(VectorizationRequest(
            source_image_path=tmp_path,
            bezier_tolerance=bezier_tolerance,
            min_region_area=min_region_area,
            layer_naming=layer_naming,
        ))
    except EmoSVGError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return VectorizeResponse(
        output_path=str(result.output_path),
        layer_count=result.layer_count,
        total_paths=result.total_paths,
        backend_used=result.backend_used,
    )
