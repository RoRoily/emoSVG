"""POST /animate — animation-only endpoint."""
from __future__ import annotations

import os, shutil, tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.api.dependencies import get_pipeline
from src.core.exceptions import EmoSVGError
from src.modules.meme_animator.schemas import MemeExpression
from src.pipeline.full_pipeline import FullPipeline, FullPipelineRequest

router = APIRouter(prefix="/animate", tags=["animate"])


class AnimateResponse(BaseModel):
    output_path: str
    frame_count: int
    duration_ms: float
    backend_used: str


@router.post("", response_model=AnimateResponse)
async def animate(
    file: UploadFile = File(...),
    expression: MemeExpression = Form(MemeExpression.SHOCK),
    output_format: str = Form("gif"),
    fps: int = Form(24),
    width: int = Form(512),
    height: int = Form(512),
    use_toon_crafter: bool = Form(False),
    frames_between: int = Form(4),
    pipeline: FullPipeline = Depends(get_pipeline),
) -> AnimateResponse:
    """Generate a Squash-and-Stretch meme animation only (skip 3D and SVG)."""
    if not (64 <= width <= 2048):
        raise HTTPException(status_code=422, detail="width must be between 64 and 2048")
    if not (64 <= height <= 2048):
        raise HTTPException(status_code=422, detail="height must be between 64 and 2048")
    if not (8 <= fps <= 60):
        raise HTTPException(status_code=422, detail="fps must be between 8 and 60")
    if output_format not in ("gif", "mp4", "webp"):
        raise HTTPException(status_code=422, detail="output_format must be gif, mp4, or webp")
    if not (1 <= frames_between <= 16):
        raise HTTPException(status_code=422, detail="frames_between must be between 1 and 16")
    suffix = Path(file.filename or "upload.png").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        req = FullPipelineRequest(
            source_image_path=tmp_path,
            expression=expression,
            output_format=output_format,
            fps=fps,
            resolution=(width, height),
            run_3d=False,
            run_svg=False,
            use_toon_crafter=use_toon_crafter,
            frames_between=frames_between,
        )
        result = pipeline.execute(req)
    except EmoSVGError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    anim = result.animation
    return AnimateResponse(
        output_path=str(anim.output_path),
        frame_count=anim.frame_count,
        duration_ms=anim.duration_ms,
        backend_used=anim.backend_used,
    )
