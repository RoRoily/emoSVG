"""POST /generate — full pipeline endpoint."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import get_pipeline
from src.core.exceptions import EmoSVGError
from src.modules.meme_animator.schemas import MemeExpression
from src.pipeline.full_pipeline import FullPipeline, FullPipelineRequest

router = APIRouter(prefix="/generate", tags=["generate"])


class GenerateResponse(BaseModel):
    animation_path: str
    frame_count: int
    duration_ms: float
    reconstruction_paths: dict[str, str]
    svg_path: str | None
    elapsed_seconds: float
    backends: dict[str, str]


@router.post("", response_model=GenerateResponse)
async def generate(
    file: UploadFile = File(..., description="Source IP character image"),
    expression: MemeExpression = Form(MemeExpression.SHOCK),
    output_format: str = Form("gif"),
    fps: int = Form(24),
    width: int = Form(512),
    height: int = Form(512),
    run_3d: bool = Form(True),
    run_svg: bool = Form(True),
    use_source_for_3d_svg: bool = Form(False),
    use_toon_crafter: bool = Form(False),
    frames_between: int = Form(4),
    pipeline: FullPipeline = Depends(get_pipeline),
) -> GenerateResponse:
    """
    Run the full IP Meme generation pipeline.

    Upload a character image and receive:
    - A Squash-and-Stretch meme animation (GIF/MP4/WebP)
    - Optional 3D mesh (OBJ + GLB)
    - Optional SVG vector file
    """
    import os
    import shutil
    import tempfile

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

    # Save upload to a temp file
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
            run_3d=run_3d,
            run_svg=run_svg,
            use_source_for_3d_svg=use_source_for_3d_svg,
            use_toon_crafter=use_toon_crafter,
            frames_between=frames_between,
        )
        result = pipeline.execute(req)
    except EmoSVGError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    finally:
        os.unlink(tmp_path)

    return GenerateResponse(
        animation_path=str(result.animation.output_path),
        frame_count=result.animation.frame_count,
        duration_ms=result.animation.duration_ms,
        reconstruction_paths={
            k: str(v)
            for k, v in (result.reconstruction.output_paths.items()
                         if result.reconstruction else {})
        },
        svg_path=str(result.vectorization.output_path) if result.vectorization else None,
        elapsed_seconds=result.elapsed_seconds,
        backends={
            "ip": result.ip_features.backend_used,
            "animation": result.animation.backend_used,
            "reconstruction": result.reconstruction.backend_used if result.reconstruction else "skipped",
            "svg": result.vectorization.backend_used if result.vectorization else "skipped",
        },
    )
