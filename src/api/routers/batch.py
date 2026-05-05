"""POST /batch/generate — process multiple images in one request."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.api.dependencies import get_pipeline
from src.core.exceptions import EmoSVGError
from src.modules.meme_animator.schemas import MemeExpression
from src.pipeline.full_pipeline import FullPipeline, FullPipelineRequest

router = APIRouter(prefix="/batch", tags=["batch"])

_MAX_BATCH = 8  # hard cap to prevent OOM on a single request


class BatchItemResult(BaseModel):
    filename: str
    animation_path: Optional[str] = None
    frame_count: Optional[int] = None
    duration_ms: Optional[float] = None
    reconstruction_paths: dict[str, str] = {}
    svg_path: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    backends: dict[str, str] = {}
    error: Optional[str] = None


class BatchGenerateResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchItemResult]


@router.post("/generate", response_model=BatchGenerateResponse)
async def batch_generate(
    files: list[UploadFile] = File(..., description="Up to 8 source IP character images"),
    expression: MemeExpression = Form(MemeExpression.SHOCK),
    output_format: str = Form("gif"),
    fps: int = Form(24),
    width: int = Form(512),
    height: int = Form(512),
    run_3d: bool = Form(False),
    run_svg: bool = Form(False),
    use_toon_crafter: bool = Form(False),
    frames_between: int = Form(4),
    pipeline: FullPipeline = Depends(get_pipeline),
) -> BatchGenerateResponse:
    """
    Run the full pipeline on multiple images sequentially.

    - Accepts up to 8 images per request.
    - Each image is processed independently; a failure on one does not abort the rest.
    - 3D and SVG are disabled by default to keep batch latency reasonable.
    - Returns per-image results including any error messages.
    """
    if len(files) > _MAX_BATCH:
        raise HTTPException(
            status_code=422,
            detail=f"Batch size {len(files)} exceeds maximum of {_MAX_BATCH}.",
        )
    if len(files) == 0:
        raise HTTPException(status_code=422, detail="At least one file is required.")
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

    results: list[BatchItemResult] = []

    for upload in files:
        filename = upload.filename or "upload.png"
        suffix = Path(filename).suffix or ".png"
        tmp_path: Optional[Path] = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(upload.file, tmp)
                tmp_path = Path(tmp.name)

            req = FullPipelineRequest(
                source_image_path=tmp_path,
                expression=expression,
                output_format=output_format,
                fps=fps,
                resolution=(width, height),
                run_3d=run_3d,
                run_svg=run_svg,
                use_toon_crafter=use_toon_crafter,
                frames_between=frames_between,
            )
            result = pipeline.execute(req)
            anim = result.animation

            results.append(BatchItemResult(
                filename=filename,
                animation_path=str(anim.output_path),
                frame_count=anim.frame_count,
                duration_ms=anim.duration_ms,
                reconstruction_paths={
                    k: str(v)
                    for k, v in (result.reconstruction.output_paths.items()
                                 if result.reconstruction else {})
                },
                svg_path=str(result.vectorization.output_path)
                         if result.vectorization else None,
                elapsed_seconds=result.elapsed_seconds,
                backends={
                    "ip":             result.ip_features.backend_used,
                    "animation":      anim.backend_used,
                    "reconstruction": result.reconstruction.backend_used
                                      if result.reconstruction else "skipped",
                    "svg":            result.vectorization.backend_used
                                      if result.vectorization else "skipped",
                },
            ))

        except EmoSVGError as exc:
            results.append(BatchItemResult(filename=filename, error=str(exc)))
        except Exception as exc:
            results.append(BatchItemResult(filename=filename, error=f"Internal error: {exc}"))
        finally:
            if tmp_path and tmp_path.exists():
                os.unlink(tmp_path)

    succeeded = sum(1 for r in results if r.error is None)
    return BatchGenerateResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )
