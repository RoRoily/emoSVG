"""
FastAPI application entry point.

Start the server:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
or:
    python -m src.api.main
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware import RequestLoggingMiddleware
from src.api.routers import animate, batch, generate, reconstruct, vectorize

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────
app = FastAPI(
    title="emoSVG — IP Meme Animation Engine",
    description=(
        "Multi-modal generative AI pipeline: "
        "IP character image → Meme animation → 3D mesh → SVG vector"
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (permissive for local dev; tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# ── Routers ───────────────────────────────────────────────────────────────
app.include_router(generate.router)
app.include_router(animate.router)
app.include_router(reconstruct.router)
app.include_router(vectorize.router)
app.include_router(batch.router)


@app.get("/health", tags=["health"])
async def health() -> dict:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/expressions", tags=["info"])
async def expressions() -> dict:
    """List all available MemeExpression values with descriptions."""
    from src.modules.meme_animator.schemas import MemeExpression
    descriptions = {
        MemeExpression.SHOCK:     "Eyes bulge out, jaw drops — classic shock/surprise",
        MemeExpression.LAUGH:     "Eyes squint, mouth wide open — exaggerated laughter",
        MemeExpression.CRY:       "Eyes squint downward, mouth corners droop — crying",
        MemeExpression.RAGE:      "Brows furrow, jaw clenches — intense anger",
        MemeExpression.SMUG:      "One brow raised, slight smirk — smug satisfaction",
        MemeExpression.SURPRISED: "Wide eyes, open mouth — mild surprise",
        MemeExpression.CUSTOM:    "Provide custom_params in the request body",
    }
    return {
        "expressions": [
            {"value": expr.value, "description": descriptions[expr]}
            for expr in MemeExpression
        ]
    }


@app.get("/status", tags=["health"])
async def status() -> dict:
    """Return ModelRegistry state and current VRAM usage."""
    from src.core import ModelRegistry
    reg = ModelRegistry.instance()
    models = reg.status()

    vram_info: dict = {}
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 3
            reserved  = torch.cuda.memory_reserved()  / 1024 ** 3
            total     = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            vram_info = {
                "total_gb":     round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb":  round(reserved, 2),
                "free_gb":      round(total - reserved, 2),
            }
    except Exception:
        vram_info = {"available": False}

    return {"models": models, "vram": vram_info}


# ── Dev server entry point ────────────────────────────────────────────────
def start() -> None:
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        workers=1,   # single process — ModelRegistry is not fork-safe
        reload=False,
    )


if __name__ == "__main__":
    start()
