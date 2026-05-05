"""
FastAPI dependency injection.

The FullPipeline singleton is created once at startup and shared across
all requests so the ModelRegistry (and its VRAM budget) is never duplicated.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.core import ModelRegistry
from src.pipeline.full_pipeline import FullPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> FullPipeline:
    """Return the process-wide FullPipeline singleton."""
    return FullPipeline.from_config("configs/base.yaml")


@lru_cache(maxsize=1)
def get_registry() -> ModelRegistry:
    return ModelRegistry.instance()
