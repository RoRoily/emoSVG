"""
YAML config loader with Pydantic validation and environment variable interpolation.

Usage:
    cfg = load_config("configs/base.yaml")
    meme_cfg = load_config("configs/meme_animation.yaml", BaseConfig)
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, field_validator

from .exceptions import ConfigError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Matches ${ENV_VAR} or ${ENV_VAR:-default}
_ENV_RE = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _interpolate(value: Any) -> Any:
    """Recursively expand ${ENV_VAR} placeholders in strings."""
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var, default = m.group(1), m.group(2)
            result = os.environ.get(var, default or "")
            if not result:
                logger.warning("Config: env var '%s' is not set and has no default.", var)
            return result
        return _ENV_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    return value


def load_raw(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and interpolate environment variables."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p.resolve()}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse {p}: {exc}") from exc
    return _interpolate(raw)


def load_config(path: str | Path, model: type[T] | None = None) -> Any:
    """
    Load and optionally validate a YAML config file.

    Parameters
    ----------
    path:   Path to the YAML file.
    model:  Optional Pydantic model class for validation.
            If None, returns the raw dict.
    """
    raw = load_raw(path)
    if model is None:
        return raw
    try:
        return model(**raw)
    except Exception as exc:
        raise ConfigError(f"Config validation failed for {path}: {exc}") from exc


# ── Pydantic models for each config file ─────────────────────────────────────

class VRAMConfig(BaseModel):
    budget_gb: float = 10.0
    safety_margin_gb: float = 0.5
    offload_target: str = "cpu"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"


class BaseConfig(BaseModel):
    device: str = "cuda"
    torch_dtype: str = "float16"
    vram: VRAMConfig = VRAMConfig()
    models_root: str = "./models"
    output_root: str = "./outputs"
    logging: LoggingConfig = LoggingConfig()

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        if v not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{v}'")
        return v

    @field_validator("torch_dtype")
    @classmethod
    def _validate_dtype(cls, v: str) -> str:
        if v not in ("float16", "bfloat16", "float32"):
            raise ValueError(f"torch_dtype must be float16/bfloat16/float32, got '{v}'")
        return v
