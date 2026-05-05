"""
Custom exception hierarchy for emoSVG.

All pipeline errors inherit from EmoSVGError so callers can catch the
entire family with a single except clause when needed.
"""

from __future__ import annotations


class EmoSVGError(Exception):
    """Base class for all emoSVG errors."""


# ── Registry / device errors ──────────────────────────────────────────────────

class ModelNotRegisteredError(EmoSVGError, KeyError):
    """Raised when a model_id is requested but was never registered."""


class VRAMExhaustedError(EmoSVGError, MemoryError):
    """
    Raised when VRAM cannot be freed sufficiently to load a requested model,
    even after evicting all other GPU-resident models.
    """


class DeviceError(EmoSVGError):
    """Raised for device configuration or transfer failures."""


# ── Configuration errors ──────────────────────────────────────────────────────

class ConfigError(EmoSVGError):
    """Raised when a config file is missing, malformed, or fails validation."""


class MissingModelWeightsError(EmoSVGError):
    """Raised when a model checkpoint path does not exist on disk."""


# ── Module-level errors ───────────────────────────────────────────────────────

class IPExtractionError(EmoSVGError):
    """Raised when IP feature extraction fails."""


class AnimationError(EmoSVGError):
    """Raised when Meme animation generation fails."""


class ReconstructionError(EmoSVGError):
    """Raised when 3D reconstruction fails."""


class VectorizationError(EmoSVGError):
    """Raised when SVG vectorization fails."""


# ── Pipeline errors ───────────────────────────────────────────────────────────

class PipelineError(EmoSVGError):
    """Raised for errors in pipeline orchestration (not within a single module)."""
