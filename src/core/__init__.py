"""
emoSVG core infrastructure — device management, model registry, config loading.

Public surface:
    from src.core import ModelRegistry, DeviceManager, DeviceConfig, load_config
"""

from .config_loader import BaseConfig, load_config, load_raw
from .device_manager import DeviceConfig, DeviceManager, VRAMSnapshot
from .exceptions import (
    AnimationError,
    ConfigError,
    DeviceError,
    EmoSVGError,
    IPExtractionError,
    MissingModelWeightsError,
    ModelNotRegisteredError,
    PipelineError,
    ReconstructionError,
    VectorizationError,
    VRAMExhaustedError,
)
from .model_registry import ModelEntry, ModelRegistry, ModelState

__all__ = [
    # config
    "load_config",
    "load_raw",
    "BaseConfig",
    # device
    "DeviceConfig",
    "DeviceManager",
    "VRAMSnapshot",
    # registry
    "ModelRegistry",
    "ModelEntry",
    "ModelState",
    # exceptions
    "EmoSVGError",
    "ModelNotRegisteredError",
    "VRAMExhaustedError",
    "DeviceError",
    "ConfigError",
    "MissingModelWeightsError",
    "IPExtractionError",
    "AnimationError",
    "ReconstructionError",
    "VectorizationError",
    "PipelineError",
]
