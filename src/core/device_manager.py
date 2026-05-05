"""
VRAM / device state management.

Responsibilities:
- Query current VRAM usage and free headroom
- Decide whether a model can be loaded onto GPU or must stay on CPU
- Move tensors/modules between devices
- Provide a consistent dtype policy across the process
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class VRAMSnapshot:
    total_gb: float
    used_gb: float
    free_gb: float
    reserved_gb: float  # PyTorch allocator reserved but not yet used

    @property
    def available_gb(self) -> float:
        """Conservative estimate: free minus allocator overhead."""
        return max(0.0, self.free_gb - 0.2)


@dataclass
class DeviceConfig:
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    vram_budget_gb: float = 10.0
    safety_margin_gb: float = 0.5
    offload_target: str = "cpu"  # "cpu" | "disk" (disk NYI)

    @property
    def effective_budget_gb(self) -> float:
        return self.vram_budget_gb - self.safety_margin_gb


class DeviceManager:
    """
    Singleton-friendly helper that wraps all device/VRAM queries.

    ModelRegistry owns the single instance; modules should never
    instantiate this directly — obtain it via ModelRegistry.device_manager.
    """

    def __init__(self, config: DeviceConfig) -> None:
        self.config = config
        self._device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        if config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available — falling back to CPU.")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def offload_device(self) -> torch.device:
        return torch.device(self.config.offload_target)

    @property
    def dtype(self) -> torch.dtype:
        # On CPU, float16 matmul is unsupported on most hardware; fall back.
        if self._device.type == "cpu":
            return torch.float32
        return self.config.torch_dtype

    def snapshot(self) -> Optional[VRAMSnapshot]:
        """Return current VRAM state, or None when running on CPU."""
        if self._device.type != "cuda":
            return None
        idx = self._device.index or 0
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory / 1024**3
        reserved = torch.cuda.memory_reserved(idx) / 1024**3
        allocated = torch.cuda.memory_allocated(idx) / 1024**3
        free_in_reserved = reserved - allocated
        free_outside = (props.total_memory - torch.cuda.memory_reserved(idx)) / 1024**3
        free = free_in_reserved + free_outside
        return VRAMSnapshot(
            total_gb=total,
            used_gb=allocated,
            free_gb=free,
            reserved_gb=reserved,
        )

    def can_fit(self, estimated_gb: float) -> bool:
        """
        Return True if loading a model of `estimated_gb` would stay within
        the configured budget.
        """
        snap = self.snapshot()
        if snap is None:
            # CPU — always "fits" (RAM is managed by OS)
            return True
        headroom = min(snap.available_gb, self.config.effective_budget_gb - snap.used_gb)
        fits = headroom >= estimated_gb
        logger.debug(
            "can_fit(%.2f GB): headroom=%.2f GB → %s",
            estimated_gb,
            headroom,
            "YES" if fits else "NO",
        )
        return fits

    def move_to_gpu(self, module: nn.Module) -> nn.Module:
        """Move a module to the primary GPU device.

        If `module` is not an nn.Module (e.g. a tuple bundle of processor+model),
        each nn.Module element is moved individually and the container is returned as-is.
        """
        if isinstance(module, nn.Module):
            return module.to(device=self._device, dtype=self.dtype)
        if isinstance(module, (tuple, list)):
            moved = [
                m.to(device=self._device, dtype=self.dtype) if isinstance(m, nn.Module) else m
                for m in module
            ]
            return type(module)(moved)  # type: ignore[call-arg]
        return module  # passthrough for non-tensor objects

    def move_to_offload(self, module: nn.Module) -> nn.Module:
        """Move a module to the offload device (CPU by default)."""
        if isinstance(module, nn.Module):
            module.to(device=self.offload_device, dtype=torch.float32)
        elif isinstance(module, (tuple, list)):
            for m in module:
                if isinstance(m, nn.Module):
                    m.to(device=self.offload_device, dtype=torch.float32)
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        return module

    def log_vram(self, tag: str = "") -> None:
        snap = self.snapshot()
        if snap is None:
            logger.info("[VRAM %s] running on CPU", tag)
            return
        logger.info(
            "[VRAM %s] used=%.2f GB | free=%.2f GB | reserved=%.2f GB | total=%.2f GB",
            tag,
            snap.used_gb,
            snap.free_gb,
            snap.reserved_gb,
            snap.total_gb,
        )

    @staticmethod
    def empty_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Factory helpers ───────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "DeviceManager":
        """Build a DeviceManager from environment variables / defaults."""
        import os

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        cfg = DeviceConfig(
            device=os.getenv("DEVICE", "cuda"),
            torch_dtype=dtype_map.get(os.getenv("TORCH_DTYPE", "float16"), torch.float16),
            vram_budget_gb=float(os.getenv("VRAM_BUDGET_GB", "10.0")),
            safety_margin_gb=0.5,
            offload_target="cpu",
        )
        return cls(cfg)
