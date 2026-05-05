"""
Central model registry — the single source of truth for all loaded models.

Design:
- Singleton: only one instance exists per process (thread-safe via a lock).
- Each model is tracked as a ModelEntry with its load state and estimated VRAM cost.
- load() follows an LRU-style eviction policy: when VRAM is tight, the least
  recently used GPU-resident model is offloaded to CPU before loading the new one.
- Callers receive a context manager (model_context) that guarantees the model is
  on GPU for the duration of the block, then optionally offloads it afterward.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generator, Optional

import torch
import torch.nn as nn

from .device_manager import DeviceConfig, DeviceManager

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

class ModelState(Enum):
    UNLOADED  = auto()   # weights not in memory at all
    ON_CPU    = auto()   # weights in RAM, not on GPU
    ON_GPU    = auto()   # weights on CUDA device, ready to infer


@dataclass
class ModelEntry:
    model_id: str
    loader: Callable[[], nn.Module]   # zero-arg factory that returns the model
    estimated_vram_gb: float          # rough upper bound for can_fit() checks
    state: ModelState = ModelState.UNLOADED
    module: Optional[nn.Module] = None
    last_used_ts: float = field(default_factory=time.monotonic)

    def touch(self) -> None:
        self.last_used_ts = time.monotonic()


# ── Singleton registry ────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Global registry for all heavy models in the pipeline.

    Usage
    -----
    registry = ModelRegistry.instance()
    registry.register("live_portrait", loader_fn, estimated_vram_gb=4.5)

    with registry.model_context("live_portrait") as model:
        output = model(input_tensor)
    """

    _instance: Optional["ModelRegistry"] = None
    _init_lock: threading.Lock = threading.Lock()

    def __init__(self, device_manager: DeviceManager) -> None:
        self._dm = device_manager
        self._entries: OrderedDict[str, ModelEntry] = OrderedDict()
        self._lock = threading.Lock()   # guards _entries mutations

    # ── Singleton access ──────────────────────────────────────────────────────

    @classmethod
    def instance(
        cls,
        device_config: Optional[DeviceConfig] = None,
    ) -> "ModelRegistry":
        """
        Return (or create) the process-wide singleton.

        Pass `device_config` only on the very first call; subsequent calls
        ignore it and return the existing instance.
        """
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    dm = DeviceManager(device_config or DeviceConfig())
                    cls._instance = cls(dm)
                    logger.info(
                        "ModelRegistry initialised — device=%s dtype=%s budget=%.1f GB",
                        dm.device,
                        dm.dtype,
                        dm.config.effective_budget_gb,
                    )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Tear down the singleton (useful in tests)."""
        with cls._init_lock:
            if cls._instance is not None:
                cls._instance._unload_all()
            cls._instance = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def device_manager(self) -> DeviceManager:
        return self._dm

    def register(
        self,
        model_id: str,
        loader: Callable[[], nn.Module],
        estimated_vram_gb: float,
    ) -> None:
        """
        Declare a model. Does NOT load weights yet — loading is lazy.

        Parameters
        ----------
        model_id:
            Unique string key (e.g. "live_portrait", "triposr").
        loader:
            Zero-argument callable that constructs and returns the nn.Module.
            Will be called at most once; result is cached in the entry.
        estimated_vram_gb:
            Conservative upper-bound VRAM estimate used for eviction decisions.
        """
        with self._lock:
            if model_id in self._entries:
                logger.warning("Model '%s' already registered — skipping.", model_id)
                return
            self._entries[model_id] = ModelEntry(
                model_id=model_id,
                loader=loader,
                estimated_vram_gb=estimated_vram_gb,
            )
            logger.debug("Registered model '%s' (est. %.2f GB).", model_id, estimated_vram_gb)

    def load_to_gpu(self, model_id: str) -> nn.Module:
        """
        Ensure `model_id` is loaded and resident on GPU.

        Steps:
        1. If already ON_GPU → touch LRU timestamp and return.
        2. If UNLOADED → call loader(), store module, mark ON_CPU.
        3. Evict LRU GPU models until there is enough headroom.
        4. Move module to GPU.
        """
        with self._lock:
            entry = self._get_entry(model_id)

            if entry.state == ModelState.ON_GPU:
                entry.touch()
                return entry.module  # type: ignore[return-value]

            # Step 2 — materialise weights if needed
            if entry.state == ModelState.UNLOADED:
                logger.info("Loading weights for '%s' …", model_id)
                entry.module = entry.loader()
                entry.state = ModelState.ON_CPU
                logger.info("'%s' loaded to CPU.", model_id)

            # Step 3 — free VRAM if necessary
            self._evict_until_fits(model_id, entry.estimated_vram_gb)

            # Step 4 — move to GPU
            self._dm.log_vram(f"before loading {model_id}")
            entry.module = self._dm.move_to_gpu(entry.module)  # type: ignore[arg-type]
            entry.state = ModelState.ON_GPU
            entry.touch()
            self._dm.log_vram(f"after loading {model_id}")
            logger.info("'%s' is now ON_GPU.", model_id)
            return entry.module  # type: ignore[return-value]

    def offload_to_cpu(self, model_id: str) -> None:
        """Explicitly move a model from GPU to CPU."""
        with self._lock:
            entry = self._get_entry(model_id)
            if entry.state != ModelState.ON_GPU:
                return
            logger.info("Offloading '%s' to CPU …", model_id)
            entry.module = self._dm.move_to_offload(entry.module)  # type: ignore[arg-type]
            entry.state = ModelState.ON_CPU
            self._dm.log_vram(f"after offloading {model_id}")

    @contextmanager
    def model_context(
        self,
        model_id: str,
        offload_after: bool = True,
    ) -> Generator[nn.Module, None, None]:
        """
        Context manager that guarantees the model is on GPU inside the block.

        Parameters
        ----------
        offload_after:
            If True (default), offload back to CPU when the block exits.
            Set to False when you plan to call the same model again soon.

        Example
        -------
        with registry.model_context("triposr") as model:
            mesh = model(image_tensor)
        """
        module = self.load_to_gpu(model_id)
        try:
            yield module
        finally:
            if offload_after:
                self.offload_to_cpu(model_id)

    def status(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all registered models and their states."""
        with self._lock:
            return {
                mid: {
                    "state": entry.state.name,
                    "estimated_vram_gb": entry.estimated_vram_gb,
                    "last_used_ts": entry.last_used_ts,
                }
                for mid, entry in self._entries.items()
            }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_entry(self, model_id: str) -> ModelEntry:
        if model_id not in self._entries:
            raise KeyError(
                f"Model '{model_id}' is not registered. "
                "Call registry.register() before loading."
            )
        return self._entries[model_id]

    def _evict_until_fits(self, requesting_id: str, needed_gb: float) -> None:
        """
        Offload GPU-resident models (LRU first) until `needed_gb` fits within
        the VRAM budget.  The requesting model itself is never evicted.
        """
        if self._dm.can_fit(needed_gb):
            return

        # Build LRU-sorted list of eviction candidates
        candidates = sorted(
            [
                e for mid, e in self._entries.items()
                if e.state == ModelState.ON_GPU and mid != requesting_id
            ],
            key=lambda e: e.last_used_ts,  # oldest first
        )

        for candidate in candidates:
            if self._dm.can_fit(needed_gb):
                break
            logger.info(
                "VRAM pressure: evicting '%s' (LRU, est. %.2f GB) to make room for '%s'.",
                candidate.model_id,
                candidate.estimated_vram_gb,
                requesting_id,
            )
            # Bypass the public lock — we already hold it
            candidate.module = self._dm.move_to_offload(candidate.module)  # type: ignore[arg-type]
            candidate.state = ModelState.ON_CPU

        if not self._dm.can_fit(needed_gb):
            snap = self._dm.snapshot()
            free = snap.available_gb if snap else float("inf")
            raise MemoryError(
                f"Cannot load '{requesting_id}' (needs ~{needed_gb:.2f} GB): "
                f"only {free:.2f} GB available after evicting all other models. "
                "Consider reducing vram_budget_gb or using a smaller model variant."
            )

    def _unload_all(self) -> None:
        """Release all model references (used by reset())."""
        for entry in self._entries.values():
            if entry.module is not None:
                if entry.state == ModelState.ON_GPU:
                    entry.module.cpu()
                del entry.module
                entry.module = None
                entry.state = ModelState.UNLOADED
        DeviceManager.empty_cache()
        logger.info("ModelRegistry: all models unloaded.")
