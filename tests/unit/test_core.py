"""
Unit tests for src/core -- DeviceManager and ModelRegistry.
Runs CPU-only, no real model weights needed.
"""
from __future__ import annotations
import threading, time
import pytest
import torch
import torch.nn as nn
from src.core.device_manager import DeviceConfig, DeviceManager
from src.core.model_registry import ModelRegistry, ModelState

@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry.reset()
    yield
    ModelRegistry.reset()

def _cpu_config(budget_gb: float = 100.0) -> DeviceConfig:
    return DeviceConfig(device="cpu", torch_dtype=torch.float32,
                        vram_budget_gb=budget_gb, safety_margin_gb=0.0, offload_target="cpu")

def _make_registry(budget_gb: float = 100.0) -> ModelRegistry:
    return ModelRegistry.instance(device_config=_cpu_config(budget_gb))

def _tiny_loader() -> nn.Module:
    return nn.Linear(4, 4)

class TestDeviceManager:
    def test_dtype_float32_on_cpu(self):
        dm = DeviceManager(DeviceConfig(device="cpu", torch_dtype=torch.float16))
        assert dm.dtype == torch.float32

    def test_snapshot_none_on_cpu(self):
        assert DeviceManager(_cpu_config()).snapshot() is None

    def test_can_fit_always_true_on_cpu(self):
        assert DeviceManager(_cpu_config(budget_gb=1.0)).can_fit(999.0) is True

    def test_effective_budget(self):
        cfg = DeviceConfig(vram_budget_gb=10.0, safety_margin_gb=1.5)
        assert cfg.effective_budget_gb == pytest.approx(8.5)

class TestSingleton:
    def test_same_instance(self):
        assert _make_registry() is ModelRegistry.instance()

    def test_reset_new_instance(self):
        r1 = _make_registry()
        ModelRegistry.reset()
        r2 = _make_registry()
        assert r1 is not r2

    def test_thread_safety(self):
        instances = []
        barrier = threading.Barrier(8)
        def _get():
            barrier.wait()
            instances.append(ModelRegistry.instance(_cpu_config()))
        threads = [threading.Thread(target=_get) for _ in range(8)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert len({id(i) for i in instances}) == 1

class TestLifecycle:
    def test_register_and_load(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        assert isinstance(reg.load_to_gpu("tiny"), nn.Linear)

    def test_state_transitions(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        assert reg.status()["tiny"]["state"] == "UNLOADED"
        reg.load_to_gpu("tiny")
        assert reg.status()["tiny"]["state"] == "ON_GPU"
        reg.offload_to_cpu("tiny")
        assert reg.status()["tiny"]["state"] == "ON_CPU"

    def test_load_unregistered_raises(self):
        with pytest.raises(KeyError):
            _make_registry().load_to_gpu("nonexistent")

    def test_duplicate_register_noop(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        assert len(reg.status()) == 1

    def test_loader_called_once(self):
        count = {"n": 0}
        def _loader():
            count["n"] += 1
            return nn.Linear(4, 4)
        reg = _make_registry()
        reg.register("c", _loader, estimated_vram_gb=0.001)
        reg.load_to_gpu("c")
        reg.offload_to_cpu("c")
        reg.load_to_gpu("c")
        assert count["n"] == 1

    def test_offload_idempotent(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        reg.load_to_gpu("tiny")
        reg.offload_to_cpu("tiny")
        reg.offload_to_cpu("tiny")
        assert reg.status()["tiny"]["state"] == "ON_CPU"

class TestModelContext:
    def test_yields_module_on_gpu(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        with reg.model_context("tiny") as m:
            assert isinstance(m, nn.Module)
            assert reg.status()["tiny"]["state"] == "ON_GPU"

    def test_offload_after_true(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        with reg.model_context("tiny", offload_after=True): pass
        assert reg.status()["tiny"]["state"] == "ON_CPU"

    def test_offload_after_false(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        with reg.model_context("tiny", offload_after=False): pass
        assert reg.status()["tiny"]["state"] == "ON_GPU"

    def test_offload_on_exception(self):
        reg = _make_registry()
        reg.register("tiny", _tiny_loader, estimated_vram_gb=0.001)
        with pytest.raises(RuntimeError):
            with reg.model_context("tiny", offload_after=True):
                raise RuntimeError("boom")
        assert reg.status()["tiny"]["state"] == "ON_CPU"

class TestLRUEviction:
    def test_lru_evicts_oldest(self):
        # _evict_until_fits call sequence:
        #   call 1 (entry check)  → False  → enter eviction loop
        #   call 2 (loop guard)   → False  → evict candidate "a"
        #   call 3 (final check)  → True   → no MemoryError, proceed
        reg = _make_registry(budget_gb=0.0)
        dm = reg.device_manager

        reg.register("a", _tiny_loader, estimated_vram_gb=1.0)
        reg.register("b", _tiny_loader, estimated_vram_gb=1.0)

        reg.load_to_gpu("a")
        reg._entries["a"].last_used_ts = time.monotonic() - 10

        call_n = {"n": 0}
        def _patched(needed_gb: float) -> bool:
            call_n["n"] += 1
            # calls 1 and 2 → False (trigger + sustain eviction loop)
            # call 3 onward  → True  (post-eviction final check passes)
            return call_n["n"] >= 3
        dm.can_fit = _patched  # type: ignore[method-assign]

        reg.load_to_gpu("b")

        assert reg.status()["a"]["state"] == "ON_CPU"
        assert reg.status()["b"]["state"] == "ON_GPU"

    def test_memory_error_no_candidates(self):
        reg = _make_registry(budget_gb=0.0)
        reg.device_manager.can_fit = lambda _: False  # type: ignore[method-assign]
        reg.register("only", _tiny_loader, estimated_vram_gb=999.0)
        with pytest.raises(MemoryError):
            reg.load_to_gpu("only")

class TestStatus:
    def test_all_registered_present(self):
        reg = _make_registry()
        reg.register("a", _tiny_loader, estimated_vram_gb=0.1)
        reg.register("b", _tiny_loader, estimated_vram_gb=0.2)
        s = reg.status()
        assert set(s.keys()) == {"a", "b"}
        assert s["a"]["estimated_vram_gb"] == pytest.approx(0.1)
