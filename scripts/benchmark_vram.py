"""
VRAM benchmark: loads each model sequentially and reports peak VRAM usage.

Usage:
    python scripts/benchmark_vram.py

Requires a CUDA GPU and all model weights to be present.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))


def _vram_gb() -> float:
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1024**3


def benchmark_model(name: str, loader_fn, registry) -> dict:
    import torch
    torch.cuda.empty_cache()
    before = _vram_gb()
    t0 = time.monotonic()
    try:
        with registry.model_context(name, offload_after=True):
            peak = _vram_gb()
        elapsed = time.monotonic() - t0
        return {"model": name, "peak_vram_gb": round(peak - before, 3), "load_time_s": round(elapsed, 2), "status": "ok"}
    except Exception as exc:
        return {"model": name, "peak_vram_gb": None, "load_time_s": None, "status": f"FAILED: {exc}"}


def main() -> None:
    import os
    import torch
    from src.core import ModelRegistry, DeviceConfig

    if not torch.cuda.is_available():
        print("No CUDA GPU detected — benchmark requires a GPU.")
        sys.exit(1)

    models_root = Path(os.getenv("MODELS_ROOT", "./models"))
    registry = ModelRegistry.instance(DeviceConfig(device="cuda", torch_dtype=torch.float16))

    # Register all models
    from src.modules.ip_extractor.extractor import IPExtractor
    from src.modules.meme_animator.live_portrait import LivePortraitWrapper
    from src.modules.reconstructor_3d.reconstructor import Reconstructor3D
    from src.modules.svg_vectorizer.segmentor import Segmentor

    IPExtractor(ip_adapter_path=models_root / "ip_adapter", registry=registry)
    LivePortraitWrapper(model_path=models_root / "live_portrait", registry=registry)
    Reconstructor3D(model_id=str(models_root / "triposr"), output_dir=Path("outputs/meshes"), registry=registry)
    Segmentor(sam_checkpoint=models_root / "sam" / "sam_vit_h_4b8939.pth", registry=registry)

    model_ids = [
        IPExtractor.MODEL_ID,
        LivePortraitWrapper.MODEL_ID,
        Reconstructor3D.MODEL_ID,
        Segmentor.MODEL_ID,
    ]

    print("
" + "=" * 60)
    print("emoSVG VRAM Benchmark")
    print("=" * 60)
    results = [benchmark_model(mid, None, registry) for mid in model_ids]
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['model']:<25} peak={r['peak_vram_gb']:.2f} GB  load={r['load_time_s']:.2f}s")
        else:
            print(f"  {r['model']:<25} {r['status']}")
    print("=" * 60)

    ModelRegistry.reset()


if __name__ == "__main__":
    main()
