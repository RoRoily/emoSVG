"""
Quick end-to-end smoke test — runs the full pipeline on a synthetic image
and prints a pass/fail summary. No model weights required.

Usage:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import tempfile
import time
import os
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import torch
from dotenv import load_dotenv

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from src.core.model_registry import ModelRegistry, ModelState
from src.modules.ip_extractor.extractor import IPExtractor
from src.modules.meme_animator.schemas import MemeExpression
from src.pipeline.full_pipeline import FullPipeline, FullPipelineRequest


def _make_mock_clip():
    proc = MagicMock()
    proc.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
    out = MagicMock()
    out.image_embeds = torch.zeros(1, 768)
    model = MagicMock()
    model.return_value = out
    model.to = MagicMock(return_value=model)
    return (proc, model)


def _models_root() -> Path | None:
    raw = os.getenv("MODELS_ROOT")
    return Path(raw).expanduser() if raw else None


def _model_dir(env_name: str, subdir: str) -> Path | None:
    raw = os.getenv(env_name)
    if raw:
        return Path(raw).expanduser()
    root = _models_root()
    return root / subdir if root else None


def _sam_checkpoint() -> Path | None:
    raw = os.getenv("SAM_MODEL_PATH")
    if raw:
        return Path(raw).expanduser()
    root = _models_root()
    return root / "sam" / "sam_vit_h_4b8939.pth" if root else None


def main() -> None:
    print("=" * 60)
    print("emoSVG Smoke Test")
    print("=" * 60)

    results: list[tuple[str, bool, str]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create synthetic test image
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:128, :] = (200, 100, 50)
        img[128:, :] = (50, 200, 100)
        img_path = tmp / "test_character.png"
        cv2.imwrite(str(img_path), img)

        # Build pipeline
        ModelRegistry.reset()
        pipeline = FullPipeline(
            output_root=tmp,
            live_portrait_path=_model_dir("LIVE_PORTRAIT_MODEL_PATH", "live_portrait"),
            toon_crafter_path=_model_dir("TOON_CRAFTER_MODEL_PATH", "toon_crafter"),
            sam_checkpoint=_sam_checkpoint(),
        )

        # Inject mock CLIP
        bundle = _make_mock_clip()
        entry = pipeline._registry._entries.get(IPExtractor.MODEL_ID)
        if entry:
            entry.module = bundle
            entry.state = ModelState.ON_CPU

        # Test 1: Animation only
        t0 = time.monotonic()
        try:
            req = FullPipelineRequest(
                source_image_path=img_path,
                expression=MemeExpression.SHOCK,
                fps=12, resolution=(64, 64),
                run_3d=False, run_svg=False,
            )
            result = pipeline.execute(req)
            assert result.animation.output_path.exists()
            assert result.animation.frame_count > 0
            results.append(("Animation (SHOCK)", True, f"{time.monotonic()-t0:.2f}s"))
        except Exception as exc:
            results.append(("Animation (SHOCK)", False, str(exc)))

        # Test 2: All expressions
        for expr in [MemeExpression.LAUGH, MemeExpression.CRY, MemeExpression.RAGE]:
            t0 = time.monotonic()
            try:
                req = FullPipelineRequest(
                    source_image_path=img_path,
                    expression=expr,
                    fps=12, resolution=(64, 64),
                    run_3d=False, run_svg=False,
                )
                result = pipeline.execute(req)
                assert result.animation.output_path.exists()
                results.append((f"Animation ({expr.value})", True, f"{time.monotonic()-t0:.2f}s"))
            except Exception as exc:
                results.append((f"Animation ({expr.value})", False, str(exc)))

        # Test 3: Full pipeline (3D + SVG)
        t0 = time.monotonic()
        try:
            req = FullPipelineRequest(
                source_image_path=img_path,
                expression=MemeExpression.SURPRISED,
                fps=12, resolution=(64, 64),
                run_3d=True, run_svg=True,
            )
            result = pipeline.execute(req)
            assert result.animation.output_path.exists()
            assert result.reconstruction is not None
            assert result.vectorization is not None
            results.append(("Full pipeline (3D+SVG)", True, f"{time.monotonic()-t0:.2f}s"))
        except Exception as exc:
            results.append(("Full pipeline (3D+SVG)", False, str(exc)))

        ModelRegistry.reset()

    # Print summary
    print()
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    for name, ok, info in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<35} {info}")
    print()
    print(f"Result: {passed}/{total} passed")
    print("=" * 60)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
