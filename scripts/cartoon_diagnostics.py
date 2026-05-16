"""Run cartoon geometry/layer diagnostics without loading neural animation models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.cartoon_analyzer import CartoonFaceAnalyzer  # noqa: E402
from src.modules.cartoon_layer_parser import CartoonLayerParser  # noqa: E402
from src.pipeline.debug_artifacts import PipelineDebugWriter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Q-style cartoon landmarks and layers.")
    parser.add_argument("image", type=Path, nargs="?", help="Source character image")
    parser.add_argument(
        "--image",
        dest="image_option",
        type=Path,
        default=None,
        help="Source character image. Kept for compatibility with older notes.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs"), help="Output root")
    parser.add_argument("--width", type=int, default=512, help="Debug canvas width")
    parser.add_argument("--height", type=int, default=512, help="Debug canvas height")
    parser.add_argument(
        "--backend",
        default=None,
        choices=("auto", "external_anime_face_detector", "anime_face_detector", "heuristic"),
        help="Cartoon analyzer backend. Defaults to CARTOON_ANALYZER_BACKEND or auto.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for anime_face_detector, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--external-python",
        type=Path,
        default=None,
        help="Python executable from the separate animeFaceDetector environment.",
    )
    parser.add_argument(
        "--external-script",
        type=Path,
        default=None,
        help="External detector script. Defaults to scripts/anime_face_detect.py.",
    )
    parser.add_argument(
        "--external-timeout",
        type=float,
        default=None,
        help="Timeout in seconds for the external detector subprocess.",
    )
    parser.add_argument(
        "--strict-trained",
        action="store_true",
        help="Fail instead of falling back to heuristic when the trained detector is unavailable.",
    )
    args = parser.parse_args()
    if args.image_option is not None:
        args.image = args.image_option
    if args.image is None:
        parser.error("image path is required, either as positional IMAGE or --image IMAGE")
    return args


def main() -> None:
    args = parse_args()
    if not args.image.exists():
        raise SystemExit(f"Image does not exist: {args.image}")

    analyzer = CartoonFaceAnalyzer(
        backend=args.backend,
        device=args.device,
        external_python=args.external_python,
        external_script=args.external_script,
        external_timeout_seconds=args.external_timeout,
        allow_fallback=not args.strict_trained,
    )
    parser = CartoonLayerParser(analyzer=analyzer)

    bgr = CartoonFaceAnalyzer._load_bgr(args.image, (args.width, args.height))
    analysis = analyzer.analyze_bgr(bgr)
    layers = parser.parse_bgr(bgr, analysis.geometry, analysis.foreground_mask)

    writer = PipelineDebugWriter(args.output_root, args.image.stem, "diagnostics")
    writer.write_input(bgr)
    writer.write_analysis_overlay(bgr, analysis)
    writer.write_layer_overlay(bgr, layers)
    writer.write_layers(layers)
    writer.update_metrics(
        cartoon_analysis=analysis.to_dict(),
        cartoon_layers=layers.to_dict(),
    )
    writer.write_metrics()

    print(json.dumps({
        "debug_dir": str(writer.debug_dir),
        "backend_used": analysis.backend_used,
        "confidence": analysis.geometry.confidence,
        "layer_count": len(layers.layers),
        "warnings": analysis.geometry.warnings + layers.warnings,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
