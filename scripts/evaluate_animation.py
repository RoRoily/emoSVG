"""Evaluate a generated animation against its source image with lightweight metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modules.cartoon_evaluator import evaluate_animation_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate emoSVG animation quality proxies.")
    parser.add_argument("--source", required=True, type=Path, help="Original source image")
    parser.add_argument("--animation", required=True, type=Path, help="Generated GIF/MP4/WebP")
    parser.add_argument("--output", type=Path, default=None, help="Optional metrics JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = cv2.imread(str(args.source))
    if source is None:
        raise SystemExit(f"Failed to read source image: {args.source}")
    frames = _read_animation(args.animation)
    metrics = evaluate_animation_frames(frames, source_bgr=source)
    text = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


def _read_animation(path: Path):
    if not path.exists():
        raise SystemExit(f"Animation does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix in {".gif", ".webp"}:
        frames = []
        for frame in imageio.mimread(str(path)):
            frames.append(_imageio_frame_to_bgr(frame))
        return frames

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise SystemExit(f"No frames decoded from animation: {path}")
    return frames


def _imageio_frame_to_bgr(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    main()
