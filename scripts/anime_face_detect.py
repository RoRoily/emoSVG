"""Run hysts/anime-face-detector and emit bbox/keypoints JSON.

This script is meant to run inside the separate animeFaceDetector conda
environment. The main EmoSVG environment calls it as a subprocess so OpenMMLab
dependencies do not pollute the primary project environment.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect anime face landmarks as JSON.")
    parser.add_argument("image", type=Path, help="Input character image")
    parser.add_argument("--detector", default="yolov3", help="anime-face-detector model name")
    parser.add_argument("--device", default="cpu", help="cpu, cuda:0, ...")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    with contextlib.redirect_stdout(sys.stderr):
        from anime_face_detector import create_detector

        detector = create_detector(args.detector, device=args.device)
        predictions = detector(image)
    faces = []
    for pred in predictions:
        faces.append({
            "bbox": _to_list(pred["bbox"]),
            "keypoints": _to_list(pred["keypoints"]),
        })
    sys.stdout.write(json.dumps({"faces": faces}, ensure_ascii=False))


def _to_list(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
