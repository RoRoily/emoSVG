"""
One-shot script to download all required model weights.

Usage:
    python scripts/download_models.py [--models all|ip_adapter|live_portrait|triposr|sam|toon_crafter]

Models downloaded:
    ip_adapter    — IP-Adapter FaceID (CLIP ViT-L/14 image encoder)
    triposr       — stabilityai/TripoSR via HuggingFace Hub
    sam           — SAM ViT-H checkpoint (~2.4 GB)
    live_portrait — KwaiVGI/LivePortrait via HuggingFace Hub
    toon_crafter  — ToonCrafter checkpoint from HuggingFace Hub (~8 GB)

Note: live_portrait requires the liveportrait package to be installed first.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "./models"))


def download_ip_adapter() -> None:
    logger.info("Downloading IP-Adapter FaceID (CLIP image encoder)...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="h94/IP-Adapter",
            allow_patterns=["models/image_encoder/**", "ip-adapter-faceid_sd15.bin"],
            local_dir=str(MODELS_ROOT / "ip_adapter"),
        )
        logger.info("IP-Adapter downloaded to %s", MODELS_ROOT / "ip_adapter")
    except Exception as exc:
        logger.error("IP-Adapter download failed: %s", exc)


def download_triposr() -> None:
    logger.info("Downloading TripoSR...")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="stabilityai/TripoSR",
            local_dir=str(MODELS_ROOT / "triposr"),
        )
        logger.info("TripoSR downloaded to %s", MODELS_ROOT / "triposr")
    except Exception as exc:
        logger.error("TripoSR download failed: %s", exc)


def download_sam() -> None:
    logger.info("Downloading SAM ViT-H checkpoint (~2.4 GB)...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    dest = MODELS_ROOT / "sam" / "sam_vit_h_4b8939.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("SAM checkpoint already exists at %s — skipping.", dest)
        return
    logger.info("Downloading from %s ...", url)
    urllib.request.urlretrieve(url, str(dest))
    logger.info("SAM downloaded to %s", dest)


def download_live_portrait() -> None:
    """
    Download LivePortrait weights from HuggingFace Hub.

    Layout after download:
        models/live_portrait/
          pretrained_weights/
            liveportrait/
              base_models/
                appearance_feature_extractor.pth  (~84 MB)
                motion_extractor.pth              (~109 MB)
                warping_module.pth                (~176 MB)
                spade_generator.pth               (~1.5 GB)
              retargeting_models/
                stitching_retargeting_module.pth  (~137 MB)
          src/config/models.yaml
    Total: ~2.1 GB
    """
    dest = MODELS_ROOT / "live_portrait"
    logger.info("Downloading LivePortrait weights to %s ...", dest)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="KwaiVGI/LivePortrait",
            repo_type="model",
            local_dir=str(dest),
            # Only download weights and config — skip large video examples
            ignore_patterns=["*.mp4", "*.gif", "assets/**", "docs/**"],
        )
        logger.info("LivePortrait weights downloaded to %s", dest)
        _verify_live_portrait(dest)
    except Exception as exc:
        logger.error("LivePortrait download failed: %s", exc)
        logger.error(
            "Manual alternative:\n"
            "  git clone https://huggingface.co/KwaiVGI/LivePortrait %s",
            dest,
        )


def _verify_live_portrait(dest: Path) -> None:
    """Check that all required weight files are present after download."""
    required = [
        "pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth",
        "pretrained_weights/liveportrait/base_models/motion_extractor.pth",
        "pretrained_weights/liveportrait/base_models/warping_module.pth",
        "pretrained_weights/liveportrait/base_models/spade_generator.pth",
        "pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth",
    ]
    missing = [f for f in required if not (dest / f).exists()]
    if missing:
        logger.warning(
            "LivePortrait download may be incomplete. Missing files:\n  %s",
            "\n  ".join(missing),
        )
    else:
        logger.info("LivePortrait weight verification passed (%d files OK).", len(required))


def install_git_packages() -> None:
    """Install packages that are only available via git."""
    import subprocess
    import tempfile
    import shutil

    _use_ssh = os.getenv("GITHUB_SSH", "1") == "1"
    packages = [
        ("segment-anything", "facebookresearch/segment-anything"),
        ("TripoSR",          "VAST-AI-Research/TripoSR"),
        ("liveportrait",     "KwaiVGI/LivePortrait"),
        ("tooncrafter",      "ToonCrafter/ToonCrafter"),
    ]

    for name, repo in packages:
        if _use_ssh:
            repo_url = f"git@github.com:{repo}.git"
        else:
            _mirror = os.getenv("GITHUB_MIRROR", "https://github.com")
            repo_url = f"{_mirror}/{repo}.git"
        tmpdir = Path(tempfile.mkdtemp(prefix=f"emosvg_{name}_"))
        try:
            logger.info("Cloning %s into %s ...", name, tmpdir)
            clone = subprocess.run(
                ["git", "clone", "--depth=1", repo_url, str(tmpdir)],
                capture_output=True, text=True,
            )
            if clone.returncode != 0:
                logger.warning("%s clone failed:\n%s", name, clone.stderr)
                continue
            logger.info("Installing %s from local clone...", name)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", str(tmpdir)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                logger.info("%s installed OK", name)
            else:
                logger.warning("%s install failed:\n%s", name, result.stderr)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def download_toon_crafter() -> None:
    """
    Download ToonCrafter checkpoint from HuggingFace Hub.

    Layout after download:
        models/toon_crafter/
          model.ckpt    (~8 GB)
          config.yaml
    """
    dest = MODELS_ROOT / "toon_crafter"
    logger.info("Downloading ToonCrafter weights to %s ...", dest)
    try:
        from huggingface_hub import hf_hub_download
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ["model.ckpt", "config.yaml"]:
            out = dest / filename
            if out.exists():
                logger.info("%s already exists — skipping.", out)
                continue
            hf_hub_download(
                repo_id="Doubiiu/ToonCrafter",
                filename=filename,
                local_dir=str(dest),
            )
            logger.info("Downloaded %s", out)
        logger.info("ToonCrafter weights downloaded to %s", dest)
    except Exception as exc:
        logger.error("ToonCrafter download failed: %s", exc)
        logger.error(
            "Manual alternative:\n"
            "  huggingface-cli download Doubiiu/ToonCrafter --local-dir %s",
            dest,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download emoSVG model weights")
    parser.add_argument(
        "--models",
        default="all",
        choices=[
            "all", "ip_adapter", "live_portrait", "triposr",
            "sam", "toon_crafter", "git_packages",
        ],
        help="Which models to download (default: all)",
    )
    args = parser.parse_args()

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.models in ("all", "git_packages"):
        install_git_packages()
    if args.models in ("all", "ip_adapter"):
        download_ip_adapter()
    if args.models in ("all", "triposr"):
        download_triposr()
    if args.models in ("all", "sam"):
        download_sam()
    if args.models in ("all", "live_portrait"):
        download_live_portrait()
    if args.models in ("all", "toon_crafter"):
        download_toon_crafter()

    logger.info("Done. Model weights are in: %s", MODELS_ROOT.resolve())


if __name__ == "__main__":
    main()
