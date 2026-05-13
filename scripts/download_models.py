"""
One-shot script to download all required model weights.

Usage:
    python scripts/download_models.py [--models all|ip_adapter|live_portrait|triposr|sam|toon_crafter]

Models downloaded:
    ip_adapter    — IP-Adapter FaceID (CLIP ViT-L/14 image encoder)
    triposr       — stabilityai/TripoSR via HuggingFace Hub
    sam           — SAM ViT-H checkpoint (~2.4 GB)
    live_portrait — KwaiVGI/LivePortrait via HuggingFace Hub
    toon_crafter  — ToonCrafter checkpoint from HuggingFace Hub (~10.5 GB)

Note: live_portrait requires the liveportrait package to be installed first.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "./models"))


def _download_file_with_resume(url: str, dest: Path, retries: int = 8) -> None:
    """Download a large file with basic retry and HTTP Range resume support."""
    import shutil
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing else "wb"
        try:
            logger.info(
                "Downloading %s -> %s%s",
                url,
                dest,
                f" (resume from {existing / 1024 ** 2:.1f} MB)" if existing else "",
            )
            with urllib.request.urlopen(req, timeout=60) as response, part.open(mode) as fh:
                shutil.copyfileobj(response, fh, length=1024 * 1024)
            part.replace(dest)
            return
        except Exception as exc:
            if attempt >= retries:
                raise
            wait = min(60, attempt * 5)
            logger.warning(
                "Download interrupted (%s). Retry %d/%d after %ds.",
                exc,
                attempt,
                retries,
                wait,
            )
            time.sleep(wait)


def _copy_if_newer(src: Path, dest: Path) -> None:
    """Copy a small metadata/config file if it is missing or stale."""
    import shutil

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return
    shutil.copy2(src, dest)


def _ensure_symlink(src: Path, dest: Path) -> None:
    """Create a convenience symlink when possible without duplicating huge files."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        dest.symlink_to(src)
        logger.info("Linked %s -> %s", dest, src)
    except OSError as exc:
        logger.warning(
            "Could not create symlink %s -> %s (%s). This is optional.",
            dest,
            src,
            exc,
        )


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
    url = os.getenv(
        "SAM_DOWNLOAD_URL",
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    )
    dest = MODELS_ROOT / "sam" / "sam_vit_h_4b8939.pth"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        logger.info("SAM checkpoint already exists at %s - skipping.", dest)
        return
    try:
        _download_file_with_resume(url, dest)
        logger.info("SAM downloaded to %s", dest)
    except (URLError, OSError, TimeoutError) as exc:
        logger.error("SAM download failed: %s", exc)
        logger.error(
            "Manual alternative:\n"
            "  mkdir -p %s\n"
            "  cd %s\n"
            "  wget -c %s\n"
            "or:\n"
            "  curl -L --retry 20 --retry-delay 5 -C - -o %s %s",
            dest.parent,
            dest.parent,
            url,
            dest,
            url,
        )


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
    import urllib.request
    import zipfile

    _use_ssh = os.getenv("GITHUB_SSH", "1") == "1"
    packages = [
        ("segment-anything", "facebookresearch/segment-anything"),
        ("TripoSR",          "VAST-AI-Research/TripoSR"),
        ("liveportrait",     "KwaiVGI/LivePortrait"),
        ("tooncrafter",      "ToonCrafter/ToonCrafter"),
    ]

    for name, repo in packages:
        tmpdir = Path(tempfile.mkdtemp(prefix=f"emosvg_{name}_"))
        try:
            cloned = False
            # Try zip download first (faster, no LFS overhead)
            zip_url = f"https://github.com/{repo}/archive/refs/heads/main.zip"
            zip_path = tmpdir / "repo.zip"
            extract_dir = tmpdir / "extracted"
            try:
                logger.info("Downloading %s as zip...", name)
                urllib.request.urlretrieve(zip_url, str(zip_path))
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(str(extract_dir))
                # extracted folder is named <repo>-main or <repo>-master
                subdirs = list(extract_dir.iterdir())
                src_dir = subdirs[0] if subdirs else extract_dir
                cloned = True
                logger.info("Downloaded %s via zip", name)
            except Exception as zip_exc:
                logger.warning("Zip download failed for %s (%s), falling back to git clone", name, zip_exc)
                src_dir = tmpdir / "repo"
                if _use_ssh:
                    repo_url = f"git@github.com:{repo}.git"
                else:
                    mirror = os.getenv("GITHUB_MIRROR", "https://github.com")
                    repo_url = f"{mirror}/{repo}.git"
                clone = subprocess.run(
                    ["git", "clone", "--depth=1", "--filter=blob:none", repo_url, str(src_dir)],
                    capture_output=True, text=True,
                )
                if clone.returncode != 0:
                    logger.warning("%s clone failed:\n%s", name, clone.stderr)
                    continue
                cloned = True

            if cloned:
                logger.info("Installing %s...", name)
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", str(src_dir)],
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
    Download ToonCrafter checkpoint and inference config.

    Layout after download:
        models/toon_crafter/
          model.ckpt                         (~10.5 GB)
          config.yaml                        (copy of inference_512_v1.0.yaml)
          configs/inference_512_v1.0.yaml    (official config layout)
          checkpoints/tooncrafter_512_interp_v1/model.ckpt -> ../../model.ckpt
    """
    dest = Path(os.getenv("TOON_CRAFTER_MODEL_PATH", str(MODELS_ROOT / "toon_crafter")))
    logger.info("Downloading ToonCrafter weights to %s ...", dest)

    model_repo = os.getenv("TOON_CRAFTER_REPO_ID", "Doubiiu/ToonCrafter")
    model_file = os.getenv("TOON_CRAFTER_MODEL_FILE", "model.ckpt")
    config_repo = os.getenv("TOON_CRAFTER_CONFIG_REPO_ID", "Doubiiu/tooncrafter")
    config_file = os.getenv("TOON_CRAFTER_CONFIG_FILE", "configs/inference_512_v1.0.yaml")
    config_url = os.getenv("TOON_CRAFTER_CONFIG_URL")

    try:
        from huggingface_hub import hf_hub_download

        dest.mkdir(parents=True, exist_ok=True)

        ckpt = dest / "model.ckpt"
        if ckpt.exists():
            logger.info("%s already exists — skipping.", ckpt)
        else:
            hf_hub_download(
                repo_id=model_repo,
                repo_type="model",
                filename=model_file,
                local_dir=str(dest),
            )
            downloaded_ckpt = dest / model_file
            if downloaded_ckpt.exists() and downloaded_ckpt != ckpt:
                downloaded_ckpt.replace(ckpt)
            logger.info("Downloaded %s", ckpt)

        config_dest = dest / "config.yaml"
        official_config_dest = dest / config_file
        if config_dest.exists():
            logger.info("%s already exists — skipping.", config_dest)
        elif config_url:
            _download_file_with_resume(config_url, config_dest)
            logger.info("Downloaded %s from %s", config_dest, config_url)
        else:
            try:
                downloaded_config = Path(
                    hf_hub_download(
                        repo_id=config_repo,
                        repo_type="space",
                        filename=config_file,
                        local_dir=str(dest / "_space"),
                    )
                )
            except Exception as space_exc:
                logger.warning(
                    "Failed to fetch ToonCrafter config from HF Space %s (%s). "
                    "Falling back to GitHub raw URL.",
                    config_repo,
                    space_exc,
                )
                github_config_url = os.getenv(
                    "TOON_CRAFTER_GITHUB_CONFIG_URL",
                    "https://raw.githubusercontent.com/Doubiiu/ToonCrafter/main/configs/inference_512_v1.0.yaml",
                )
                _download_file_with_resume(github_config_url, config_dest)
            else:
                _copy_if_newer(downloaded_config, config_dest)

        if config_dest.exists():
            _copy_if_newer(config_dest, official_config_dest)
        if ckpt.exists():
            _ensure_symlink(
                Path("../../model.ckpt"),
                dest / "checkpoints" / "tooncrafter_512_interp_v1" / "model.ckpt",
            )

        missing = [
            str(path)
            for path in [ckpt, config_dest]
            if not path.exists()
        ]
        if missing:
            logger.warning("ToonCrafter download may be incomplete. Missing:\n  %s", "\n  ".join(missing))
        else:
            logger.info("ToonCrafter weight verification passed: %s", dest)
    except Exception as exc:
        logger.error("ToonCrafter download failed: %s", exc)
        logger.error(
            "Manual alternative:\n"
            "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Doubiiu/ToonCrafter model.ckpt --repo-type model --local-dir %s\n"
            "  HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Doubiiu/tooncrafter configs/inference_512_v1.0.yaml --repo-type space --local-dir %s/_space\n"
            "  cp %s/_space/configs/inference_512_v1.0.yaml %s/config.yaml",
            dest,
            dest,
            dest,
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
