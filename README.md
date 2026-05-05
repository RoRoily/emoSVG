# emoSVG — IP Meme Animation Engine

IP character image → Squash-and-Stretch meme animation → 3D mesh → SVG vector

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Download Model Weights](#download-model-weights)
- [Running the Server](#running-the-server)
- [API Quick Start](#api-quick-start)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Overview

emoSVG is a single-process multi-modal generative AI pipeline. Given a flat IP character image (PNG with or without transparency), it produces:

1. **Meme animation** — Squash-and-Stretch GIF/MP4/WebP driven by LivePortrait
2. **3D mesh** — OBJ + GLB via TripoSR single-image reconstruction (optional)
3. **SVG vector** — layered SVG via SAM segmentation + Bézier fitting (optional)

Every module has a CPU-only fallback so the full pipeline runs without a GPU or model weights (useful for development and CI).

---

## Requirements

| Requirement | Version |
|---|---|
| Python | 3.10 or 3.11 |
| CUDA (optional) | 11.8 or 12.x |
| GPU VRAM (optional) | 10 GB+ recommended |
| RAM | 16 GB+ |

Python packages are listed in `pyproject.toml`. The heavy ML packages (LivePortrait, TripoSR, ToonCrafter, SAM) are installed separately via git URLs.

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url> emoSVG
cd emoSVG
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -e ".[dev]"
```

If `pyproject.toml` does not declare extras yet, install the core deps directly:

```bash
pip install fastapi uvicorn[standard] pydantic python-dotenv \
            torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
            transformers diffusers accelerate \
            opencv-python-headless pillow imageio imageio-ffmpeg \
            trimesh scipy numpy
```

### 4. Install git-only packages

```bash
python scripts/download_models.py --models git_packages
```

This installs:
- `segment-anything` (SAM)
- `TripoSR`
- `liveportrait`
- `tooncrafter`

---

## Download Model Weights

All weights are downloaded by a single script. You can download everything at once or pick individual models.

```bash
# Download everything (~20 GB total)
python scripts/download_models.py --models all

# Individual models
python scripts/download_models.py --models ip_adapter      # ~1 GB
python scripts/download_models.py --models live_portrait   # ~2.1 GB
python scripts/download_models.py --models triposr         # ~1.5 GB
python scripts/download_models.py --models sam             # ~2.4 GB
python scripts/download_models.py --models toon_crafter    # ~8 GB
```

By default weights are saved to `./models/`. Override with the `MODELS_ROOT` environment variable:

```bash
MODELS_ROOT=/data/models python scripts/download_models.py --models all
```

### Expected directory layout after download

```
models/
  ip_adapter/
    models/image_encoder/          ← CLIP ViT-L/14 weights
    ip-adapter-faceid_sd15.bin
  live_portrait/
    pretrained_weights/
      liveportrait/
        base_models/
          appearance_feature_extractor.pth
          motion_extractor.pth
          warping_module.pth
          spade_generator.pth
        retargeting_models/
          stitching_retargeting_module.pth
  triposr/
    config.yaml
    model.ckpt
  sam/
    sam_vit_h_4b8939.pth
  toon_crafter/
    model.ckpt
    config.yaml
```

---

## Running the Server

```bash
# Default: host=0.0.0.0, port=8000
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1

# Or use the installed entry point
emosvg-server

# Or run directly
python -m src.api.main
```

> **Important:** Always use `--workers 1`. The ModelRegistry is a process-level singleton and is not fork-safe.

Once running, open:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health check**: http://localhost:8000/health

---

## API Quick Start

### Generate a meme animation (minimal)

```bash
curl -X POST http://localhost:8000/generate \
  -F "file=@my_character.png" \
  -F "expression=shock" \
  -F "output_format=gif"
```

### Generate with 3D and SVG

```bash
curl -X POST http://localhost:8000/generate \
  -F "file=@my_character.png" \
  -F "expression=laugh" \
  -F "output_format=gif" \
  -F "run_3d=true" \
  -F "run_svg=true" \
  -F "fps=24" \
  -F "width=512" \
  -F "height=512"
```

### Animation only (fastest)

```bash
curl -X POST http://localhost:8000/animate \
  -F "file=@my_character.png" \
  -F "expression=shock" \
  -F "use_toon_crafter=true" \
  -F "frames_between=4"
```

### Batch processing (up to 8 images)

```bash
curl -X POST http://localhost:8000/batch/generate \
  -F "files=@char1.png" \
  -F "files=@char2.png" \
  -F "expression=shock" \
  -F "fps=24"
```

### List available expressions

```bash
curl http://localhost:8000/expressions
```

### Check model/VRAM status

```bash
curl http://localhost:8000/status
```

---

## Pipeline Walkthrough

### Step 1 — IP Feature Extraction

The source image is encoded by a CLIP ViT-L/14 model into a 768-dim embedding. This embedding is passed downstream to LivePortrait as a character identity conditioning signal.

- **With weights**: uses the IP-Adapter image encoder from `models/ip_adapter/`
- **Without weights**: downloads `openai/clip-vit-large-patch14` from HuggingFace automatically

### Step 2 — Meme Animation

`MotionDesigner` generates a sequence of `SquashParams` objects describing the Squash-and-Stretch motion curve (wind-up → attack → hold → bounce → settle). `LivePortraitWrapper` renders each frame by:

1. Extracting 21 3D keypoints from the source image
2. Adding expression coefficient deltas (63-dim) derived from `SquashParams`
3. Running the warp + SPADE generator to produce the output frame

Optionally, `ToonCrafterWrapper` inserts `frames_between` interpolated frames between each consecutive LivePortrait keyframe for smoother cartoon motion.

- **With weights**: real LivePortrait inference (~4.5 GB VRAM)
- **Without weights**: affine-warp CPU fallback

### Step 3 — 3D Reconstruction (optional, `run_3d=true`)

The peak animation keyframe (or original source if `use_source_for_3d_svg=true`) is passed to TripoSR for single-image 3D reconstruction. The foreground is automatically cropped and recentered to fill 85% of the frame before inference. Output: OBJ + GLB files.

- **With weights**: TripoSR inference (~6 GB VRAM)
- **Without weights**: UV-sphere trimesh fallback

### Step 4 — SVG Vectorization (optional, `run_svg=true`)

The same input image is segmented by SAM into semantic regions. Each region's contour is fitted with Bézier curves (Schneider algorithm) and assembled into a layered SVG file.

- **With weights**: SAM ViT-H segmentation (~2.4 GB VRAM)
- **Without weights**: OpenCV K-means contour fallback

---

## Configuration

Copy and edit the base config:

```bash
cp configs/base.yaml.example configs/base.yaml
```

Key fields:

```yaml
models_root: ./models        # path to downloaded weights
output_root: ./outputs       # path for generated files

device: cuda                 # cuda | cpu
vram_budget_gb: 10.0         # hard cap on GPU memory usage
torch_dtype: float16         # float16 | bfloat16 | float32
```

Environment variables override config values (see [Environment Variables](#environment-variables)).

---

## Running Tests

```bash
# Full test suite (221 tests, no GPU required)
python -m pytest tests/ -q

# Unit tests only
python -m pytest tests/unit/ -q

# Integration tests only
python -m pytest tests/integration/ -q

# Smoke tests
python -m pytest tests/smoke_test.py -v
```

All tests run in fallback mode — no GPU, no model weights required.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODELS_ROOT` | `./models` | Root directory for model weights |
| `OUTPUT_ROOT` | `./outputs` | Root directory for generated files |
| `API_HOST` | `0.0.0.0` | Server bind address |
| `API_PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `DEVICE` | `cuda` | Compute device (`cuda` or `cpu`) |
| `TORCH_DTYPE` | `float16` | Model dtype (`float16`/`bfloat16`/`float32`) |
| `VRAM_BUDGET_GB` | `10.0` | VRAM budget cap in GB |

---

## API Reference

### POST /generate

Full pipeline: IP extraction → animation → optional 3D → optional SVG.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Source character image |
| `expression` | string | `shock` | One of: shock, laugh, cry, rage, smug, surprised, custom |
| `output_format` | string | `gif` | gif, mp4, or webp |
| `fps` | int | `24` | 8–60 |
| `width` | int | `512` | 64–2048 |
| `height` | int | `512` | 64–2048 |
| `run_3d` | bool | `true` | Enable 3D reconstruction |
| `run_svg` | bool | `true` | Enable SVG vectorization |
| `use_source_for_3d_svg` | bool | `false` | Use source image instead of peak keyframe for 3D/SVG |
| `use_toon_crafter` | bool | `false` | Enable ToonCrafter inter-frame smoothing |
| `frames_between` | int | `4` | 1–16 interpolated frames between keyframes |

### POST /animate

Animation only (skips 3D and SVG). Same fields as `/generate` minus `run_3d`, `run_svg`, `use_source_for_3d_svg`.

### POST /reconstruct

3D reconstruction only.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Source image |
| `mc_resolution` | int | `256` | Marching cubes resolution |
| `remove_background` | bool | `true` | Run rembg background removal |
| `foreground_ratio` | float | `0.85` | Fraction of frame the subject should fill |

### POST /vectorize

SVG vectorization only.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Source image |
| `bezier_tolerance` | float | `2.0` | Bézier fitting tolerance (lower = more detail) |
| `min_region_area` | int | `100` | Minimum pixel area for a region to be vectorized |

### POST /batch/generate

Process up to 8 images in one request. Accepts the same fields as `/generate` plus `files` (list of uploads). Per-image failures do not abort the batch.

---

## Troubleshooting

**`CUDA out of memory`**
Lower `VRAM_BUDGET_GB` or disable 3D/SVG with `run_3d=false run_svg=false`. The pipeline evicts LRU models automatically but needs accurate budget settings.

**`liveportrait package not installed`**
Run `python scripts/download_models.py --models git_packages`.

**`LivePortrait weights not found — using affine-warp fallback`**
Run `python scripts/download_models.py --models live_portrait`.

**Server returns 500 on `/generate`**
Check `LOG_LEVEL=DEBUG` output. Common causes: missing weights, VRAM exhaustion, unsupported image format.

**Tests fail with import errors**
Ensure the virtual environment is active and `pip install -e .` has been run.
