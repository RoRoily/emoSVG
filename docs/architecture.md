# emoSVG — Architecture Reference

This document describes the internal design of the emoSVG pipeline: module boundaries, data flow, VRAM management, fallback strategy, and key design decisions.

---

## Table of Contents

- [High-Level Overview](#high-level-overview)
- [Directory Structure](#directory-structure)
- [Core Layer](#core-layer)
  - [ModelRegistry](#modelregistry)
  - [DeviceManager](#devicemanager)
  - [Exceptions](#exceptions)
- [Module Layer](#module-layer)
  - [IPExtractor](#ipextractor)
  - [MemeAnimator](#memeanimator)
  - [Reconstructor3D](#reconstructor3d)
  - [SVGVectorizer](#svgvectorizer)
- [Pipeline Layer](#pipeline-layer)
  - [FullPipeline](#fullpipeline)
  - [Partial Pipelines](#partial-pipelines)
- [API Layer](#api-layer)
- [Data Flow](#data-flow)
- [VRAM Management](#vram-management)
- [Fallback Strategy](#fallback-strategy)
- [Key Design Decisions](#key-design-decisions)

---

## High-Level Overview

```
Source Image (PNG)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                      FullPipeline                       │
│                                                         │
│  Step 1        Step 2          Step 3       Step 4      │
│  IPExtractor → MemeAnimator → Reconstructor3D → SVGVect │
│  (CLIP)        (LivePortrait   (TripoSR)        (SAM +  │
│                + ToonCrafter)                   Bézier) │
└─────────────────────────────────────────────────────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
  IPFeatures    GIF/MP4/WebP      OBJ + GLB       .svg
  (numpy)       animation         mesh            vector
```

The pipeline is **single-process** by design. All four modules share one `ModelRegistry` singleton, which enforces a global VRAM budget and LRU eviction policy. This avoids the complexity of inter-process communication while keeping GPU memory usage predictable.

---

## Directory Structure

```
emoSVG/
├── src/
│   ├── core/                    # Shared infrastructure
│   │   ├── model_registry.py    # Singleton model lifecycle manager
│   │   ├── device_manager.py    # VRAM queries and device moves
│   │   ├── config_loader.py     # YAML config loader
│   │   ├── exceptions.py        # Domain exception hierarchy
│   │   └── __init__.py          # Re-exports: ModelRegistry, load_config
│   │
│   ├── modules/
│   │   ├── ip_extractor/        # CLIP image encoding
│   │   ├── meme_animator/       # LivePortrait + ToonCrafter + MotionDesigner
│   │   ├── reconstructor_3d/    # TripoSR + mesh post-processing
│   │   └── svg_vectorizer/      # SAM + Bézier fitting + SVG assembly
│   │
│   ├── pipeline/
│   │   ├── full_pipeline.py     # Orchestrates all 4 modules
│   │   ├── partial_pipelines.py # Single-module convenience wrappers
│   │   └── base_pipeline.py     # Abstract base class
│   │
│   └── api/
│       ├── main.py              # FastAPI app factory
│       ├── dependencies.py      # Dependency injection (get_pipeline)
│       ├── middleware.py        # Request logging
│       └── routers/             # One file per endpoint group
│
├── tests/
│   ├── unit/                    # Per-module unit tests
│   ├── integration/             # Full pipeline + API endpoint tests
│   └── smoke_test.py            # End-to-end smoke tests
│
├── scripts/
│   └── download_models.py       # Weight download utility
│
└── configs/                     # YAML configuration files
```

---

## Core Layer

### ModelRegistry

`src/core/model_registry.py`

The central authority for all model lifecycle operations. It is a **process-level singleton** (double-checked locking) that tracks every registered model as a `ModelEntry`.

#### States

```
UNLOADED  →  ON_CPU  →  ON_GPU
                ↑           │
                └───────────┘  (offload_to_cpu)
```

| State | Meaning |
|---|---|
| `UNLOADED` | Loader has not been called; no memory used |
| `ON_CPU` | Weights in RAM; loader has been called once |
| `ON_GPU` | Weights on CUDA device; ready for inference |

#### Key methods

```python
registry.register(model_id, loader_fn, estimated_vram_gb)
# Declares a model. Lazy — does not call loader_fn yet.

registry.load_to_gpu(model_id) -> nn.Module
# Ensures the model is ON_GPU. Triggers LRU eviction if needed.

registry.offload_to_cpu(model_id)
# Moves the model back to CPU and frees CUDA memory.

with registry.model_context(model_id, offload_after=True) as model:
    output = model(input)
# Context manager: load → yield → offload.
```

#### LRU eviction

When `load_to_gpu()` is called and the VRAM budget would be exceeded, the registry evicts GPU-resident models in least-recently-used order until enough headroom exists. Each model's `last_used_ts` is updated on every `load_to_gpu()` call.

#### Thread safety

All mutations to `_entries` are protected by a single `threading.Lock`. The singleton creation uses a separate `_init_lock` with double-checked locking.

---

### DeviceManager

`src/core/device_manager.py`

Owned by `ModelRegistry`. Modules never instantiate it directly — they access it via `registry.device_manager`.

Responsibilities:
- Query VRAM state (`snapshot()`, `can_fit()`)
- Move modules between devices (`move_to_gpu()`, `move_to_offload()`)
- Enforce a consistent dtype policy (float16 on GPU, float32 on CPU)
- Handle tuple bundles (processor, model) for CLIP-style two-part models

```python
@dataclass
class DeviceConfig:
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    vram_budget_gb: float = 10.0
    safety_margin_gb: float = 0.5   # always kept free
```

The effective budget is `vram_budget_gb - safety_margin_gb`. The safety margin prevents OOM from allocator fragmentation.

---

### Exceptions

`src/core/exceptions.py`

Domain-specific exception hierarchy:

```
EmoSVGError (base)
├── IPExtractionError
├── AnimationError
├── ReconstructionError
└── VectorizationError
```

All module-level errors are wrapped in the appropriate domain exception before propagating. The API layer catches `EmoSVGError` and returns HTTP 422; unexpected exceptions return HTTP 500.

---

## Module Layer

Each module follows the same pattern:

1. `__init__` checks whether real weights are available → sets `_use_fallback`
2. If not fallback, calls `_register_model()` to declare the model with `ModelRegistry`
3. Public methods use `registry.model_context()` for inference
4. A CPU-only fallback path is always available

### IPExtractor

`src/modules/ip_extractor/`

Extracts character identity features from the source image.

**Primary backend**: IP-Adapter image encoder (CLIP ViT-L/14 loaded from `models/ip_adapter/image_encoder/`)

**Fallback**: `openai/clip-vit-large-patch14` downloaded automatically from HuggingFace

**Output**: `IPFeatures`
```python
@dataclass
class IPFeatures:
    image_embeds: np.ndarray    # (1, 768) CLIP embedding
    face_embeds: np.ndarray | None  # (1, 512) InsightFace embedding, if available
    preprocessed: np.ndarray    # normalised image array
    backend_used: str           # "ip_adapter" | "clip_fallback"
```

The `image_embeds` array is passed downstream to `MemeAnimator` as `ip_image_embeds`. LivePortrait uses it as a soft conditioning signal via `ip_adapter_embeds` kwarg to `execute_portraits()`.

**VRAM**: ~2.5 GB

**Sub-components**:
- `ImagePreprocessor` — resizes and normalises to CLIP input spec
- `IPExtractor._encode_face()` — optional InsightFace identity embedding (gracefully skipped if `insightface` not installed)

---

### MemeAnimator

`src/modules/meme_animator/`

The core deliverable module. Produces a Squash-and-Stretch meme animation.

**Sub-components**:

#### MotionDesigner

Pure math, no model. Generates a list of `SquashParams` objects describing the motion curve for a given `MemeExpression`.

Motion phases:
```
wind-up (2 frames) → attack (3 frames) → hold (N) → bounce (M) → settle (3 frames)
```

Wind-up is a counter-pose (slightly opposite to the peak) for cartoon anticipation. Each phase interpolates linearly between neutral and peak `SquashParams`.

Presets are defined in `_PRESETS: dict[MemeExpression, SquashParams]` in `motion_designer.py`.

#### SquashParams

Pydantic model with 11 validated fields:

| Field | Range | Effect |
|---|---|---|
| `eye_bulge_scale` | 0.1–5.0 | Eye size multiplier |
| `eye_squint_scale` | 0.1–2.0 | Eye vertical compression |
| `brow_raise_offset` | -1.0–1.0 | Brow vertical offset |
| `jaw_drop_scale` | 0.5–3.0 | Jaw opening multiplier |
| `mouth_width_scale` | 0.5–2.5 | Mouth width multiplier |
| `mouth_corner_offset` | -1.0–1.0 | Smile (+) / frown (-) |
| `head_squash_scale` | 0.3–1.5 | Vertical head compression |
| `head_stretch_scale` | 0.5–2.0 | Vertical head stretch |
| `head_tilt_deg` | -30–30 | Head roll in degrees |
| `hold_frames` | 1–30 | Frames at peak expression |
| `bounce_frames` | 0–10 | Overshoot bounce frames |

#### LivePortraitWrapper

Translates `SquashParams` into LivePortrait's 63-dim expression coefficient space and renders frames.

**Expression coefficient mapping** (63 dims = 21 keypoints × 3 coordinates):

```
dims  0- 5   left-eye  (z → bulge, y → squint)
dims  6-11   right-eye
dims 12-17   brow      (y → raise/lower)
dims 18-23   nose
dims 24-29   upper-lip
dims 30-35   lower-lip
dims 36-41   jaw/chin  (y → drop)
dims 42-47   mouth-corner (x → width, y → smile/frown)
dims 48-62   remaining face keypoints
```

**Real inference path** (when weights present):
```
source_bgr
    │
    ▼
_preprocess_source()     → (1, 3, H, W) float tensor [-1, 1]
    │
    ▼
pipeline.get_kp_info()   → kp_source dict (pitch, yaw, roll, t, exp, scale, kp)
    │
    ▼
_build_driving_info()    → x_d_info dict (kp_source + SquashParams deltas)
    │
    ▼
pipeline.execute_portraits(img_rgb, x_s_info, x_d_info, ip_adapter_embeds=...)
    │
    ▼
_tensor_to_bgr()         → HxWx3 uint8 BGR frame
```

`render_sequence()` extracts source keypoints once and reuses them for all frames in the batch, avoiding redundant `get_kp_info()` calls.

**MEME_MOTION_TEMPLATES**: Five hardcoded extreme expression templates (`SHOCK_EXTREME`, `LAUGH_EXTREME`, `RAGE_EXTREME`, `CRY_EXTREME`, `SMUG_EXTREME`) defined as 63-dim `exp_delta` arrays. These are the peak targets that `MotionDesigner` interpolates toward.

**VRAM**: ~4.5 GB

**Fallback**: affine-warp using OpenCV (scale + rotate + eye-region zoom)

#### ToonCrafterWrapper

Optional inter-frame smoother. After LivePortrait renders N keyframes, ToonCrafter inserts `frames_between` interpolated frames between each consecutive pair.

```
LivePortrait keyframes:  [F0, F1, F2, F3, ...]
After ToonCrafter (n=3): [F0, i1, i2, i3, F1, i4, i5, i6, F2, ...]
```

`smooth_sequence()` calls `interpolate(start, end, n)` for each consecutive pair and concatenates the results, avoiding duplicate boundary frames.

**VRAM**: ~8 GB

**Fallback**: linear pixel blending between frames

---

### Reconstructor3D

`src/modules/reconstructor_3d/`

Single-image 3D reconstruction via TripoSR.

**Pipeline**:
```
source image
    │
    ▼
_load_image()
  ├── optional rembg background removal
  └── _recenter_foreground()   ← crops tight bbox, pads to square
    │                             so subject fills foreground_ratio (0.85)
    ▼
TripoSR inference
  └── scene_codes → extract_mesh(mc_resolution)
    │
    ▼
MeshProcessor.process()
  ├── remove degenerate faces (trimesh 4.x: nondegenerate_faces() mask)
  ├── merge duplicate vertices
  └── compute vertex normals
    │
    ▼
MeshExporter.export()
  └── OBJ + GLB files
```

**Foreground recentering**: TripoSR performs best when the subject fills ~85% of the input frame. `_recenter_foreground()` detects the alpha-channel bounding box, computes the minimum enclosing square, and pads to a canvas where `side / foreground_ratio = canvas_side`.

**VRAM**: ~6 GB

**Fallback**: trimesh UV-sphere (32×32 subdivisions)

---

### SVGVectorizer

`src/modules/svg_vectorizer/`

Converts a raster image to a layered SVG.

**Pipeline**:
```
source image
    │
    ▼
Segmentor.segment()
  ├── SAM: automatic mask generation (all regions)
  └── fallback: OpenCV K-means colour clustering + contour extraction
    │
    ▼
BezierFitter.mask_to_paths()   (per mask)
  └── Schneider algorithm: contour points → cubic Bézier curves
    │
    ▼
SVGBuilder.build()
  └── one <g> layer per mask, filled with mean region colour
```

**Output**: layered SVG where each semantic region is a separate `<g>` element with a unique `id`. Layer naming follows the `layer_naming` strategy (colour-based by default).

**VRAM**: ~2.4 GB (SAM ViT-H)

**Fallback**: OpenCV K-means (k=8 colours) + `findContours` → polyline paths

---

## Pipeline Layer

### FullPipeline

`src/pipeline/full_pipeline.py`

Orchestrates all four modules in sequence. Owns the module instances and passes the shared `ModelRegistry` to each.

```python
class FullPipelineRequest:
    source_image_path: Path
    expression: MemeExpression
    output_format: str          # gif | mp4 | webp
    fps: int
    resolution: tuple[int, int]
    run_3d: bool
    run_svg: bool
    use_source_for_3d_svg: bool  # False = use peak keyframe; True = use source
    use_toon_crafter: bool
    frames_between: int
```

**Secondary input resolution**: Steps 3 and 4 share a single secondary input path, computed once:
- `use_source_for_3d_svg=False` (default): the peak animation keyframe (middle of the keyframe list) is saved to `outputs/_tmp_keyframes/` and used as input for both TripoSR and SAM
- `use_source_for_3d_svg=True`: the original source image is used directly

This avoids writing the keyframe to disk twice when both `run_3d` and `run_svg` are enabled.

### Partial Pipelines

`src/pipeline/partial_pipelines.py`

Convenience wrappers for single-module use via the API:

| Class | Endpoint | Module |
|---|---|---|
| `AnimatePipeline` | `POST /animate` | `MemeAnimator` |
| `ReconstructPipeline` | `POST /reconstruct` | `Reconstructor3D` |
| `VectorizePipeline` | `POST /vectorize` | `SVGVectorizer` |

Each has a `from_config()` classmethod that delegates to the underlying module's `from_config()`.

---

## API Layer

`src/api/`

FastAPI application with five router groups:

| Router | Prefix | Description |
|---|---|---|
| `generate` | `/generate` | Full pipeline |
| `animate` | `/animate` | Animation only |
| `reconstruct` | `/reconstruct` | 3D only |
| `vectorize` | `/vectorize` | SVG only |
| `batch` | `/batch` | Multi-image batch |

Plus two utility endpoints: `GET /health`, `GET /expressions`, `GET /status`.

**Dependency injection**: `get_pipeline()` in `dependencies.py` returns the `FullPipeline` singleton. Tests override this via `app.dependency_overrides`.

**Input validation**: All numeric parameters are validated in the router handler before the pipeline is invoked (width 64–2048, height 64–2048, fps 8–60, frames_between 1–16, output_format in {gif, mp4, webp}).

**Batch endpoint**: Processes images sequentially (single-process constraint). Per-image failures are caught and recorded in the result; they do not abort the remaining images. Hard cap: 8 images per request (`_MAX_BATCH = 8`).

**Workers**: Always `workers=1`. The `ModelRegistry` singleton is not fork-safe.

---

## Data Flow

```
POST /generate
    │
    ├── save upload → tmp file
    │
    ├── FullPipeline.execute(FullPipelineRequest)
    │       │
    │       ├── IPExtractor.extract()
    │       │       └── CLIP encode → IPFeatures(image_embeds)
    │       │
    │       ├── MemeAnimator.generate()
    │       │       ├── MotionDesigner.design(expression) → [SquashParams]
    │       │       ├── LivePortraitWrapper.render_sequence(
    │       │       │       source_bgr, param_sequence, ip_image_embeds)
    │       │       │       └── [rendered frames]
    │       │       ├── ToonCrafterWrapper.smooth_sequence() [optional]
    │       │       └── FrameComposer.compose() → GIF/MP4/WebP file
    │       │
    │       ├── Reconstructor3D.reconstruct() [if run_3d]
    │       │       ├── _recenter_foreground()
    │       │       ├── TripoSR inference
    │       │       ├── MeshProcessor.process()
    │       │       └── MeshExporter.export() → OBJ + GLB
    │       │
    │       └── SVGVectorizer.vectorize() [if run_svg]
    │               ├── Segmentor.segment() → masks
    │               ├── BezierFitter.mask_to_paths() per mask
    │               └── SVGBuilder.build() → .svg file
    │
    ├── delete tmp file
    │
    └── GenerateResponse(animation_path, frame_count, ...)
```

---

## VRAM Management

The pipeline's total theoretical VRAM requirement is ~15.4 GB (all models loaded simultaneously). In practice, models are loaded one at a time and evicted between steps.

| Model | VRAM estimate |
|---|---|
| CLIP (IPExtractor) | 2.5 GB |
| LivePortrait | 4.5 GB |
| ToonCrafter | 8.0 GB |
| TripoSR | 6.0 GB |
| SAM ViT-H | 2.4 GB |

**Eviction sequence** for a full pipeline run on a 10 GB GPU:

1. CLIP loads (2.5 GB) → extracts features → offloads
2. LivePortrait loads (4.5 GB) → renders all frames → offloads
3. ToonCrafter loads (8.0 GB) → smooths sequence → offloads
4. TripoSR loads (6.0 GB) → reconstructs mesh → offloads
5. SAM loads (2.4 GB) → segments image → offloads

Each step fits within a 10 GB budget because only one model is on GPU at a time. The `offload_after=True` default in `model_context()` ensures automatic cleanup.

If `use_toon_crafter=False` (default), ToonCrafter is never loaded, reducing peak VRAM to 6 GB.

---

## Fallback Strategy

Every module detects at construction time whether its weights are available and sets `_use_fallback: bool`. The fallback path is always a CPU-only implementation that produces structurally correct (but visually simplified) output.

| Module | Real backend | Fallback |
|---|---|---|
| IPExtractor | IP-Adapter CLIP ViT-L/14 | HF CLIP ViT-L/14 (auto-download) |
| LivePortraitWrapper | LivePortrait pipeline | OpenCV affine warp |
| ToonCrafterWrapper | ToonCrafter inference | Linear pixel blend |
| Reconstructor3D | TripoSR | trimesh UV-sphere |
| SVGVectorizer | SAM ViT-H | OpenCV K-means contours |

This design means the full test suite (221 tests) runs without any GPU or downloaded weights.

---

## Key Design Decisions

**Single process, shared registry**
Multi-process workers would require each process to load its own model copies, multiplying VRAM usage. Single-process with `workers=1` allows all modules to share one registry and one VRAM budget.

**Bundle registration for LivePortrait**
LivePortrait has 5 sub-networks (AppearanceExtractor, MotionExtractor, WarpingNetwork, SPADEGenerator, StitchingRetargeting) with complex inter-network tensor flows. Registering the entire `LivePortraitPipeline` object as a single bundle avoids manual intermediate tensor management and matches the official API design.

**Source keypoints extracted once per sequence**
`render_sequence()` calls `get_kp_info()` once on the source image and reuses the result for all frames. This avoids N redundant forward passes through the MotionExtractor.

**Peak keyframe as 3D/SVG input**
By default, TripoSR and SAM receive the peak animation keyframe (the most exaggerated pose) rather than the original source image. This produces 3D meshes and SVG vectors that match the animated character's expression. Set `use_source_for_3d_svg=True` to use the canonical source image instead.

**Secondary input computed once**
When both `run_3d=True` and `run_svg=True`, the peak keyframe is saved to disk exactly once and reused for both steps. Earlier versions called `_resolve_secondary_input()` twice, writing the file twice.

**trimesh 4.x API**
trimesh 4.x removed `remove_degenerate_faces()`. The mesh processor uses the `nondegenerate_faces()` boolean mask instead: `mesh.update_faces(mesh.nondegenerate_faces())`.

**DeviceManager handles tuple bundles**
CLIP is loaded as a `(processor, model)` tuple. `move_to_gpu()` and `move_to_offload()` detect tuples and move each `nn.Module` element individually, returning the container type unchanged.

**TYPE_CHECKING guard for torch**
`live_portrait.py` uses `torch.Tensor` in type annotations but imports `torch` lazily inside methods (to avoid import overhead when running in fallback mode). A `TYPE_CHECKING` guard makes the annotations available to type checkers without a runtime import.
