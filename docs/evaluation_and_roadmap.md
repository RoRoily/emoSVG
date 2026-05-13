# emoSVG Evaluation And Roadmap

## 1. Project Reading

emoSVG is a single-process multi-modal asset pipeline:

1. source character image
2. CLIP/IP feature extraction
3. squash-and-stretch meme animation with LivePortrait plus optional ToonCrafter
4. optional single-image 3D reconstruction
5. optional semantic layered SVG vectorization

The current architecture is strongest as an integrated controllable asset pipeline, not as a single SOTA model wrapper. Its competitive value should be measured by identity preservation, cartoon expressiveness, asset editability, and resource cost across the full workflow.

## 2. Field Landscape

| Area | Mature baseline | Frontier direction | Relevance to emoSVG |
|---|---|---|---|
| Portrait and character animation | LivePortrait uses implicit keypoints for efficient and controllable portrait animation, with better practical speed/controllability than many diffusion-only approaches. Source: https://liveportrait.github.io/ | Large video/image-to-video models such as LTX-Video, Wan2.1 FLF2V, Wan2.2 I2V/Animate, and similar systems provide stronger motion priors but require much larger inference budgets. Sources: https://huggingface.co/Lightricks/LTX-Video, https://github.com/Wan-Video/Wan2.1, https://github.com/Wan-Video/Wan2.2 | Keep LivePortrait as the fast controllable path. Treat large video models as benchmark upper bounds or optional premium backends, not the default path. |
| Cartoon interpolation | ToonCrafter targets cartoon interpolation where linear/correspondence methods fail under large nonlinear motion and disocclusion. Source: https://huggingface.co/papers/2405.17933 | First-last-frame video generation is becoming the more general version of this idea. Wan2.1 FLF2V is especially aligned with emoSVG's source/peak boundary setup. Source: https://github.com/Wan-Video/Wan2.1 | The driver-mode design is directionally correct, but must be exposed end-to-end and evaluated against interpolation mode. |
| Segmentation | SAM is a solid image segmentation baseline; SAM 2 extends the model to images and videos with streaming memory and stronger/faster image segmentation. Sources: https://about.fb.com/news/2024/07/our-new-ai-model-can-segment-video/, https://huggingface.co/papers/2408.00714 | Video-aware masks can stabilize animated layer tracking and SVG/3D extraction over generated frames. | Move SVG segmentation from static SAM ViT-H to SAM 2 or a promptable mask stack when temporal consistency matters. |
| Single-image 3D | TripoSR is fast and practical; its paper reports feed-forward single-image 3D mesh generation around sub-second scale on high-end GPU. Source: https://huggingface.co/papers/2403.02151 | Stable Fast 3D, TRELLIS, Hunyuan3D 2.x, and similar systems improve textured/PBR asset quality and representation flexibility. Sources: https://stability.ai/news/introducing-stable-fast-3d, https://microsoft.github.io/TRELLIS/, https://github.com/Tencent-Hunyuan/Hunyuan3D-2 | Keep TripoSR for low-latency fallback, add a backend abstraction for higher-quality mesh/PBR paths. |
| Raster-to-SVG | VTracer/Potrace-style tracing remains a practical production baseline for compact SVG. Source: https://github.com/visioncortex/vtracer | Layered Image Vectorization via Semantic Simplification and DiffVG-style optimization point toward semantic, compact, editable SVGs rather than raw contours. Sources: https://szuviz.github.io/layered_vectorization/, https://cseweb.ucsd.edu/~tzli/diffvg/ | Current SAM + Bezier path is a reasonable prototype. The next step is semantic hierarchy plus differentiable/raster-loss refinement. |
| Evaluation | VBench/VBench++ decompose video quality into dimensions such as identity consistency, motion smoothness, flicker, and condition consistency. Sources: https://huggingface.co/papers/2311.17982, https://huggingface.co/papers/2411.13503 | Benchmarks are moving from one scalar score to diagnostic metrics plus human preference alignment. | emoSVG should build a task-specific benchmark rather than depend on generic FVD/FID alone. |

## 3. Current Project Judgment

### Strengths

- Clear modular boundaries and a shared `ModelRegistry` make heavy-model orchestration testable and VRAM-aware.
- CPU fallback paths allow CI to verify contracts without model weights.
- The two ToonCrafter modes are conceptually strong: interpolation for smoothness, driver mode for exaggerated cartoon motion from source/peak boundaries.
- The source-vs-peak-keyframe switch for 3D/SVG is product-relevant because creators may want either canonical assets or expressive assets.

### Risks And Gaps

- FullPipeline and API expose `use_toon_crafter` and `frames_between`, but not `use_toon_crafter_as_driver` or `driver_num_frames`. The architecture's most important animation path is therefore not reachable from `/generate` or `/animate`.
- The ToonCrafter adapter assumes a packaged `tooncrafter.inference.ToonCrafterInference` API. The local run notes suggest ToonCrafter is not cleanly packaged, so the real adapter path needs a real-weight integration test.
- Passing `ip_adapter_embeds` into LivePortrait is an optimistic extension. It needs verification against the actual LivePortrait API; otherwise identity conditioning may be silently ineffective.
- Current tests are mostly fallback/mocked tests. They validate shape, routing, and files, but not real visual quality.
- Documentation and code have drift: SAM VRAM differs between architecture docs and code, IP fallback comments differ, and driver-mode request fields are missing in the full pipeline.
- 3D/SVG secondary input uses sampled animation keyframe metadata. This may not always select the true peak frame, especially after smoothing or driver generation.

## 4. Evaluation Plan

### Dataset

Build a small but diagnostic `eval_assets/` suite:

- 40 flat IP characters: mascot, sticker, anime, emoji-like, non-human, transparent PNG, busy-background PNG.
- 20 face-like characters where ArcFace/InsightFace identity metrics are meaningful.
- 20 synthetic/vector-known assets with source SVGs for SVG reconstruction/editability evaluation.
- 20 Objaverse or in-house 3D assets rendered to 2D with known mesh ground truth for 3D metrics.
- 10 adversarial cases: tiny eyes, no mouth, multiple faces, transparent holes, thin outlines, text/logo regions.

Each sample should have a manifest:

```json
{
  "id": "mascot_001",
  "category": "flat_mascot",
  "source": "path/to/image.png",
  "has_face": false,
  "has_svg_gt": true,
  "has_mesh_gt": false,
  "expected_primary_regions": ["head", "eyes", "mouth", "body"]
}
```

### Automated Metrics

| Output | Metrics |
|---|---|
| Animation | first-frame/final-frame CLIP or DINO similarity, optional ArcFace similarity, foreground mask IoU over time, color drift, temporal LPIPS/flicker, optical-flow smoothness, frame count/duration correctness, out-of-frame and alpha-boundary errors |
| Meme expressiveness | silhouette height/width change, landmark/keypoint displacement where detectable, mouth/eye region deformation, preset separability between shock/laugh/rage/cry/smug |
| 3D | multi-view render similarity to source, CLIP/DINO image alignment, vertex/face counts, watertightness, nondegenerate face ratio, normal consistency, self-intersection warnings, GLB validation, if GT exists: Chamfer distance and F-score |
| SVG | rasterized SVG vs source PSNR/SSIM/LPIPS, number of layers, number of paths, node count, file size, render time, semantic mask IoU, layer naming quality, editability score |
| System | p50/p95 latency, peak VRAM, CPU RAM, model load time, offload count, failure rate, deterministic replay by seed, output artifact size |

### Human Review

Use blind pairwise review for a small weekly panel:

- A/B: LivePortrait only vs ToonCrafter interpolation vs ToonCrafter driver.
- A/B: TripoSR vs candidate high-quality 3D backend.
- A/B: current SVG vs VTracer vs semantic layered vectorization.

Rate on 1-5:

- identity preservation
- meme impact
- temporal coherence
- visual artifacts
- 3D usability
- SVG editability

Aggregate with win rate plus Bradley-Terry/Elo so each iteration has a clear leaderboard.

### Benchmark Harness

Add a script such as `scripts/eval_pipeline.py`:

- input: manifest JSONL, backend config, output root
- output: per-sample JSON, aggregate CSV, HTML gallery
- modes: `animation_only`, `svg_only`, `reconstruct_only`, `full`
- compare baselines: `fallback`, `live_portrait`, `live_portrait_toon_interp`, `live_portrait_toon_driver`

Gate releases with minimum checks:

- no endpoint regression
- no increase in failure rate
- peak VRAM within configured budget
- animation identity score does not regress more than a chosen tolerance
- SVG file/node count does not explode for flat-art inputs

## 5. Iteration Roadmap

### Phase 0: Make The Current Claims Verifiable

- Fix architecture/README drift and encoding issues.
- Expose `use_toon_crafter_as_driver` and `driver_num_frames` through `FullPipelineRequest`, `/animate`, `/generate`, and `/batch`.
- Add real-backend smoke tests behind explicit environment flags, for example `EMOSVG_RUN_REAL_MODELS=1`.
- Add backend capability checks that report missing imports, missing weights, and unsupported adapter APIs separately.
- Replace sampled-keyframe secondary input with an explicit saved peak frame from the animator.

### Phase 1: Build The Evaluation Loop

- Create the eval manifest and a small seed dataset.
- Implement automated animation/SVG/3D/system metrics.
- Generate an HTML comparison gallery per run.
- Store benchmark summaries under `outputs/eval/YYYYMMDD-HHMM/`.
- Define release thresholds for "developer pass", "visual pass", and "demo pass".

### Phase 2: Improve Animation Quality

- Calibrate the `SquashParams -> LivePortrait exp_delta` mapping with measured safe ranges per expression.
- Add mask-aware deformation so non-face mascots and flat stickers do not depend solely on human-face coefficients.
- Compare ToonCrafter driver against ToonCrafter interpolation, LTX-Video, and Wan FLF2V on the same source/peak pairs.
- Add optional reference-motion templates: bounce, shake, pop, recoil, inhale, snapback.
- Track identity drift and temporal flicker as first-class scores.

### Phase 3: Improve SVG And 3D Asset Quality

- Add VTracer as a production baseline for SVG compactness.
- Upgrade or supplement SAM with SAM 2 for better segmentation and possible video mask propagation.
- Add semantic layer grouping: background, body, eyes, mouth, accessories, outlines, highlights.
- Add DiffVG-style or raster-loss refinement after Bezier fitting.
- Add a 3D backend abstraction with TripoSR as fast mode and Hunyuan3D/TRELLIS/Stable Fast 3D as quality modes.
- Add mesh cleanup targets for GLB production: scale normalization, material naming, texture atlas, manifold checks.

### Phase 4: Productize The Workflow

- Add async job management instead of long blocking requests.
- Cache IP features and model outputs by source hash, expression, backend, resolution, and seed.
- Add a lightweight review UI for galleries, scores, and manual accept/reject.
- Emit a `manifest.json` for every generation containing inputs, seed, model versions, metrics, output paths, and warnings.

## 6. Priority Recommendation

The next best sprint is:

1. expose ToonCrafter driver mode end-to-end;
2. add the eval harness and gallery;
3. run a 30-image animation-only benchmark comparing LivePortrait, ToonCrafter interpolation, and ToonCrafter driver;
4. only after those scores exist, decide whether to replace or supplement ToonCrafter with newer I2V/FLF2V models.

This keeps the project grounded: the roadmap stays ambitious, but every model change must beat a measured baseline on the actual IP-meme workflow.
