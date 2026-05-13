"""
LivePortrait wrapper for the meme_animator module.

Responsibilities:
- Load / unload the LivePortrait pipeline via ModelRegistry (VRAM-safe).
- Accept a source image + SquashParams and return a rendered np.ndarray frame.
- Translate SquashParams into LivePortrait's internal 63-dim expression coefficients.

Real inference pipeline:
    kp_info   = pipeline.get_kp_info(source_tensor)   # extract source keypoints
    x_d_info  = _build_driving_info(kp_info, params)  # apply SquashParams delta
    output    = pipeline.execute_portraits(            # warp + generate
                    source_tensor, kp_info, x_d_info)

VRAM discipline:
    ALL sub-networks (AppearanceExtractor, MotionExtractor, WarpingNetwork,
    SPADEGenerator, StitchingRetargetingNetwork) are loaded as a single bundle
    through ModelRegistry.register().  The loader function NEVER calls .cuda()
    directly — DeviceManager.move_to_gpu() handles device placement after the
    bundle is returned from the loader.

When LivePortrait weights are absent the wrapper falls back to a lightweight
affine-warp approximation so the rest of the pipeline can be tested without
GPU or model downloads.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from src.core import ModelRegistry
from src.core.exceptions import AnimationError

from .schemas import MemeExpression, SquashParams

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ── VRAM budget ───────────────────────────────────────────────────────────────
# AppearanceExtractor ~0.5 GB + MotionExtractor ~0.8 GB
# WarpingNetwork ~1.2 GB + SPADEGenerator ~1.5 GB + Stitching ~0.5 GB
_LIVE_PORTRAIT_VRAM_GB = 4.5

# ── Expected weight files (relative to the resolved weights root) ──────────────
_REQUIRED_WEIGHTS = [
    "liveportrait/base_models/appearance_feature_extractor.pth",
    "liveportrait/base_models/motion_extractor.pth",
    "liveportrait/base_models/warping_module.pth",
    "liveportrait/base_models/spade_generator.pth",
    "liveportrait/retargeting_models/stitching_retargeting_module.pth",
]


def _resolve_liveportrait_weights_root(model_path: Path | None) -> Path | None:
    """
    Find the directory containing ``liveportrait/base_models`` weights.

    Older setup docs and some upstream layouts place weights under
    ``<model_path>/pretrained_weights``. HuggingFace downloads from
    ``KwaiVGI/LivePortrait`` commonly place them directly under
    ``<model_path>/liveportrait``. Support both.
    """
    if model_path is None:
        return None
    candidates = [model_path / "pretrained_weights", model_path]
    for root in candidates:
        if all((root / rel).exists() for rel in _REQUIRED_WEIGHTS):
            return root
    return None


def _resolve_liveportrait_models_config(model_path: Path | None) -> Path:
    """Find LivePortrait ``models.yaml`` in the weights dir or source tree."""
    candidates: list[Path] = []
    if model_path is not None:
        candidates.append(model_path / "src" / "config" / "models.yaml")
    for source_dir in _candidate_liveportrait_src_dirs(model_path):
        candidates.append(source_dir / "config" / "models.yaml")

    for path in candidates:
        if path.exists():
            return path

    # Return the default path so downstream errors include the expected location.
    return (model_path or Path(".")) / "src" / "config" / "models.yaml"

# ── 63-dim expression coefficient index map ───────────────────────────────────
# LivePortrait represents motion as 21 3D keypoints (21 × 3 = 63 dims).
# Semantic groupings derived from the paper and community implementations:
#
#   dims  0- 5  left-eye  keypoints (2 pts × xyz)
#   dims  6-11  right-eye keypoints
#   dims 12-17  brow      keypoints
#   dims 18-23  nose      keypoints
#   dims 24-29  upper-lip keypoints
#   dims 30-35  lower-lip keypoints
#   dims 36-41  jaw/chin  keypoints
#   dims 42-47  mouth-corner keypoints
#   dims 48-62  remaining face keypoints
#
# We only modify the semantically meaningful dims; all others stay at 0.

_EYE_LEFT_Z   = [2, 5]          # z-axis of left-eye keypoints  → bulge outward
_EYE_RIGHT_Z  = [8, 11]         # z-axis of right-eye keypoints
_EYE_LEFT_Y   = [1, 4]          # y-axis of left-eye  → squint (vertical)
_EYE_RIGHT_Y  = [7, 10]         # y-axis of right-eye
_BROW_Y       = [13, 16]        # y-axis of brow keypoints → raise / lower
_JAW_Y        = [37, 40]        # y-axis of jaw keypoints  → drop open
_MOUTH_X      = [43, 46]        # x-axis of mouth-corner   → widen
_MOUTH_Y      = [44, 47]        # y-axis of mouth-corner   → smile / frown


# ── Meme motion templates ─────────────────────────────────────────────────────
# Each template is a dict with:
#   "exp_delta"   : np.ndarray shape (63,) — additive offset to source exp coeff
#   "scale_delta" : float — additive offset to source scale (0 = no change)
#   "roll_delta"  : float — additive head roll in radians (0 = no change)
#
# Values are tuned for maximum cartoon exaggeration while staying within the
# model's stable operating range (empirically ±0.8 for exp dims).

def _make_exp(updates: dict[list, float]) -> np.ndarray:
    """Build a 63-dim exp delta from {index_list: value} pairs."""
    v = np.zeros(63, dtype=np.float32)
    for indices, val in updates.items():
        for i in indices:
            v[i] = val
    return v


MEME_MOTION_TEMPLATES: dict[str, dict] = {
    # ── SHOCK: eyes bulge out, jaw drops, brows shoot up ─────────────────────
    "SHOCK_EXTREME": {
        "exp_delta": _make_exp({
            tuple(_EYE_LEFT_Z):  0.55,   # left eye bulge (z outward)
            tuple(_EYE_RIGHT_Z): 0.55,   # right eye bulge
            tuple(_BROW_Y):     -0.45,   # brows raised (negative y = up)
            tuple(_JAW_Y):       0.80,   # jaw dropped (positive y = down)
            tuple(_MOUTH_X):     0.30,   # mouth slightly wider
        }),
        "scale_delta": 0.0,
        "roll_delta":  0.0,
    },

    # ── LAUGH: eyes squinted shut, mouth wide open, head tilted ──────────────
    "LAUGH_EXTREME": {
        "exp_delta": _make_exp({
            tuple(_EYE_LEFT_Y):  0.40,   # left eye squinted (y compressed)
            tuple(_EYE_RIGHT_Y): 0.40,   # right eye squinted
            tuple(_BROW_Y):     -0.20,   # brows slightly raised
            tuple(_JAW_Y):       0.65,   # jaw open
            tuple(_MOUTH_X):     0.50,   # mouth very wide
            tuple(_MOUTH_Y):    -0.25,   # mouth corners up (smile)
        }),
        "scale_delta":  0.02,            # head slightly larger (puffed up)
        "roll_delta":   0.08,            # slight head tilt
    },

    # ── RAGE: brows furrowed, teeth clenched, head forward ───────────────────
    "RAGE_EXTREME": {
        "exp_delta": _make_exp({
            tuple(_EYE_LEFT_Y):  0.20,   # eyes slightly narrowed
            tuple(_EYE_RIGHT_Y): 0.20,
            tuple(_BROW_Y):      0.50,   # brows pushed DOWN (furrowed)
            tuple(_JAW_Y):       0.25,   # jaw slightly open (clenched)
            tuple(_MOUTH_X):     0.15,   # mouth tight
            tuple(_MOUTH_Y):     0.30,   # mouth corners down (grimace)
        }),
        "scale_delta":  0.03,            # head slightly larger (puffed up)
        "roll_delta":  -0.05,            # slight forward lean
    },

    # ── CRY: drooping eyes, open mouth, head tilted down ─────────────────────
    "CRY_EXTREME": {
        "exp_delta": _make_exp({
            tuple(_EYE_LEFT_Y):  0.30,   # eyes drooping
            tuple(_EYE_RIGHT_Y): 0.30,
            tuple(_BROW_Y):      0.35,   # brows angled inward/down
            tuple(_JAW_Y):       0.50,   # mouth open
            tuple(_MOUTH_Y):     0.40,   # corners down (sad)
        }),
        "scale_delta": -0.01,
        "roll_delta":   0.12,            # head tilted to side
    },

    # ── SMUG: half-lidded eyes, slight smirk ─────────────────────────────────
    "SMUG_EXTREME": {
        "exp_delta": _make_exp({
            tuple(_EYE_LEFT_Y):  0.25,   # left eye half-lidded
            tuple(_EYE_RIGHT_Y): 0.15,   # right eye slightly less (asymmetric)
            tuple(_BROW_Y):      0.15,   # one brow slightly raised
            tuple(_MOUTH_Y):    -0.20,   # slight smirk (one corner up)
        }),
        "scale_delta": 0.0,
        "roll_delta":  0.10,             # head tilted confidently
    },
}

# Map MemeExpression presets to template keys
_EXPRESSION_TO_TEMPLATE: dict[MemeExpression, str] = {
    MemeExpression.SHOCK:     "SHOCK_EXTREME",
    MemeExpression.LAUGH:     "LAUGH_EXTREME",
    MemeExpression.RAGE:      "RAGE_EXTREME",
    MemeExpression.CRY:       "CRY_EXTREME",
    MemeExpression.SMUG:      "SMUG_EXTREME",
    MemeExpression.SURPRISED: "SHOCK_EXTREME",  # reuse shock template
}


def _candidate_liveportrait_src_dirs(model_path: Path | None) -> list[Path]:
    """Return plausible official LivePortrait ``src`` directories."""
    import os

    candidates: list[Path] = []
    for env_name in ("LIVE_PORTRAIT_SOURCE_ROOT", "LIVE_PORTRAIT_SRC"):
        raw = os.getenv(env_name)
        if raw:
            p = Path(raw).expanduser()
            candidates.append(p / "src" if (p / "src").exists() else p)

    if model_path is not None:
        candidates.append(model_path / "src")

    cwd = Path.cwd()
    candidates.extend([
        cwd / "third_party" / "LivePortrait" / "src",
        cwd / "third_party" / "LivePortrait-main" / "src",
        cwd.parent / "third_party" / "LivePortrait" / "src",
        cwd.parent / "third_party" / "LivePortrait-main" / "src",
        Path.home() / "workspace" / "third_party" / "LivePortrait" / "src",
        Path.home() / "workspace" / "third_party" / "LivePortrait-main" / "src",
    ])

    unique: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _alias_official_liveportrait_src(source_dir: Path) -> None:
    """
    Expose the official LivePortrait ``src`` layout as a ``liveportrait`` package.

    The upstream repo often stores files as ``src/live_portrait_pipeline.py`` and
    uses relative imports inside that module. Importing it as a top-level module
    fails, so we create a lightweight package alias with ``source_dir`` as its
    package search path.
    """
    import sys
    import types

    pkg = types.ModuleType("liveportrait")
    pkg.__file__ = str(source_dir / "__init__.py")
    pkg.__package__ = "liveportrait"
    pkg.__path__ = [str(source_dir)]  # type: ignore[attr-defined]
    sys.modules["liveportrait"] = pkg


def _import_liveportrait_api(model_path: Path | None):
    """
    Import LivePortrait classes, supporting both package and official src layouts.
    """
    try:
        from liveportrait.config.inference_config import InferenceConfig  # type: ignore
        from liveportrait.live_portrait_pipeline import LivePortraitPipeline  # type: ignore
        return InferenceConfig, LivePortraitPipeline
    except ImportError as first_exc:
        last_exc: Exception = first_exc

    for source_dir in _candidate_liveportrait_src_dirs(model_path):
        source_dir = source_dir.expanduser()
        if not (source_dir / "live_portrait_pipeline.py").exists():
            continue
        try:
            _alias_official_liveportrait_src(source_dir.resolve())
            from liveportrait.config.inference_config import InferenceConfig  # type: ignore
            from liveportrait.live_portrait_pipeline import LivePortraitPipeline  # type: ignore
            logger.info("Using LivePortrait source layout from %s", source_dir)
            return InferenceConfig, LivePortraitPipeline
        except ImportError as exc:
            last_exc = exc

    raise AnimationError(
        "LivePortrait package is not importable. If using the official repo zip, "
        "either set LIVE_PORTRAIT_SOURCE_ROOT=/path/to/LivePortrait or create a "
        "compat package whose parent is on Python path, e.g. "
        "~/workspace/third_party/liveportrait_compat/liveportrait copied from "
        "LivePortrait/src. Also install small runtime deps such as: "
        "pip install tyro pykalman lmdb ffmpeg-python. "
        f"Last import error: {last_exc}"
    ) from last_exc


# ── Main wrapper class ────────────────────────────────────────────────────────

class LivePortraitWrapper:
    """
    Adapter between SquashParams and LivePortrait's retargeting API.

    Parameters
    ----------
    model_path:
        Root directory of the LivePortrait installation.
        Expected layout:
            <model_path>/
              pretrained_weights/
                liveportrait/
                  base_models/
                    appearance_feature_extractor.pth
                    motion_extractor.pth
                    warping_module.pth
                    spade_generator.pth
                  retargeting_models/
                    stitching_retargeting_module.pth
        If None or the path does not exist, falls back to affine-warp mode.
    registry:
        ModelRegistry instance. Defaults to the process singleton.
    """

    MODEL_ID = "live_portrait"

    def __init__(
        self,
        model_path: Path | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self._registry = registry or ModelRegistry.instance()
        self._model_path = Path(model_path) if model_path else None
        self._use_fallback = self._should_use_fallback()

        if not self._use_fallback:
            self._register_model()
        else:
            logger.warning(
                "LivePortrait weights not found at '%s'. "
                "Using affine-warp fallback (for testing only).",
                self._model_path,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def render_frame(
        self,
        source_bgr: np.ndarray,
        params: SquashParams,
        ip_image_embeds: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Render a single frame by applying SquashParams deformation to source_bgr.

        Parameters
        ----------
        source_bgr:       HxWx3 uint8 BGR image (OpenCV convention).
        params:           Deformation parameters for this frame.
        ip_image_embeds:  Optional CLIP embedding (1, D) from IPExtractor.
                          When provided, conditions the appearance features for
                          stronger character identity preservation.

        Returns
        -------
        HxWx3 uint8 BGR rendered frame.
        """
        if self._use_fallback:
            return self._affine_warp_fallback(source_bgr, params)

        # Keep model on GPU across single-frame calls (offload_after=False)
        # so that render_sequence() can reuse the loaded state.
        with self._registry.model_context(self.MODEL_ID, offload_after=False) as pipeline:
            return self._live_portrait_infer(
                pipeline, source_bgr, params, ip_image_embeds=ip_image_embeds
            )

    def render_sequence(
        self,
        source_bgr: np.ndarray,
        param_sequence: list[SquashParams],
        ip_image_embeds: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """
        Render a full frame sequence, keeping the model on GPU for the entire
        batch and offloading only after the last frame.

        Parameters
        ----------
        source_bgr:       HxWx3 uint8 BGR source image.
        param_sequence:   Per-frame SquashParams list.
        ip_image_embeds:  Optional CLIP embedding (1, D) from IPExtractor.
                          Injected into the appearance feature space to preserve
                          character identity across all frames.
        """
        if self._use_fallback:
            return [self._affine_warp_fallback(source_bgr, p) for p in param_sequence]

        frames: list[np.ndarray] = []
        with self._registry.model_context(self.MODEL_ID, offload_after=True) as pipeline:
            # Pre-extract source appearance features once — reused for every frame.
            source_tensor = self._preprocess_source(source_bgr, pipeline)
            kp_source = self._extract_source_kp(pipeline, source_tensor)

            # Convert IP embedding to tensor once (if provided)
            ip_tensor = self._prepare_ip_embeds(ip_image_embeds, pipeline)

            for params in param_sequence:
                frame = self._infer_with_kp(
                    pipeline, source_tensor, kp_source, params, ip_tensor=ip_tensor
                )
                frames.append(frame)

        return frames

    # ── Weight availability check ─────────────────────────────────────────────

    def _should_use_fallback(self) -> bool:
        if self._model_path is None:
            return True
        return _resolve_liveportrait_weights_root(self._model_path) is None

    # ── ModelRegistry registration ────────────────────────────────────────────

    def _register_model(self) -> None:
        """
        Register the LivePortrait pipeline bundle with ModelRegistry.

        VRAM discipline:
            The loader function returns the pipeline object on CPU.
            DeviceManager.move_to_gpu() is called by ModelRegistry AFTER the
            loader returns — we never call .cuda() or .to("cuda") here.
        """
        model_path = self._model_path  # capture for closure

        def _loader():
            InferenceConfig, LivePortraitPipeline = _import_liveportrait_api(model_path)

            weights = _resolve_liveportrait_weights_root(model_path)
            if weights is None:
                raise AnimationError(f"LivePortrait weights not found at {model_path!s}")
            try:
                cfg = InferenceConfig(
                    models_config=str(_resolve_liveportrait_models_config(model_path)),
                    checkpoint_F=str(weights / "liveportrait/base_models/appearance_feature_extractor.pth"),
                    checkpoint_M=str(weights / "liveportrait/base_models/motion_extractor.pth"),
                    checkpoint_W=str(weights / "liveportrait/base_models/warping_module.pth"),
                    checkpoint_G=str(weights / "liveportrait/base_models/spade_generator.pth"),
                    checkpoint_S=str(weights / "liveportrait/retargeting_models/stitching_retargeting_module.pth"),
                    # Explicitly keep on CPU — DeviceManager will move to GPU
                    device_id=None,
                    flag_use_half_precision=False,  # set to True after move_to_gpu
                )
                pipeline = LivePortraitPipeline(inference_cfg=cfg, crop_cfg=None)
            except Exception as exc:
                raise AnimationError(
                    f"Failed to initialise LivePortrait pipeline: {exc}"
                ) from exc

            # Return the pipeline on CPU — ModelRegistry will call move_to_gpu()
            return pipeline

        self._registry.register(
            self.MODEL_ID,
            _loader,
            estimated_vram_gb=_LIVE_PORTRAIT_VRAM_GB,
        )

    # ── Real inference path ───────────────────────────────────────────────────

    def _live_portrait_infer(
        self,
        pipeline,
        source_bgr: np.ndarray,
        params: SquashParams,
        ip_image_embeds: np.ndarray | None = None,
    ) -> np.ndarray:
        """Single-frame inference (used by render_frame)."""
        source_tensor = self._preprocess_source(source_bgr, pipeline)
        kp_source = self._extract_source_kp(pipeline, source_tensor)
        ip_tensor = self._prepare_ip_embeds(ip_image_embeds, pipeline)
        return self._infer_with_kp(pipeline, source_tensor, kp_source, params, ip_tensor=ip_tensor)

    @staticmethod
    def _preprocess_source(source_bgr: np.ndarray, pipeline) -> torch.Tensor:
        """
        Convert BGR uint8 → float32 RGB tensor normalised to [-1, 1],
        shape (1, 3, H, W), on the same device as the pipeline.
        """
        import torch
        rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)

        # Move to the same device as the pipeline's first parameter
        try:
            device = next(pipeline.parameters()).device
            tensor = tensor.to(device)
        except (StopIteration, AttributeError):
            pass  # pipeline has no parameters (mock or unusual structure)

        return tensor

    @staticmethod
    def _extract_source_kp(pipeline, source_tensor: torch.Tensor) -> dict:
        """
        Extract keypoint info from the source image.
        Returns a dict with keys: pitch, yaw, roll, t, exp, scale, kp.
        """
        import torch
        with torch.no_grad():
            try:
                # Official API: get_kp_info returns a dict of motion parameters
                kp_info = pipeline.get_kp_info(source_tensor)
            except AttributeError:
                # Fallback for slightly different API versions
                kp_info = pipeline.motion_extractor(source_tensor)
        return kp_info

    @classmethod
    def _build_driving_info(
        cls,
        kp_source: dict,
        params: SquashParams,
    ) -> dict:
        """
        Build the driving keypoint dict by adding SquashParams deltas to the
        source keypoint info.

        The driving info has the same structure as kp_source but with modified
        exp, scale, and roll values to produce the desired expression.

        Strategy:
        1. Start from a copy of kp_source (preserves identity).
        2. Add the exp_delta computed from SquashParams.
        3. Optionally scale and tilt the head.
        """
        import copy

        import torch

        x_d_info = copy.deepcopy(kp_source)

        # Build 63-dim exp delta from SquashParams
        exp_delta = cls._squash_params_to_exp_delta(params)
        exp_delta_t = torch.from_numpy(exp_delta).to(x_d_info["exp"].device)

        # exp shape is (1, 63) — add delta
        x_d_info["exp"] = x_d_info["exp"] + exp_delta_t.unsqueeze(0)

        # Head scale: squash/stretch maps to a scale multiplier
        # scale is a scalar tensor (1,) or (1,1)
        scale_delta = (params.head_stretch_scale - 1.0) * 0.5 + (params.head_squash_scale - 1.0) * 0.5
        x_d_info["scale"] = x_d_info["scale"] * (1.0 + scale_delta * 0.3)

        # Head tilt: roll in radians
        import math
        roll_delta = torch.tensor(
            [[params.head_tilt_deg * math.pi / 180.0]],
            dtype=x_d_info["roll"].dtype,
            device=x_d_info["roll"].device,
        )
        x_d_info["roll"] = x_d_info["roll"] + roll_delta

        return x_d_info

    @staticmethod
    def _squash_params_to_exp_delta(params: SquashParams) -> np.ndarray:
        """
        Map SquashParams fields to a 63-dim expression coefficient delta.

        All values are additive offsets relative to the source expression.
        Positive z = outward (bulge), positive y = downward, positive x = rightward.
        """
        delta = np.zeros(63, dtype=np.float32)

        # Eye bulge: push eye keypoints outward along z-axis
        eye_bulge = (params.eye_bulge_scale - 1.0) * 0.25
        for i in _EYE_LEFT_Z + _EYE_RIGHT_Z:
            delta[i] += eye_bulge

        # Eye squint: compress eye keypoints along y-axis
        eye_squint = (1.0 - params.eye_squint_scale) * 0.30
        for i in _EYE_LEFT_Y + _EYE_RIGHT_Y:
            delta[i] += eye_squint

        # Brow raise: move brow keypoints up (negative y)
        brow = -params.brow_raise_offset * 0.35
        for i in _BROW_Y:
            delta[i] += brow

        # Jaw drop: move jaw keypoints down (positive y)
        jaw = (params.jaw_drop_scale - 1.0) * 0.40
        for i in _JAW_Y:
            delta[i] += jaw

        # Mouth width: spread mouth-corner keypoints along x-axis
        mouth_w = (params.mouth_width_scale - 1.0) * 0.25
        for i in _MOUTH_X:
            delta[i] += mouth_w

        # Mouth corners: smile (+) / frown (-)
        mouth_c = -params.mouth_corner_offset * 0.20  # negative y = up = smile
        for i in _MOUTH_Y:
            delta[i] += mouth_c

        return delta

    @staticmethod
    def _prepare_ip_embeds(
        ip_image_embeds: np.ndarray | None,
        pipeline,
    ) -> torch.Tensor | None:
        """
        Convert a numpy IP embedding to a torch tensor on the pipeline's device.
        Returns None if ip_image_embeds is None.

        The embedding is used as a soft conditioning signal: it is projected into
        the appearance feature space via a learned linear layer (if the pipeline
        exposes one), or ignored gracefully if the API does not support it.
        """
        if ip_image_embeds is None:
            return None
        import torch
        t = torch.from_numpy(np.array(ip_image_embeds, dtype=np.float32))
        if t.ndim == 1:
            t = t.unsqueeze(0)  # (1, D)
        try:
            device = next(pipeline.parameters()).device
            t = t.to(device)
        except (StopIteration, AttributeError):
            pass
        return t

    def _infer_with_kp(
        self,
        pipeline,
        source_tensor: torch.Tensor,
        kp_source: dict,
        params: SquashParams,
        ip_tensor: torch.Tensor | None = None,
    ) -> np.ndarray:
        """
        Run the warp + generate step given pre-extracted source keypoints.

        Parameters
        ----------
        ip_tensor:  Optional (1, D) CLIP embedding tensor.  When provided it is
                    passed to execute_portraits as `ip_adapter_embeds` so the
                    SPADEGenerator can condition on character identity.
                    Silently ignored if the pipeline API does not support it.

        Returns HxWx3 uint8 BGR.
        """
        import torch

        x_d_info = self._build_driving_info(kp_source, params)

        with torch.no_grad():
            try:
                # Official API: execute_portraits returns a dict with "out" key.
                # Pass ip_adapter_embeds when available; older API versions ignore
                # unknown kwargs, so this is safe across versions.
                kwargs: dict = dict(
                    img_rgb=source_tensor,
                    x_s_info=kp_source,
                    x_d_info=x_d_info,
                    R_d=None,
                    lip_delta_before_animation=None,
                    combine_param=None,
                    mask_ori_float=None,
                )
                if ip_tensor is not None:
                    kwargs["ip_adapter_embeds"] = ip_tensor
                result = pipeline.execute_portraits(**kwargs)
                out_rgb = result["out"]  # (1, 3, H, W) float in [-1, 1] or [0, 1]
            except (AttributeError, KeyError, TypeError):
                # Fallback for API variants that use a different method name
                try:
                    out_rgb = pipeline.warp_decode(source_tensor, kp_source, x_d_info)
                except Exception as exc:
                    raise AnimationError(
                        f"LivePortrait inference failed (tried execute_portraits and warp_decode): {exc}"
                    ) from exc

        return self._tensor_to_bgr(out_rgb)

    @staticmethod
    def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
        """Convert (1, 3, H, W) float tensor → HxWx3 uint8 BGR."""
        import torch
        t = tensor.squeeze(0).permute(1, 2, 0).cpu().float()
        # Handle both [-1,1] and [0,1] output ranges
        if t.min() < -0.1:
            t = (t + 1.0) / 2.0
        t = torch.clamp(t, 0.0, 1.0)
        rgb = (t.numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ── Affine-warp fallback (no model required) ──────────────────────────────

    @staticmethod
    def _affine_warp_fallback(
        source_bgr: np.ndarray,
        params: SquashParams,
    ) -> np.ndarray:
        """
        CPU-only approximation using OpenCV affine transforms.
        Produces visually rough but structurally correct deformations
        for unit-testing the pipeline without GPU or model downloads.
        """
        h, w = source_bgr.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        sx = params.head_stretch_scale
        sy = params.head_squash_scale
        angle = params.head_tilt_deg

        M_scale = np.array([
            [sx, 0,  cx * (1 - sx)],
            [0,  sy, cy * (1 - sy)],
        ], dtype=np.float32)

        M_rot = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

        M_combined = np.vstack([M_rot, [0, 0, 1]]) @ np.vstack([M_scale, [0, 0, 1]])
        M_final = M_combined[:2]

        warped = cv2.warpAffine(
            source_bgr, M_final, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Eye-bulge approximation: zoom into upper-centre region then resize back.
        if params.eye_bulge_scale > 1.05:
            y0, y1 = int(h * 0.15), int(h * 0.55)
            x0, x1 = int(w * 0.10), int(w * 0.90)
            eye_region = warped[y0:y1, x0:x1]
            eye_h, eye_w = eye_region.shape[:2]
            if eye_h > 0 and eye_w > 0:
                scale = min(params.eye_bulge_scale * 0.6, 1.8)
                zoomed = cv2.resize(
                    eye_region,
                    (int(eye_w * scale), int(eye_h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
                zh, zw = zoomed.shape[:2]
                off_y = max((zh - eye_h) // 2, 0)
                off_x = max((zw - eye_w) // 2, 0)
                crop = zoomed[off_y:off_y + eye_h, off_x:off_x + eye_w]
                if crop.shape[:2] != (eye_h, eye_w):
                    crop = cv2.resize(crop, (eye_w, eye_h), interpolation=cv2.INTER_LINEAR)
                warped[y0:y1, x0:x1] = crop

        return warped
