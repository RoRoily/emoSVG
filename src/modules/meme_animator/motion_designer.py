"""
Squash-and-Stretch motion curve designer.

Converts a MemeExpression preset (or raw SquashParams) into a per-frame
sequence that describes the full animation arc:
    neutral -> wind-up -> peak -> hold -> bounce-back -> neutral

No model inference here — pure math / interpolation.
"""
from __future__ import annotations

import math

from .schemas import MemeExpression, SquashParams

# ── Expression presets ────────────────────────────────────────────────────

_PRESETS: dict[MemeExpression, SquashParams] = {
    MemeExpression.SHOCK: SquashParams(
        eye_bulge_scale=2.8, brow_raise_offset=0.7,
        jaw_drop_scale=2.2, mouth_width_scale=1.4,
        head_squash_scale=0.85, head_stretch_scale=1.15,
        hold_frames=5, bounce_frames=4,
    ),
    MemeExpression.LAUGH: SquashParams(
        eye_squint_scale=0.3, mouth_width_scale=2.0,
        mouth_corner_offset=0.8, head_squash_scale=0.75,
        head_stretch_scale=1.25, hold_frames=6, bounce_frames=3,
    ),
    MemeExpression.CRY: SquashParams(
        eye_squint_scale=0.5, brow_raise_offset=-0.4,
        jaw_drop_scale=1.6, mouth_corner_offset=-0.7,
        head_squash_scale=0.9, head_tilt_deg=-8.0,
        hold_frames=8, bounce_frames=2,
    ),
    MemeExpression.RAGE: SquashParams(
        eye_squint_scale=0.4, brow_raise_offset=-0.8,
        jaw_drop_scale=1.3, mouth_width_scale=1.6,
        head_squash_scale=1.2, head_stretch_scale=0.85,
        head_tilt_deg=5.0, hold_frames=6, bounce_frames=2,
    ),
    MemeExpression.SMUG: SquashParams(
        eye_squint_scale=0.6, brow_raise_offset=-0.2,
        mouth_corner_offset=0.5, head_tilt_deg=10.0,
        hold_frames=8, bounce_frames=1,
    ),
    MemeExpression.SURPRISED: SquashParams(
        eye_bulge_scale=2.2, brow_raise_offset=0.9,
        jaw_drop_scale=1.8, mouth_width_scale=1.2,
        head_squash_scale=0.9, head_stretch_scale=1.1,
        hold_frames=4, bounce_frames=3,
    ),
}

_NEUTRAL = SquashParams()
_NUMERIC_FIELDS = [
    "eye_bulge_scale", "eye_squint_scale", "brow_raise_offset",
    "jaw_drop_scale", "mouth_width_scale", "mouth_corner_offset",
    "head_squash_scale", "head_stretch_scale", "head_tilt_deg",
]


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease_out_elastic(t: float, amplitude: float = 1.0, period: float = 0.4) -> float:
    if t in (0.0, 1.0):
        return t
    s = period / (2 * math.pi) * math.asin(1.0 / amplitude) if amplitude >= 1 else period / 4
    return amplitude * (2 ** (-10 * t)) * math.sin((t - s) * (2 * math.pi) / period) + 1.0


def _ease_in_quad(t: float) -> float:
    return t * t


def _interp(a: SquashParams, b: SquashParams, t: float) -> SquashParams:
    kwargs = {f: _lerp(getattr(a, f), getattr(b, f), t) for f in _NUMERIC_FIELDS}
    kwargs["hold_frames"]   = b.hold_frames
    kwargs["bounce_frames"] = b.bounce_frames
    return SquashParams(**kwargs)


class MotionDesigner:
    """
    Produces a list[SquashParams] — one entry per output frame — encoding
    the full Squash-and-Stretch arc for a given expression.

    Arc structure:
        [WIND_UP_FRAMES] wind-up (anticipation)
        [ATTACK_FRAMES]  snap to peak
        [hold_frames]    freeze at peak
        [bounce_frames]  elastic return
        [SETTLE_FRAMES]  linear glide to neutral
    """

    WIND_UP_FRAMES = 3
    ATTACK_FRAMES  = 4
    SETTLE_FRAMES  = 5

    def design(
        self,
        expression: MemeExpression,
        custom_params: SquashParams | None = None,
    ) -> list[SquashParams]:
        if expression == MemeExpression.CUSTOM:
            if custom_params is None:
                raise ValueError("custom_params required for CUSTOM expression")
            peak = custom_params
        else:
            peak = _PRESETS[expression]

        frames: list[SquashParams] = []
        wind_up = self._wind_up(peak)

        for i in range(self.WIND_UP_FRAMES):
            t = _ease_in_quad((i + 1) / self.WIND_UP_FRAMES)
            frames.append(_interp(_NEUTRAL, wind_up, t))

        for i in range(self.ATTACK_FRAMES):
            t = _ease_in_quad((i + 1) / self.ATTACK_FRAMES)
            frames.append(_interp(wind_up, peak, t))

        for _ in range(peak.hold_frames):
            frames.append(peak)

        for i in range(peak.bounce_frames):
            t = _ease_out_elastic((i + 1) / peak.bounce_frames)
            frames.append(_interp(peak, _NEUTRAL, t))

        for i in range(self.SETTLE_FRAMES):
            t = (i + 1) / self.SETTLE_FRAMES
            frames.append(_interp(frames[-1], _NEUTRAL, t))

        return frames

    @staticmethod
    def _wind_up(peak: SquashParams) -> SquashParams:
        """Anticipation frame: subtle counter-pose before the main hit."""
        def _counter(v: float, strength: float = 0.15) -> float:
            return 1.0 - (v - 1.0) * strength

        return SquashParams(
            eye_bulge_scale=_counter(peak.eye_bulge_scale),
            eye_squint_scale=_counter(peak.eye_squint_scale),
            brow_raise_offset=-peak.brow_raise_offset * 0.2,
            jaw_drop_scale=_counter(peak.jaw_drop_scale),
            mouth_width_scale=_counter(peak.mouth_width_scale),
            mouth_corner_offset=-peak.mouth_corner_offset * 0.2,
            head_squash_scale=_counter(peak.head_squash_scale),
            head_stretch_scale=_counter(peak.head_stretch_scale),
            head_tilt_deg=-peak.head_tilt_deg * 0.3,
            hold_frames=peak.hold_frames,
            bounce_frames=peak.bounce_frames,
        )
