from .motion_designer import MotionDesigner
from .schemas import AnimationRequest, AnimationResult, MemeExpression, SquashParams

__all__ = [
    "MemeAnimator",
    "MotionDesigner",
    "AnimationRequest",
    "AnimationResult",
    "MemeExpression",
    "SquashParams",
]


def __getattr__(name: str):
    if name == "MemeAnimator":
        from .animator import MemeAnimator

        return MemeAnimator
    raise AttributeError(name)
