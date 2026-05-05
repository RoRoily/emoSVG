"""Abstract base class for all emoSVG pipelines."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BasePipeline(ABC):
    """
    All pipelines expose a single run() entry point.
    Subclasses define their own input/output types.
    """

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the pipeline and return a result object."""

    @classmethod
    @abstractmethod
    def from_config(cls, config_path: str = "configs/base.yaml") -> BasePipeline:
        """Construct the pipeline from a YAML config file."""
