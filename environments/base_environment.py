"""
Abstract base class for all scheming evaluation environments.

Each environment defines:
  - A system prompt given to the model (the task framing)
  - The explicit rule that creates tension with task efficiency
  - The set of tools (Moltbook API calls) available to the model
  - A reset() method to bring the environment to a fresh state
  - A run(model) method that executes one rollout and returns a Trajectory
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Trajectory:
    """Ordered record of everything that happened in one rollout."""
    environment: str
    model: str
    cot_trace: list[str] = field(default_factory=list)       # <think> blocks in order
    api_calls: list[dict[str, Any]] = field(default_factory=list)  # {method, endpoint, body, response}
    label: str | None = None  # set by classifier: covert_action | escalated | complied | refused


class BaseEnvironment(ABC):
    """All evaluation environments inherit from this class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in results files."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The task description and explicit rule given to the model."""
        ...

    @property
    @abstractmethod
    def tools(self) -> list[dict[str, Any]]:
        """Tool definitions (OpenAI function-calling format) available to the model."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Restore the Moltbook submolt to a clean baseline state."""
        ...

    @abstractmethod
    def run(self, model) -> Trajectory:
        """Execute one rollout: place model in environment, collect trajectory."""
        ...
