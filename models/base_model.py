"""
Abstract interface for LLM model wrappers.

A model takes a system prompt and a list of tools and runs the agent loop,
returning a completed Trajectory with the full CoT trace and API call log.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from environments.base_environment import Trajectory


class BaseModel(ABC):
    """All model wrappers inherit from this class."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short model identifier used in results files (e.g. 'deepseek-r1')."""
        ...

    @abstractmethod
    def run(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        environment_name: str,
        client: Any,
    ) -> Trajectory:
        """
        Execute the agent loop.

        Args:
            system_prompt: Full task description and rules given to the model.
            tools: Tool definitions in OpenAI function-calling format.
            environment_name: Used to label the returned Trajectory.
            client: Raw Moltbook client (real or mock). The model wraps it with
                    a TrajectoryLogger internally before using it.

        Returns:
            Trajectory containing cot_trace and api_calls in order.
        """
        ...
