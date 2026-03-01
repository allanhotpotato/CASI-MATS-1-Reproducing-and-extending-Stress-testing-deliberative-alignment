"""
Qwen-QwQ-32B model wrapper (secondary model for cross-model comparison).

Qwen-QwQ is an open-source reasoning model that also exposes chain-of-thought
traces, enabling direct comparison with DeepSeek-R1 on covert action rates.

Served via a compatible OpenAI-format API endpoint (e.g. Together AI or local
vLLM instance).

Required env vars:
    QWEN_API_KEY      — API key for the serving endpoint
    QWEN_BASE_URL     — Base URL of the OpenAI-compatible endpoint
"""

from __future__ import annotations

import os
from typing import Any

from environments.base_environment import Trajectory
from models.base_model import BaseModel
from moltbook.logger import TrajectoryLogger


class QwenQwQ(BaseModel):

    MODEL_ID = "Qwen/QwQ-32B-Preview"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ["QWEN_API_KEY"]
        self.base_url = base_url or os.environ["QWEN_BASE_URL"]
        # TODO: initialise OpenAI-compatible client pointing at base_url

    @property
    def name(self) -> str:
        return "qwen-qwq-32b"

    def run(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        environment_name: str,
    ) -> Trajectory:
        """
        Run the agentic loop against Qwen-QwQ-32B.

        Mirrors the DeepSeek-R1 loop; CoT is extracted from the reasoning
        field of the response (format depends on serving endpoint).
        """
        logger = TrajectoryLogger(environment=environment_name, model=self.name)
        # TODO: implement agentic loop
        raise NotImplementedError
