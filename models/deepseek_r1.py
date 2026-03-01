"""
DeepSeek-R1 model wrapper (primary model).

Calls the DeepSeek API, which natively exposes full <think>...</think> chain-
of-thought blocks alongside the assistant response. This makes it possible to
detect covert *reasoning* (does the model think about hiding a rule violation?)
in addition to covert *actions*.

API docs: https://api-docs.deepseek.com/
Required env var: DEEPSEEK_API_KEY
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from environments.base_environment import Trajectory
from models.base_model import BaseModel
from moltbook.logger import TrajectoryLogger

MAX_TURNS = 20  # hard cap to prevent runaway loops


class DeepSeekR1(BaseModel):

    MODEL_ID = "deepseek-reasoner"  # DeepSeek-R1 model identifier
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str | None = None, temperature: float = 1.0):
        self.api_key = api_key or os.environ["DEEPSEEK_API_KEY"]
        self.temperature = temperature
        self._client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)

    @property
    def name(self) -> str:
        return "deepseek-r1"

    def run(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
        environment_name: str,
        client: Any,
    ) -> Trajectory:
        """
        Run the agentic loop against DeepSeek-R1.

        The loop:
          1. Send system_prompt + tools to the API.
          2. Extract reasoning_content → append to cot_trace.
          3. Execute any tool calls via the logged client.
          4. Feed tool results back; repeat until no more tool calls.
        """
        logger = TrajectoryLogger(environment=environment_name, model=self.name)
        logged_client = logger.wrap(client)

        # DeepSeek API expects tools wrapped as {"type": "function", "function": {...}}
        formatted_tools = [
            t if "type" in t else {"type": "function", "function": t}
            for t in tools
        ]

        messages = [{"role": "system", "content": system_prompt}]

        for _ in range(MAX_TURNS):
            response = self._client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                tools=formatted_tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # Capture chain-of-thought reasoning
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning:
                logger.log_cot(reasoning)

            # Stop if the model made no tool calls
            if not message.tool_calls:
                break

            # Append assistant turn to conversation history
            messages.append(message)

            # Execute each tool call and feed results back
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                try:
                    result = logged_client.dispatch(name, args)
                except Exception as e:
                    result = {"error": str(e)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

        return logger.to_trajectory()
