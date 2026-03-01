"""
Trajectory logger for Moltbook API calls.

Wraps a MoltbookClient and intercepts every call, appending a structured
record to the trajectory log. This gives us the ordered API call sequence
needed for rule-based classification.

Usage:
    logger = TrajectoryLogger(environment="moderator_task", model="deepseek-r1")
    logged_client = logger.wrap(raw_client)
    # use logged_client in place of raw_client during the rollout
    trajectory = logger.to_trajectory()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from environments.base_environment import Trajectory


@dataclass
class APICallRecord:
    name: str
    arguments: dict[str, Any]
    response: Any
    timestamp: float = field(default_factory=time.time)


class TrajectoryLogger:

    def __init__(self, environment: str, model: str):
        self.environment = environment
        self.model = model
        self._api_calls: list[APICallRecord] = []
        self._cot_trace: list[str] = []

    def log_cot(self, cot_block: str) -> None:
        """Append a <think> block to the CoT trace."""
        self._cot_trace.append(cot_block)

    def log_api_call(self, name: str, arguments: dict, response: Any) -> None:
        self._api_calls.append(APICallRecord(name=name, arguments=arguments, response=response))

    def wrap(self, client: Any) -> "LoggedMoltbookClient":
        """Return a proxy client that logs all calls to this logger.

        Accepts both MoltbookClient and MockMoltbookClient (duck typing).
        """
        return LoggedMoltbookClient(client, self)

    def to_trajectory(self) -> Trajectory:
        return Trajectory(
            environment=self.environment,
            model=self.model,
            cot_trace=list(self._cot_trace),
            api_calls=[
                {"name": r.name, "arguments": r.arguments, "response": r.response, "timestamp": r.timestamp}
                for r in self._api_calls
            ],
        )


class LoggedMoltbookClient:
    """Proxy that logs every client call before delegating to the underlying client."""

    def __init__(self, client: Any, logger: TrajectoryLogger):
        self._client = client
        self._logger = logger

    def _logged(self, name: str, fn: Callable, **kwargs) -> Any:
        response = fn(**kwargs)
        self._logger.log_api_call(name=name, arguments=kwargs, response=response)
        return response

    def list_posts(self, **kwargs):
        return self._logged("list_posts", self._client.list_posts, **kwargs)

    def get_post(self, **kwargs):
        return self._logged("get_post", self._client.get_post, **kwargs)

    def create_post(self, **kwargs):
        return self._logged("create_post", self._client.create_post, **kwargs)

    def delete_post(self, **kwargs):
        return self._logged("delete_post", self._client.delete_post, **kwargs)

    def search_posts(self, **kwargs):
        return self._logged("search_posts", self._client.search_posts, **kwargs)

    def create_comment(self, **kwargs):
        return self._logged("create_comment", self._client.create_comment, **kwargs)

    def send_dm(self, **kwargs):
        return self._logged("send_dm", self._client.send_dm, **kwargs)

    def dispatch(self, name: str, arguments: dict) -> Any:
        """Generic dispatcher: maps a tool name to the corresponding method.

        Used by model wrappers so they can execute any tool call without a
        hard-coded if/elif chain.
        """
        method = getattr(self, name, None)
        if method is None:
            raise ValueError(f"Unknown tool: {name!r}")
        return method(**arguments)
