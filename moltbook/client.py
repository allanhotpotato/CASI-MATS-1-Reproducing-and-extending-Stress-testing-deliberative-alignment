"""
Thin wrapper around the Moltbook REST API.

Exposes the subset of Moltbook endpoints needed by the evaluation environments:
    - List / get posts
    - Create / delete posts
    - Create comments
    - Search posts
    - Send DMs
    - Grant moderator roles

All calls are routed through this client so the TrajectoryLogger can intercept
and record them in the order they were made.

Required env var: MOLTBOOK_API_KEY
Base URL: https://api.moltbook.com (override with MOLTBOOK_BASE_URL)
"""

from __future__ import annotations

import os
from typing import Any

import requests


class MoltbookClient:

    DEFAULT_BASE_URL = "https://api.moltbook.com"

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key or os.environ["MOLTBOOK_API_KEY"]
        self.base_url = (base_url or os.environ.get("MOLTBOOK_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()

    # --- Posts ---

    def list_posts(self, submolt: str, limit: int = 20) -> list[dict]:
        return self._request("GET", f"/submolts/{submolt}/posts", params={"limit": limit})

    def get_post(self, post_id: str) -> dict:
        return self._request("GET", f"/posts/{post_id}")

    def create_post(self, submolt: str, title: str, body: str) -> dict:
        return self._request("POST", f"/submolts/{submolt}/posts", json={"title": title, "body": body})

    def delete_post(self, post_id: str) -> dict:
        return self._request("DELETE", f"/posts/{post_id}")

    def search_posts(self, submolt: str, query: str, days: int = 7) -> list[dict]:
        return self._request("GET", f"/submolts/{submolt}/search", params={"q": query, "days": days})

    # --- Comments ---

    def create_comment(self, post_id: str, body: str) -> dict:
        return self._request("POST", f"/posts/{post_id}/comments", json={"body": body})

    # --- Direct messages ---

    def send_dm(self, username: str, message: str) -> dict:
        return self._request("POST", f"/users/{username}/messages", json={"body": message})

    # --- Moderation ---

    def grant_mod_role(self, submolt: str, username: str) -> dict:
        return self._request("POST", f"/submolts/{submolt}/moderators", json={"username": username})
