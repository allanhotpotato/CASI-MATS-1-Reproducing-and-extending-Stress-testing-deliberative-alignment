"""
In-memory mock of the Moltbook API.

Replaces MoltbookClient for development and testing — no network, no account
required. State is held in plain Python dicts and reset between rollouts by
calling reset() on the environment (which creates a fresh MockMoltbookClient).

The mock exposes the same method interface as MoltbookClient so it can be
passed anywhere a real client is expected.
"""

from typing import Any


class MockMoltbookClient:

    def __init__(self):
        self._posts: dict[str, dict] = {}
        self._dms: list[dict] = []
        self._next_id: int = 1

    # --- Seed helpers (called by environment reset()) ---

    def seed_post(self, post: dict) -> str:
        """Insert a post into the mock state. Returns the post_id."""
        post_id = post.get("id", self._new_id("post"))
        self._posts[post_id] = {**post, "id": post_id, "comments": []}
        return post_id

    # --- MoltbookClient interface ---

    def list_posts(self, submolt: str, limit: int = 20) -> list[dict]:
        return [p for p in self._posts.values() if p.get("submolt") == submolt][:limit]

    def get_post(self, post_id: str) -> dict:
        if post_id not in self._posts:
            raise KeyError(f"Post not found: {post_id}")
        return self._posts[post_id]

    def create_post(self, submolt: str, title: str, body: str) -> dict:
        post_id = self._new_id("post")
        post = {"id": post_id, "submolt": submolt, "title": title, "body": body, "comments": []}
        self._posts[post_id] = post
        return post

    def delete_post(self, post_id: str) -> dict:
        self._posts.pop(post_id, None)
        return {"success": True, "deleted_id": post_id}

    def search_posts(self, submolt: str, query: str, days: int = 7) -> list[dict]:
        q = query.lower()
        return [
            p for p in self._posts.values()
            if p.get("submolt") == submolt
            and (q in p.get("title", "").lower() or q in p.get("body", "").lower())
        ]

    def create_comment(self, post_id: str, body: str) -> dict:
        comment_id = self._new_id("comment")
        comment = {"id": comment_id, "post_id": post_id, "body": body}
        if post_id in self._posts:
            self._posts[post_id]["comments"].append(comment)
        return comment

    def send_dm(self, username: str, message: str) -> dict:
        dm = {"to": username, "message": message}
        self._dms.append(dm)
        return {"success": True}

    def grant_mod_role(self, submolt: str, username: str) -> dict:
        return {"success": True}

    # --- Internal ---

    def _new_id(self, prefix: str) -> str:
        id_ = f"{prefix}_{self._next_id:03d}"
        self._next_id += 1
        return id_
