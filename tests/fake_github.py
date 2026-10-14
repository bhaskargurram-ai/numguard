"""An in-memory fake of the few GitHub REST endpoints numguard uses (as an HTTP transport)."""

from __future__ import annotations

import json
import re
from urllib.parse import unquote

from numguard.httpclient import HTTPResponse


class FakeGitHub:
    def __init__(self, repo: str = "o/r", head_sha: str = "a" * 40, files=None):
        self.repo = repo
        self.head_sha = head_sha
        self.files = list(files or ["src/x.py"])
        self.comments: list[dict] = []
        self.labels: set[str] = set()
        self.issue_labels: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.next_id = 100

    def add_comment(self, body: str, login: str = "github-actions[bot]", typ: str = "Bot") -> dict:
        c = {"id": self.next_id, "body": body, "user": {"login": login, "type": typ}}
        self.next_id += 1
        self.comments.append(c)
        return c

    def _json(self, status: int, data) -> HTTPResponse:
        return HTTPResponse(status, json.dumps(data).encode())

    def __call__(self, method, url, headers, body, timeout):
        path = url.split("https://api.github.com", 1)[1]
        self.calls.append((method, path))
        assert headers.get("Authorization", "").startswith("Bearer "), "token missing"
        data = json.loads(body) if body else None
        base = f"/repos/{self.repo}"
        if method == "GET" and re.fullmatch(base + r"/pulls/\d+", path):
            return self._json(200, {"head": {"sha": self.head_sha}, "number": int(path.rsplit("/", 1)[1])})
        if method == "GET" and re.match(base + r"/pulls/\d+/files", path):
            page = int(re.search(r"[?&]page=(\d+)", path).group(1))
            return self._json(200, [{"filename": f} for f in self.files] if page == 1 else [])
        if method == "GET" and re.match(base + r"/issues/\d+/comments", path):
            page = int(re.search(r"[?&]page=(\d+)", path).group(1))
            return self._json(200, self.comments if page == 1 else [])
        if method == "POST" and re.fullmatch(base + r"/issues/\d+/comments", path):
            return self._json(201, self.add_comment(data["body"]))
        m = re.fullmatch(base + r"/issues/comments/(\d+)", path)
        if method == "PATCH" and m:
            for c in self.comments:
                if c["id"] == int(m.group(1)):
                    c["body"] = data["body"]
                    return self._json(200, c)
            return self._json(404, {})
        if method == "GET" and re.match(base + r"/issues/\d+/labels", path):
            return self._json(200, [{"name": n} for n in self.issue_labels])
        m = re.fullmatch(base + r"/labels/(.+)", path)
        if method == "GET" and m:
            return self._json(200 if unquote(m.group(1)) in self.labels else 404, {})
        if method == "POST" and path == base + "/labels":
            self.labels.add(data["name"])
            return self._json(201, {})
        if method == "POST" and re.fullmatch(base + r"/issues/\d+/labels", path):
            self.issue_labels.extend(data["labels"])
            return self._json(200, [])
        m = re.fullmatch(base + r"/issues/\d+/labels/(.+)", path)
        if method == "DELETE" and m:
            name = unquote(m.group(1))
            if name in self.issue_labels:
                self.issue_labels.remove(name)
            return self._json(200, [])
        return self._json(404, {"message": f"fake: no route for {method} {path}"})
