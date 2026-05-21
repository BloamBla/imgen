"""HuggingFace token: load / validate / atomic save."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .paths import TOKEN_FILE


def load_token() -> str | None:
    """Load HF token from $HF_TOKEN or ~/.hf_token."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()
    return None


def check_token_perms() -> bool:
    """Return True if token file has safe permissions (600)."""
    if not TOKEN_FILE.exists():
        return True
    mode = TOKEN_FILE.stat().st_mode & 0o777
    return mode == 0o600


def save_token_atomic(tok: str) -> None:
    """Write token with 0600 perms atomically.

    O_CREAT|O_EXCL ensures no world-readable window between write and chmod.
    Caller must delete the existing file first if updating.
    """
    fd = os.open(str(TOKEN_FILE),
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(tok)


def validate_token(token: str) -> str | None:
    """Hit HF whoami; return username on success, None on failure."""
    try:
        req = urllib.request.Request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            # Cap response size — defends against DNS hijack / captive portal
            # serving arbitrary bytes.
            raw = resp.read(64_000)
            if len(raw) >= 64_000:
                return None
            data = json.loads(raw)
            return data.get("name") or data.get("fullname")
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None
