"""HuggingFace token: load / validate / atomic save."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .colors import warn
from .paths import TOKEN_FILE

# Cap on ~/.hf_token size. Real HF tokens are ~70 chars (`hf_` + 37-char
# secret + room to grow). 4 KB is several orders above realistic use; a
# larger file means something's wrong — refuse rather than slurp into
# memory and pass to mflux.
TOKEN_MAX_BYTES = 4096


def load_token() -> str | None:
    """Load HF token from $HF_TOKEN or ~/.hf_token.

    Files larger than TOKEN_MAX_BYTES are rejected with a warn — a real
    token wouldn't be that long, and passing junk to mflux would just
    fail later with a confusing 401.
    """
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    if not TOKEN_FILE.exists():
        return None
    try:
        size = TOKEN_FILE.stat().st_size
    except OSError as e:
        warn(f"Couldn't stat {TOKEN_FILE}: {e}")
        return None
    if size > TOKEN_MAX_BYTES:
        warn(f"{TOKEN_FILE} too large ({size} bytes; cap {TOKEN_MAX_BYTES}) "
             "— refusing to load. Replace the file with a valid token.")
        return None
    try:
        return TOKEN_FILE.read_text().strip()
    except OSError as e:
        warn(f"Couldn't read {TOKEN_FILE}: {e}")
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
