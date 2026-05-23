"""v0.7.1: `imgen doctor` HF whoami ping helper.

Surfaces dead / revoked / wrong-scope tokens BEFORE the user wastes
~13s on a first `snapshot_download` attempt that 401s buried inside a
mflux+huggingface_hub stack trace. Real-mflux smoke during the v0.7.0
pre-tag round caught the failure mode live — user's old token had
been revoked but doctor reported "HF_TOKEN found" (presence-only).

Tests target `_ping_hf_whoami_and_report` (extracted from inline
cmd_doctor for testability) so no full cmd_doctor environment mock
needed. HfApi is monkeypatched at the import site the helper uses.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout, redirect_stderr
from types import SimpleNamespace

import pytest

from huggingface_hub.errors import HfHubHTTPError


def _capture(monkeypatch, token: str) -> tuple[int, str]:
    """Run the helper with redirected I/O; return (issue_delta, output)."""
    from imgen.commands.doctor import _ping_hf_whoami_and_report
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        delta = _ping_hf_whoami_and_report(token)
    return delta, out.getvalue() + err.getvalue()


class _FakeHfHubHTTPError(HfHubHTTPError):
    """HfHubHTTPError's __init__ inspects response.headers /
    response.request, so constructing one with a SimpleNamespace fails.
    Subclass that skips parent init, sets `.response` directly — that's
    the only attribute our helper reads (status_code via response).
    Mirrors how huggingface_hub itself constructs subclasses in its
    `_format` helper."""

    def __init__(self, status_code: int):
        Exception.__init__(self, f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)
        self.server_message = None
        self.request_id = None


class TestHfWhoamiPing:
    def test_valid_token_returns_0_and_logs_username(self, monkeypatch):
        """Happy path: HF whoami returns the user payload → status line
        carries the username, no issue counter bump."""
        class FakeHfApi:
            def whoami(self, token):
                return {"name": "stanislav", "fullname": "Stanislav K"}
        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        delta, output = _capture(monkeypatch, "hf_validtoken1234")
        assert delta == 0
        assert "logged in as stanislav" in output

    def test_401_returns_1_and_emits_remediation(self, monkeypatch):
        """v0.7.1 headline: a 401 → loud warn + remediation hint +
        issue counter bump. Caught the live failure mode user hit
        during v0.7.0 smoke."""
        class FakeHfApi:
            def whoami(self, token):
                raise _FakeHfHubHTTPError(401)
        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        delta, output = _capture(monkeypatch, "hf_revoked1234")
        assert delta == 1
        assert "token is INVALID" in output
        assert "huggingface.co/settings/tokens" in output

    def test_non_401_http_returns_1(self, monkeypatch):
        """Other HTTP errors (500, 503, etc) also bump issues — they
        indicate the token couldn't be validated even though the
        request reached HF."""
        class FakeHfApi:
            def whoami(self, token):
                raise _FakeHfHubHTTPError(503)
        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        delta, output = _capture(monkeypatch, "hf_anytoken")
        assert delta == 1
        assert "unexpected HTTP 503" in output

    def test_network_failure_returns_0(self, monkeypatch):
        """Air-gapped Mac / DNS down / HF outage: warn but DON'T mark
        as a blocking issue. User's token might be fine; we just
        can't verify right now."""
        class FakeHfApi:
            def whoami(self, token):
                raise ConnectionError("network down")
        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        delta, output = _capture(monkeypatch, "hf_token")
        assert delta == 0
        assert "could not reach HuggingFace" in output
        assert "ConnectionError" in output

    def test_timeout_returns_0(self, monkeypatch):
        """Slow network / HF timeout: same as ConnectionError — warn
        without bumping issues."""
        class FakeHfApi:
            def whoami(self, token):
                raise TimeoutError("timed out")
        monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
        delta, output = _capture(monkeypatch, "hf_token")
        assert delta == 0
        assert "could not reach HuggingFace" in output

    def test_missing_huggingface_hub_returns_0(self, monkeypatch):
        """If huggingface_hub isn't importable (broken venv), the
        helper returns 0 — the mflux check earlier in doctor would
        have already failed loudly. Belt-and-braces."""
        import sys
        # Inject a sentinel module that raises on access to HfApi.
        # Cleanest: monkeypatch the import to raise ImportError.
        orig = sys.modules.get("huggingface_hub")
        sys.modules["huggingface_hub"] = None  # type: ignore[assignment]
        try:
            delta, output = _capture(monkeypatch, "hf_token")
            # ImportError path returns 0 silently — nothing user-facing.
            assert delta == 0
        finally:
            if orig is not None:
                sys.modules["huggingface_hub"] = orig
            else:
                del sys.modules["huggingface_hub"]
