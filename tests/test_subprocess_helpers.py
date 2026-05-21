"""Stderr token redactor regex coverage.

`_TOKEN_LEAK_RE` is the safety net that strips `hf_*` tokens from mflux
stderr before it reaches the user's terminal. Lock the pattern's behaviour
so a future "tighten" or "loosen" is intentional.
"""
from __future__ import annotations

import re

from imgen.subprocess_helpers import _TOKEN_LEAK_RE


def test_redacts_full_length_token():
    """Real HF tokens are 36+ chars; the redactor must catch them."""
    line = b"Authorization: Bearer hf_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789abcdef\n"
    result = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", line)
    assert b"hf_AbCd" not in result
    assert b"hf_***REDACTED***" in result


def test_does_not_redact_short_prefix_hf_x():
    """`hf_` followed by < 36 chars must NOT match. Without this lower
    bound, a buffer-boundary split could flush `hf_` + a few chars as
    plaintext (rest of the token would land in next chunk, never
    redacted because that chunk doesn't have an `hf_` prefix).
    (security N1)"""
    line = b"random output mentioning hf_short here\n"
    result = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", line)
    # `hf_short` is 8 chars after the prefix — must NOT be redacted
    # because it CAN'T be a real token.
    assert b"hf_short" in result


def test_does_not_match_words_starting_with_hf():
    """Defensive: don't redact e.g. "hf_huggingface_xyz" (8+ chars but
    in a normal sentence) ... actually it would be redacted because it
    looks like a token. The 36+ floor is what protects normal English
    words like "hf_xyz" from false matches."""
    short_word = b"hf_quick" + b"abc"  # 11 chars after hf_
    result = _TOKEN_LEAK_RE.sub(b"hf_***REDACTED***", short_word)
    assert short_word in result  # too short to match {36,}


def test_pattern_minimum_length_is_36():
    """Pin the minimum-match length. If the regex changes from {36,} to
    something looser, this test loudly fails to force the reviewer to
    re-justify the choice."""
    pattern_src = _TOKEN_LEAK_RE.pattern
    assert b"{36,}" in pattern_src or "{36,}" in str(pattern_src), \
        f"redactor pattern {pattern_src!r} missing {{36,}} min-length"
