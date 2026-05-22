"""`imgen doctor` alias-vs-IMGEN_HOME consistency check.

Covers the v0.1.x architect #1 footgun: colleague runs `bootstrap.sh`
in ~/projects/imgen, later moves the repo to ~/imgen — the alias in
~/.zshrc still points at the old location and runs stale code on
every invocation. v0.3.6 surfaces divergence in `imgen doctor`.

Tests cover the pure functions only (parse_imgen_alias +
check_alias_consistency). The cmd_doctor printer wrapping these is
on the lower-coverage interactive carve-out per CLAUDE.md.
"""
from __future__ import annotations

from pathlib import Path

from imgen.commands.doctor import check_alias_consistency, parse_imgen_alias


# ── parse_imgen_alias ───────────────────────────────────────────────────


def test_parse_alias_bare_path():
    """The common case: a path with no special chars stays unquoted
    after shlex.quote, so the alias line is `alias imgen=/path/foo`."""
    assert parse_imgen_alias("alias imgen=/Users/foo/imgen/imgen\n") == Path(
        "/Users/foo/imgen/imgen"
    )


def test_parse_alias_single_quoted_path_with_spaces():
    """Paths with spaces get single-quote-wrapped by shlex.quote;
    parser must unwrap to get the bare path back."""
    line = "alias imgen='/Users/Foo Bar/imgen/imgen'\n"
    assert parse_imgen_alias(line) == Path("/Users/Foo Bar/imgen/imgen")


def test_parse_alias_returns_none_when_no_alias_present():
    """rc with other content but no alias imgen= → None."""
    content = "export PATH=$PATH:/usr/local/bin\n# some comment\n"
    assert parse_imgen_alias(content) is None


def test_parse_alias_returns_none_on_empty_content():
    assert parse_imgen_alias("") is None


def test_parse_alias_last_definition_wins():
    """Shell semantics: later `alias` overrides earlier. Parser mirrors
    this so the doctor reports what the user's shell actually uses."""
    content = (
        "alias imgen=/old/path\n"
        "alias imgen=/new/path\n"
    )
    assert parse_imgen_alias(content) == Path("/new/path")


def test_parse_alias_ignores_commented_out_lines():
    """A `#` at line start makes it a comment — the regex's
    `\\s*alias` anchor doesn't match a leading `#`, so commented-out
    aliases are skipped."""
    content = "# alias imgen=/commented/out\n"
    assert parse_imgen_alias(content) is None


def test_parse_alias_strips_trailing_comment():
    """`alias imgen=/path/x  # note` — shlex.split with comments=True
    discards everything after `#`."""
    content = "alias imgen=/Users/foo/imgen/imgen  # set by bootstrap.sh\n"
    assert parse_imgen_alias(content) == Path("/Users/foo/imgen/imgen")


def test_parse_alias_allows_leading_whitespace():
    """Indented alias still matches — some users style their rc files."""
    content = "    alias imgen=/Users/foo/imgen/imgen\n"
    assert parse_imgen_alias(content) == Path("/Users/foo/imgen/imgen")


def test_parse_alias_handles_double_quoted_path():
    """Double-quoting works too (shlex.split unwraps either)."""
    line = 'alias imgen="/Users/foo/imgen/imgen"\n'
    assert parse_imgen_alias(line) == Path("/Users/foo/imgen/imgen")


# ── check_alias_consistency ─────────────────────────────────────────────


def test_check_alias_returns_empty_in_pipx_mode():
    """No IMGEN_HOME → no alias is ever written → nothing to check."""
    assert check_alias_consistency(Path.home(), None) == []


def test_check_alias_returns_empty_when_no_rc_files_exist(tmp_path):
    """Fresh user with no shell rc files → empty list. Not an issue —
    user may not have run setup yet, or may invoke imgen by direct path."""
    imgen_home = tmp_path / "imgen"
    imgen_home.mkdir()
    (imgen_home / "imgen").touch()
    # home = empty tmp dir, no rc files
    assert check_alias_consistency(tmp_path, imgen_home) == []


def test_check_alias_reports_match_when_alias_points_at_imgen_home(tmp_path):
    """Happy path: alias in ~/.zshrc points at IMGEN_HOME/imgen."""
    imgen_home = tmp_path / "imgen"
    imgen_home.mkdir()
    shim = imgen_home / "imgen"
    shim.touch()

    rc = tmp_path / ".zshrc"
    rc.write_text(f"alias imgen={shim}\n")

    results = check_alias_consistency(tmp_path, imgen_home)
    assert len(results) == 1
    rc_path, aliased, status = results[0]
    assert rc_path == rc
    assert status == "match"


def test_check_alias_reports_mismatch_when_alias_points_elsewhere(tmp_path):
    """The architect #1 footgun: user moved the repo after bootstrap.
    Alias still points at /Users/foo/projects/imgen/imgen but
    IMGEN_HOME (where imgen now runs from) is /Users/foo/imgen."""
    imgen_home = tmp_path / "imgen"
    imgen_home.mkdir()
    (imgen_home / "imgen").touch()

    stale_location = tmp_path / "projects" / "imgen"
    stale_location.mkdir(parents=True)
    stale_shim = stale_location / "imgen"
    stale_shim.touch()

    rc = tmp_path / ".zshrc"
    rc.write_text(f"alias imgen={stale_shim}\n")

    results = check_alias_consistency(tmp_path, imgen_home)
    assert len(results) == 1
    rc_path, aliased, status = results[0]
    assert rc_path == rc
    assert status == "mismatch"
    assert aliased == stale_shim


def test_check_alias_reports_per_rc_file_when_multiple_shells_have_alias(
    tmp_path,
):
    """User who's switched shells over time may have stale aliases in
    multiple rc files (e.g. old .bash_profile + current .zshrc).
    Each gets its own entry so the user sees every stale spot."""
    imgen_home = tmp_path / "imgen"
    imgen_home.mkdir()
    correct_shim = imgen_home / "imgen"
    correct_shim.touch()

    stale_location = tmp_path / "old_install"
    stale_location.mkdir()
    stale_shim = stale_location / "imgen"
    stale_shim.touch()

    (tmp_path / ".zshrc").write_text(f"alias imgen={correct_shim}\n")
    (tmp_path / ".bash_profile").write_text(f"alias imgen={stale_shim}\n")

    results = check_alias_consistency(tmp_path, imgen_home)
    assert len(results) == 2
    statuses = {rc.name: status for rc, _, status in results}
    assert statuses[".zshrc"] == "match"
    assert statuses[".bash_profile"] == "mismatch"


def test_check_alias_skips_rc_file_without_imgen_alias(tmp_path):
    """rc file exists but has no `alias imgen=` line → not returned.
    Not an issue worth surfacing — user may scope their alias to one
    shell only."""
    imgen_home = tmp_path / "imgen"
    imgen_home.mkdir()
    (imgen_home / "imgen").touch()

    (tmp_path / ".zshrc").write_text(
        "# zshrc with unrelated content\n"
        "export PATH=$PATH:/usr/local/bin\n"
    )

    assert check_alias_consistency(tmp_path, imgen_home) == []


# ── Single source of truth — setup write list == doctor read list ───────


def test_doctor_reads_exactly_the_files_setup_writes():
    """v0.3.6 architect IMP-1: pre-fix, doctor.py read `.bashrc` (which
    setup.py never wrote to) and missed nothing else. Both modules now
    share `shell_rc.py` so any future shell addition stays symmetric
    by construction. This test locks the symmetry so re-introducing a
    divergence (e.g. doctor adding a "defensive" extra rc target) gets
    caught at test time."""
    from imgen.shell_rc import ALL_RC_FILES_REL, RC_FILE_BY_SHELL
    # The set of files setup.py CAN write to equals the set doctor reads.
    assert set(ALL_RC_FILES_REL) == set(RC_FILE_BY_SHELL.values())
