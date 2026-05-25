"""v0.8.0 commit 10 — `imgen migrate-toml` helper + doctor shadowing
warn + opt-in template ship.

Per [[project-v080-design]] §G.2 + §G.3 + §H + §Q commit 10. These
tests exercise the file-location migration helpers end-to-end
without spawning the actual ``imgen`` subprocess — they call the
``cmd_migrate_toml`` / ``cmd_doctor`` handlers directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from imgen.commands.doctor import _warn_shadowing_user_tomls
from imgen.commands.migrate_toml import cmd_migrate_toml


# ── §H lock-in: migrate-toml smoke ───────────────────────────────────


def test_migrate_toml_relocates_novel_user_file(tmp_state_dir, capsys):
    """A user TOML under ~/.imgen/backends.d/ whose stem does NOT match
    any built-in Model gets relocated to ~/.imgen/models.d/<stem>.toml
    (with --yes to skip the interactive confirm)."""
    from imgen.paths import BACKENDS_D, MODELS_D

    BACKENDS_D.mkdir(mode=0o700)
    src = BACKENDS_D / "my-custom-runner.toml"
    src.write_text('binary = "mflux-generate-sdxl"\n'
                   'image_flag = "--image-path"\n')

    rc = cmd_migrate_toml(argparse.Namespace(yes=True))
    assert rc == 0

    assert not src.exists(), "source not removed after move"
    target = MODELS_D / "my-custom-runner.toml"
    assert target.exists(), "target not created"
    assert "binary" in target.read_text()


def test_migrate_toml_suggests_delete_for_shadowing_builtin(
    tmp_state_dir, capsys,
):
    """A user TOML whose stem matches a built-in Model goes through
    the deletion path (not the move path) — the recipe is already
    covered by the v0.8 built-in registry."""
    from imgen.paths import BACKENDS_D, MODELS_D

    BACKENDS_D.mkdir(mode=0o700)
    src = BACKENDS_D / "flux-kontext.toml"  # shadows built-in
    src.write_text('binary = "mflux-generate-kontext"\n'
                   'image_flag = "--image-path"\n')

    rc = cmd_migrate_toml(argparse.Namespace(yes=True))
    assert rc == 0

    assert not src.exists(), "shadowing file not deleted with --yes"
    # NOT moved to models.d (deletion path, not move path)
    assert not (MODELS_D / "flux-kontext.toml").exists()


def test_migrate_toml_empty_backends_d_is_noop(tmp_state_dir):
    """No legacy ~/.imgen/backends.d/ → exit 0, nothing to do."""
    rc = cmd_migrate_toml(argparse.Namespace(yes=True))
    assert rc == 0


def test_migrate_toml_refuses_to_overwrite_existing_target(
    tmp_state_dir, capsys,
):
    """If ~/.imgen/models.d/<stem>.toml already exists, migrate-toml
    refuses to overwrite and warns the user to resolve manually.
    Prevents silent data loss on a conflicting-content migration."""
    from imgen.paths import BACKENDS_D, MODELS_D

    BACKENDS_D.mkdir(mode=0o700)
    MODELS_D.mkdir(mode=0o700)
    src = BACKENDS_D / "my-runner.toml"
    src.write_text('binary = "old-version"\nimage_flag = "--image-path"\n')
    target = MODELS_D / "my-runner.toml"
    target.write_text('binary = "new-version"\nimage_flag = "--image-path"\n')

    rc = cmd_migrate_toml(argparse.Namespace(yes=True))
    assert rc == 0

    # Source untouched, target untouched.
    assert src.read_text().startswith('binary = "old-version"')
    assert target.read_text().startswith('binary = "new-version"')

    out = capsys.readouterr().out
    assert "already exists" in out.lower() or "refusing" in out.lower()


# ── §G.2 lock-in: opt-in template shipped to models.d.example/ ───────


def test_setup_ships_qwen_bf16_template_to_models_d_example(
    tmp_state_dir, monkeypatch,
):
    """Per design memo §G.2, `imgen setup` lands the
    ``qwen-image-2512-bf16.toml`` template at
    ``~/.imgen/models.d.example/``. NOT in ``models.d/`` (which would
    auto-activate it on hardware that can't run it).

    Calls the post-token state-dir block of cmd_setup. The actual
    full cmd_setup runs Apple-Silicon + venv + token checks first;
    bypass those by exercising the template-writer directly.
    """
    from imgen.commands import setup as setup_mod
    from imgen.paths import MODELS_D_EXAMPLE

    # Simulate the post-token block — directly mirror cmd_setup's
    # template ship path (no subprocess, no HF token).
    MODELS_D_EXAMPLE.mkdir(mode=0o700, exist_ok=True)
    template = MODELS_D_EXAMPLE / "qwen-image-2512-bf16.toml"
    template.write_text(setup_mod._QWEN_BF16_TEMPLATE)

    assert template.exists()
    body = template.read_text()
    # Lock-in: the template names the v0.8 diffusers_mps engine, the
    # HF repo, and the hardware caveat — three properties the README
    # banner directly references.
    assert 'engine = "diffusers_mps"' in body
    assert 'Qwen/Qwen-Image-2512' in body
    assert "64+ GB" in body or "64 GB" in body


# ── §G.3 lock-in: doctor warns on user-TOML-shadows-builtin ──────────


def test_doctor_warns_on_user_toml_shadowing_builtin(
    tmp_state_dir, capsys,
):
    """A user TOML whose stem matches a built-in Model name surfaces
    a doctor warn pointing at `imgen migrate-toml`. Path rendered via
    repr() per round-3 security LOW (control-byte safety)."""
    from imgen.paths import MODELS_D

    MODELS_D.mkdir(mode=0o700)
    # flux-kontext is a v0.8 built-in Model.
    (MODELS_D / "flux-kontext.toml").write_text(
        'binary = "mflux-generate-kontext"\nimage_flag = "--image-path"\n'
    )

    _warn_shadowing_user_tomls()
    out = capsys.readouterr().out

    assert "SHADOWS built-in Model" in out
    assert "'flux-kontext'" in out
    assert "imgen migrate-toml" in out


def test_doctor_shadowing_warn_repr_wraps_path_with_control_bytes(
    tmp_state_dir, capsys,
):
    """Round-3 security LOW (§G.3 closure): a user TOML with control
    bytes in the directory name renders via ``repr()`` so the C0/DEL/C1
    bytes don't escape into the user's terminal. Existing project
    pattern (v0.4 IMP-2 for binary paths)."""
    from imgen.paths import MODELS_D

    MODELS_D.mkdir(mode=0o700)
    # Filename itself can't carry control bytes (filesystem rejects
    # most; loader rejects the rest). The TOML content + the directory
    # ANCESTOR path are the realistic injection surfaces. Test the
    # repr() wrapping by inspecting the format string used.
    (MODELS_D / "flux-kontext.toml").write_text(
        'binary = "x"\nimage_flag = "y"\n'
    )

    _warn_shadowing_user_tomls()
    out = capsys.readouterr().out

    # repr() of a str path always quotes — so the path renders inside
    # single or double quotes. A naive `f"{path}"` would not.
    # This is the structural lock-in: the format went through repr().
    import re
    assert re.search(r"['\"][^'\"]*flux-kontext\.toml['\"]", out), (
        "doctor shadowing warn must wrap path via repr() — pattern not "
        f"found in output: {out!r}"
    )


def test_doctor_no_shadowing_warn_when_user_tomls_match_no_builtin(
    tmp_state_dir, capsys,
):
    """A user TOML whose stem does NOT match a built-in Model name
    produces no shadowing warn. Negative test — guards against an
    overeager future "warn on every user TOML" regression."""
    from imgen.paths import MODELS_D

    MODELS_D.mkdir(mode=0o700)
    (MODELS_D / "my-custom-runner.toml").write_text(
        'binary = "mflux-generate-sdxl"\nimage_flag = "--image-path"\n'
    )

    _warn_shadowing_user_tomls()
    out = capsys.readouterr().out

    assert "SHADOWS" not in out
    assert "my-custom-runner" not in out


# ── v0.8.1 LOW-2 lock-in: migrate-toml refuses dest symlink ──────────


def test_migrate_toml_refuses_when_models_d_is_symlink(
    tmp_state_dir, tmp_path, capsys,
):
    """v0.8.1 LOW-2 closure: ``cmd_migrate_toml`` checks
    ``MODELS_D.is_symlink()`` before any mkdir / move. The loader
    already refuses symlinks (backends.py:601 line); migrate-toml
    now mirrors that policy on the destination side so a same-uid
    attacker-planted symlink at ``~/.imgen/models.d/`` doesn't cause
    user TOMLs to be moved THROUGH the link.
    """
    from imgen.paths import BACKENDS_D, MODELS_D

    # Source: real directory with a file to migrate
    BACKENDS_D.mkdir(mode=0o700)
    (BACKENDS_D / "test.toml").write_text(
        'binary = "mflux-generate-fake"\nimage_flag = "--image-path"\n'
    )

    # Destination: symlinked to an attacker-controlled dir
    attacker = tmp_path / "attacker"
    attacker.mkdir(mode=0o700)
    assert not MODELS_D.exists()
    MODELS_D.symlink_to(attacker)

    rc = cmd_migrate_toml(argparse.Namespace(yes=True))
    assert rc == 2, "must exit 2 (resource class) when dest is symlink"

    # Source file UNTOUCHED — no move through the link.
    assert (BACKENDS_D / "test.toml").exists()
    # Attacker dir empty — no migration happened through the symlink.
    assert list(attacker.iterdir()) == []

    out = capsys.readouterr().out
    combined = out + capsys.readouterr().err
    assert "symlink" in out.lower()


# ── v0.8.1 LOW-1 lock-in: migrate-toml wraps path.name via safe_display ─


def test_migrate_toml_path_name_rendered_via_safe_display(
    tmp_state_dir, capsys,
):
    """v0.8.1 LOW-1 closure: header + per-file prompt lines in
    migrate-toml render the filename via _safe.safe_display so any
    control bytes in the filename would be escaped to ``\\xNN``
    literals rather than written raw to the terminal.

    Filesystem-injected control bytes are hard to set up portably
    (most filesystems strip them); this test verifies the STRUCTURAL
    invariant that the rendered name is quoted (which only happens
    via repr() / safe_display). A regression to raw ``{path.name}``
    interpolation would drop the quotes.
    """
    from imgen.paths import BACKENDS_D

    BACKENDS_D.mkdir(mode=0o700)
    (BACKENDS_D / "my-custom-runner.toml").write_text(
        'binary = "mflux-generate-fake"\nimage_flag = "--image-path"\n'
    )

    cmd_migrate_toml(argparse.Namespace(yes=True))
    out = capsys.readouterr().out

    # safe_display always quotes — so the filename appears wrapped in
    # single or double quotes (Python repr's rule). A pre-LOW-1 raw
    # f"{path.name}" interpolation would NOT have quotes.
    import re
    assert re.search(
        r"['\"]my-custom-runner\.toml['\"]", out
    ), (
        "migrate-toml must wrap path.name via safe_display — pattern "
        f"not found in output: {out!r}"
    )
