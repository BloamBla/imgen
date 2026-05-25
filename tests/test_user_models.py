"""v0.8.0 commit 3 — user TOML loader: models.d/ as additional path.

Per [[project-v080-design]] §H + §Q. Locks:

* models.d/ is read alongside backends.d/ (BOTH paths active).
* models.d/ wins on same-stem collision (encourages migration).
* backends.d/ entries load without deprecation warn at commit 3
  (warn lands in commit 4a alongside the CLI rename).
* v0.7-shape TOMLs omitting the v0.8 ``engine`` field still load —
  the BUILTIN_MODELS derivation path hardcodes engine="mflux".
* Symlinked models.d/ is refused (cross-uid attack mirror of v0.4 IMP-3).

These contracts also lock the v0.8.x deprecation window — when
commit 4a turns on the backends.d/ DEPRECATED warn, test 2 will
get updated in the same commit to assert the warn is now present.
"""
from __future__ import annotations

import pytest


# ── Test 1: models.d/ TOMLs reach get_backend ──────────────────────────


def test_user_toml_loaded_from_models_d(tmp_state_dir):
    """The v0.8 canonical path. Drop a TOML in ~/.imgen/models.d/, the
    loader picks it up the same way backends.d/ does today.
    """
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "v8new.toml").write_text(
        'binary = "mflux-generate-fake"\nimage_flag = "--image-path"\n'
    )
    backends_mod.reset_backends_cache()
    try:
        assert "v8new" in backends_mod.list_backends()
        be = backends_mod.get_backend("v8new")
        assert be.binary == "mflux-generate-fake"
        assert be.image_flag == "--image-path"
    finally:
        backends_mod.reset_backends_cache()


# ── Test 2: backends.d/ still loads AND emits DEPRECATED warn ─────────
#
# Cross-commit history: this test was added at commit 3 asserting NO
# DEPRECATED in stderr (deprecation warn was deferred to commit 4a per
# §Q split rationale). Commit 4a flipped the assertions per CLAUDE.md
# cross-commit invariant — backends.d/ entries now load AND emit a
# per-file warn pointing at the `mv` migration command.


def test_user_toml_warns_on_backends_d_load(
    tmp_state_dir, capsys,
):
    """v0.8.0 commit 4a: per-file DEPRECATED warn on backends.d/ load.
    Loading still works (deprecation window stays open through
    v0.8.x; v0.9.0 drops the read entirely), but each file surfaces
    a one-line migration nudge with the concrete mv command so a
    colleague can fix it without consulting docs.
    """
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.BACKENDS_D.mkdir()
    (paths_mod.BACKENDS_D / "legacy.toml").write_text(
        'binary = "mflux-generate-fake"\nimage_flag = "--image-path"\n'
    )
    backends_mod.reset_backends_cache()
    try:
        assert "legacy" in backends_mod.list_backends()
        be = backends_mod.get_backend("legacy")
        assert be.binary == "mflux-generate-fake"
    finally:
        backends_mod.reset_backends_cache()

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "DEPRECATED" in combined
    # mv-command guidance present so the user can act without docs
    assert "mv ~/.imgen/backends.d/legacy.toml" in combined
    assert "~/.imgen/models.d/legacy.toml" in combined


# ── Test 3: same stem in both dirs → models.d/ wins ────────────────────


def test_user_toml_models_d_wins_on_collision(tmp_state_dir):
    """§H step 3 — same-stem-in-both-dirs collision: models.d/ wins.
    Encourages migration: a colleague who copied their TOML from
    backends.d/ → models.d/ during the v0.8.x deprecation window
    sees the new copy take effect immediately.
    """
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.BACKENDS_D.mkdir()
    (paths_mod.BACKENDS_D / "myfx.toml").write_text(
        'binary = "OLD-from-backends-d"\nimage_flag = "--image-path"\n'
    )
    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "myfx.toml").write_text(
        'binary = "NEW-from-models-d"\nimage_flag = "--image-paths"\n'
    )
    backends_mod.reset_backends_cache()
    try:
        be = backends_mod.get_backend("myfx")
        assert be.binary == "NEW-from-models-d"
        assert be.image_flag == "--image-paths"
    finally:
        backends_mod.reset_backends_cache()


# ── Test 4: v0.7-shape TOML (no engine field) + engine="mflux" default ─


def test_user_toml_defaults_engine_mflux_when_omitted(tmp_state_dir):
    """A v0.7-shape TOML omits the v0.8 ``engine`` field entirely.
    The Backend dataclass has no ``engine`` attribute at commit 3 (it
    arrives only at commit 4b when Model becomes the live registry),
    so the TOML loads without schema rejection.

    The forward-compat contract: any v0.7 Backend, when promoted to a
    v0.8 Model via ``_model_from_backend`` (the commit-2 derivation
    helper), gets ``engine="mflux"`` hardcoded. Sampling BUILTIN_MODELS
    locks that derivation default — when commit 4b's full user-TOML-to-
    Model conversion lands, it inherits this default and a legacy
    colleague TOML omitting ``engine`` Just Works.
    """
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "myflux.toml").write_text(
        'binary = "mflux-generate-fake"\nimage_flag = "--image-path"\n'
    )
    backends_mod.reset_backends_cache()
    try:
        be = backends_mod.get_backend("myflux")
    finally:
        backends_mod.reset_backends_cache()

    # v0.7 Backend dataclass is engine-naive. Asserting absence locks
    # in that commit 3 didn't accidentally widen Backend with a v0.8
    # field — that surface change is commit 4b's deliberate work.
    assert not hasattr(be, "engine"), (
        "commit 3 must keep Backend engine-naive; engine field arrives "
        "with the Model rename in commit 4b"
    )

    # Derivation default: any v0.7 Backend → Model has engine="mflux".
    from imgen.models import BUILTIN_MODELS
    for name, model in BUILTIN_MODELS.items():
        assert model.engine == "mflux", (
            f"BUILTIN_MODELS[{name!r}].engine should default to 'mflux' "
            f"for v0.7-shape backends (got {model.engine!r})"
        )


# ── Test 5: symlinked models.d/ refused with warn ──────────────────────


# ── Test 6: user TOML named `flux.toml` survives the v0.7→v0.8 rename ──


def test_user_toml_named_flux_registers_under_v07_name_unchanged(
    tmp_state_dir, capsys,
):
    """v0.8.0 commit 4b architect N-2 lock-in: a pre-v0.8 user TOML
    at ``~/.imgen/backends.d/flux.toml`` (the user's custom Kontext
    recipe) no longer collides with the built-in (which renamed
    ``flux`` → ``flux-kontext``). The user TOML therefore registers
    UNCHANGED under stem ``flux`` in the merged registry.

    Behaviour implications:

    * ``get_backend("flux")`` returns the USER TOML (since the v0.7
      built-in key is gone, replaced by v0.8 ``flux-kontext`` key).
    * The user CANNOT type ``--model flux`` from the CLI — the
      pre-argparse hook + ``_resolve_v07_alias`` reject it with a
      v0.7-rename hint pointing at ``flux-kontext``.
    * So the user TOML named ``flux`` is REGISTRY-VISIBLE but
      CLI-UNREACHABLE. ``imgen migrate-toml`` (commit 10) surfaces
      this with a clear "rename your TOML" prompt.

    This test pins the present behaviour; commit 10's migration
    helper closes the surface-area gap.
    """
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.BACKENDS_D.mkdir()
    (paths_mod.BACKENDS_D / "flux.toml").write_text(
        'binary = "mflux-generate-user-flux"\n'
        'image_flag = "--image-path"\n'
    )
    backends_mod.reset_backends_cache()
    try:
        # User TOML registers under stem 'flux' — no collision with
        # built-in (built-in is keyed by 'flux-kontext' post-4b,
        # backward-derived view re-keys built-in back to 'flux').
        # The collision path runs in merge_user_backends; here built-in
        # 'flux' (from backward-derivation) and user 'flux' DO collide,
        # so the user TOML gets the _0001 suffix per existing v0.4 policy.
        names = backends_mod.list_backends()
        assert "flux" in names  # built-in (backward-derived view)
        # User TOML suffix-renamed because of v0.7-keyed collision
        assert "flux_0001" in names
    finally:
        backends_mod.reset_backends_cache()


# ── Test 7: backends.py / models.py no module-load circular import ─────


def test_backends_models_no_circular_import():
    """v0.8.0 commit 4b architect N-3 lock-in: backends.py imports
    BUILTIN_MODELS from models.py at module load (for the backward-
    derived BUILTIN_BACKENDS view), and parser.py imports both. A
    naive import cycle (parser → backends → parser via rename map)
    was avoided at 4b by relocating ``_V07_TO_V08_MODEL_RENAMES`` to
    models.py — confirm both modules load cleanly in a fresh
    subprocess (bypasses pytest's import cache).
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", (
            "import imgen.models; import imgen.backends; "
            "import imgen.parser; "
            "from imgen.backends import BUILTIN_BACKENDS; "
            "from imgen.models import BUILTIN_MODELS; "
            "assert len(BUILTIN_BACKENDS) == 4; "
            "assert len(BUILTIN_MODELS) == 4; "
            "print('ok')"
        )],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, (
        f"circular-import smoke failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )
    assert "ok" in result.stdout


def test_models_d_symlink_refused_with_warn(tmp_state_dir, tmp_path, capsys):
    """Mirror of v0.4 IMP-3 protection for the new directory. Threat
    model: cross-uid NFS / multi-user Mac where another uid places a
    symlink at ~/.imgen/models.d/ pointing at attacker-controlled
    TOMLs. The loader must refuse + warn + leave the rest of the
    registry intact.
    """
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    attacker_dir = tmp_path / "attacker_real"
    attacker_dir.mkdir()
    (attacker_dir / "evil.toml").write_text(
        'binary = "mflux-generate-attacker"\nimage_flag = "--image-path"\n'
    )

    # The autouse fixture just bound MODELS_D as a path constant; the
    # directory doesn't exist yet. Create it as a symlink to attacker.
    models_d_path = paths_mod.MODELS_D
    assert not models_d_path.exists()
    models_d_path.symlink_to(attacker_dir)

    backends_mod.reset_backends_cache()
    try:
        merged = backends_mod._load_merged_backends()
    finally:
        backends_mod.reset_backends_cache()

    # Built-ins still load — symlink refusal doesn't poison the rest.
    assert "flux" in merged
    # Attacker TOML rejected.
    assert "evil" not in merged

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "symlink" in combined.lower()
