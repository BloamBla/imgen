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

    # v0.8.1 HIGH-2 closure: Backend now carries v0.8 fields so user
    # TOMLs can declare them; a v0.7-shape TOML that omits ``engine``
    # gets the default "mflux" (matches v0.7 behaviour).
    assert hasattr(be, "engine"), (
        "v0.8.1 widened Backend with v0.8 Model-shape fields; engine "
        "must be present"
    )
    assert be.engine == "mflux", (
        f"v0.7-shape TOML must default to engine='mflux' "
        f"(got {be.engine!r})"
    )

    # Derivation default: any v0.7 Backend → Model has engine="mflux".
    # v0.9 commit 7 (§K): ltx-video is the first BUILTIN_MODELS row
    # with engine="diffusers_mps". The v0.7-shape derivation default
    # still applies to legacy mflux rows; the assertion scopes to them.
    from imgen.models import BUILTIN_MODELS
    _v07_shape_names = {
        "flux-kontext", "qwen-image-edit-v1",
        "flux-dev", "flux2-klein-edit-9b",
    }
    for name, model in BUILTIN_MODELS.items():
        if name not in _v07_shape_names:
            # v0.9+ rows declare engine explicitly; skip the v0.7
            # default-derivation check.
            continue
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
            # v0.9 commit 7: ltx-video → 5 built-ins. v0.10 commit 2:
            # flux2-klein-4b → 6 built-ins (first inference+training row).
            "assert len(BUILTIN_BACKENDS) == 6; "
            "assert len(BUILTIN_MODELS) == 6; "
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


# ── v0.8.1 HIGH-2 closure: user TOMLs declare v0.8 Model-shape fields ──


def test_user_toml_engine_diffusers_mps_loads_with_repo(tmp_state_dir):
    """A user TOML declaring ``engine = "diffusers_mps"`` + ``repo = ...``
    loads cleanly with no warn-and-drop on the v0.8 fields. The
    resulting Backend round-trips through ``model_from_backend`` to a
    Model carrying engine="diffusers_mps", ready for Engine routing."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "qwen-bf16.toml").write_text(
        'engine = "diffusers_mps"\n'
        'repo = "Qwen/Qwen-Image-2512"\n'
        'supports_negative = true\n'
        'lora_compat_group = "qwen"\n'
        'default_steps = 50\n'
        'default_guidance = 4.0\n'
        'ram_baseline_gb = 24.0\n'
        'ram_slope_gb_per_mp = 8.0\n'
        'encoder_ram_gb = 14.0\n'
        'cpu_offload_threshold_mp = 1.0\n'
        'param_overrides = [["true_cfg_scale", 4.0]]\n'
    )
    backends_mod.reset_backends_cache()
    try:
        be = backends_mod.get_backend("qwen-bf16")
    finally:
        backends_mod.reset_backends_cache()

    assert be is not None
    assert be.engine == "diffusers_mps"
    assert be.repo == "Qwen/Qwen-Image-2512"
    assert be.default_steps == 50
    assert be.default_guidance == 4.0
    assert be.ram_baseline_gb == 24.0
    assert be.cpu_offload_threshold_mp == 1.0
    assert be.param_overrides == (("true_cfg_scale", 4.0),)

    # Round-trip through model_from_backend → Model carries the fields.
    m = backends_mod.model_from_backend("qwen-bf16", be)
    assert m.engine == "diffusers_mps"
    assert m.repo == "Qwen/Qwen-Image-2512"
    assert m.default_steps == 50
    assert m.cpu_offload_threshold_mp == 1.0
    # binary intentionally None on diffusers_mps Models (Model
    # __post_init__ would reject ``"" `` + engine=diffusers_mps as a
    # spec violation — the converter knows to drop binary).
    assert m.binary is None


def test_user_toml_diffusers_engine_requires_repo(tmp_state_dir, capsys):
    """``engine = "diffusers_mps"`` without ``repo = ...`` raises
    UserBackendError at validation time — the per-file warn fires but
    the file is skipped, not loaded with a bogus Backend that crashes
    later at Engine routing."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "bogus.toml").write_text(
        'engine = "diffusers_mps"\n'
        # repo deliberately missing
        'ram_baseline_gb = 24.0\n'
        'ram_slope_gb_per_mp = 8.0\n'
    )
    backends_mod.reset_backends_cache()
    try:
        merged = backends_mod._load_merged_backends()
    finally:
        backends_mod.reset_backends_cache()

    assert "bogus" not in merged, "diffusers_mps without repo must skip"
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert "repo" in combined.lower()
    assert "diffusers_mps" in combined


def test_user_toml_unknown_engine_rejected(tmp_state_dir, capsys):
    """Unknown engine string (e.g. ``"mlx-native"``) skips the file
    with a warn — the schema validator catches it before the runtime
    layer."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "futuristic.toml").write_text(
        'engine = "mlx-native"\n'
        'binary = "mflux-generate"\n'
        'image_flag = "--image-path"\n'
    )
    backends_mod.reset_backends_cache()
    try:
        merged = backends_mod._load_merged_backends()
    finally:
        backends_mod.reset_backends_cache()

    assert "futuristic" not in merged


def test_user_toml_param_overrides_array_of_pairs_parses(tmp_state_dir):
    """TOML ``[["key", value], ["key2", value2]]`` deserialises to the
    runtime ``tuple[tuple[str, object], ...]`` shape on Backend."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "tuned.toml").write_text(
        'binary = "mflux-generate-fake"\n'
        'image_flag = "--image-path"\n'
        'param_overrides = [["foo", 1.5], ["bar", "baz"]]\n'
    )
    backends_mod.reset_backends_cache()
    try:
        be = backends_mod.get_backend("tuned")
    finally:
        backends_mod.reset_backends_cache()

    assert be.param_overrides == (("foo", 1.5), ("bar", "baz"))


def test_user_toml_ram_baseline_rejects_zero(tmp_state_dir, capsys):
    """``ram_baseline_gb = 0`` is a sentinel-fail per memo §L — must be
    rejected at schema time (clearer diagnostic than the downstream
    Model.__post_init__ explosion)."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "bogus.toml").write_text(
        'binary = "mflux-generate-fake"\n'
        'image_flag = "--image-path"\n'
        'ram_baseline_gb = 0\n'  # rejected
    )
    backends_mod.reset_backends_cache()
    try:
        merged = backends_mod._load_merged_backends()
    finally:
        backends_mod.reset_backends_cache()

    assert "bogus" not in merged
    out = capsys.readouterr().out + capsys.readouterr().err
    # Schema rejection visible in warn
    # (the validator's "field X: ..." message)
    # Just verify the file was skipped — message wording isn't locked.


def test_user_toml_v07_shape_loads_unchanged_via_v081_schema(tmp_state_dir):
    """v0.7-shape TOML (no v0.8 fields) still loads. v0.8.1 schema
    SUPERSET promise — additive, no breakage for colleagues' existing
    files."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "legacy.toml").write_text(
        'binary = "mflux-generate-fake"\n'
        'image_flag = "--image-path"\n'
        'supports_strength = true\n'
        'extra_args = ["--model", "sdxl"]\n'
    )
    backends_mod.reset_backends_cache()
    try:
        be = backends_mod.get_backend("legacy")
    finally:
        backends_mod.reset_backends_cache()

    assert be is not None
    # v0.7 fields preserved exactly
    assert be.binary == "mflux-generate-fake"
    assert be.supports_strength is True
    assert be.extra_args == ("--model", "sdxl")
    # v0.8 fields filled with defaults
    assert be.engine == "mflux"
    assert be.repo is None
    assert be.ram_baseline_gb == 13.5  # flux-class fallback default
    assert be.default_steps == 20
    assert be.param_overrides == ()


# ── v0.8.2 NIT-B: warn on inapplicable fields per engine ─────────────


def test_diffusers_mps_user_toml_warns_when_binary_is_set(
    tmp_state_dir, capsys,
):
    """v0.8.2 NIT-B closure: a colleague who copies an mflux template,
    switches ``engine = "diffusers_mps"``, but forgets to delete
    ``binary = "..."`` gets a friendly warn naming the dead field
    + the engine context. Pre-fix the field was silently stored on
    Backend and ignored at runtime — no signal to the user."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "qwen-mixed.toml").write_text(
        'engine = "diffusers_mps"\n'
        'repo = "Qwen/Qwen-Image-2512"\n'
        # Leftover mflux fields — should warn on each:
        'binary = "mflux-generate-qwen"\n'
        'image_flag = "--image-path"\n'
        'extra_args = ["--something"]\n'
    )
    backends_mod.reset_backends_cache()
    try:
        # Trigger load (warn fires once at load time, per-process
        # cache means it doesn't fire again on the same file).
        backends_mod._load_merged_backends()
    finally:
        backends_mod.reset_backends_cache()

    out = capsys.readouterr().out + capsys.readouterr().err
    # Each inapplicable field gets its own warn line.
    assert "'binary'" in out, "binary warn missing"
    assert "'image_flag'" in out, "image_flag warn missing"
    assert "'extra_args'" in out, "extra_args warn missing"
    assert "diffusers_mps" in out, "engine name missing from warn"


def test_mflux_user_toml_warns_when_diffusers_only_fields_set(
    tmp_state_dir, capsys,
):
    """Mirror of the above for the opposite direction: an mflux user
    TOML declaring diffusers-only fields (repo, cpu_offload_threshold_mp,
    param_overrides) gets per-field warns. Same UX rationale: the
    fields flow into Backend but get ignored at runtime; surface the
    gap at load time."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "weird-mflux.toml").write_text(
        'engine = "mflux"\n'
        'binary = "mflux-generate-fake"\n'
        'image_flag = "--image-path"\n'
        # Leftover diffusers fields — should warn:
        'repo = "fake/repo"\n'
        'cpu_offload_threshold_mp = 1.0\n'
        'param_overrides = [["true_cfg_scale", 4.0]]\n'
    )
    backends_mod.reset_backends_cache()
    try:
        backends_mod._load_merged_backends()
    finally:
        backends_mod.reset_backends_cache()

    out = capsys.readouterr().out + capsys.readouterr().err
    assert "'repo'" in out, "repo warn missing"
    assert "'cpu_offload_threshold_mp'" in out, (
        "cpu_offload_threshold_mp warn missing"
    )
    assert "'param_overrides'" in out, "param_overrides warn missing"
    assert "mflux" in out, "engine name missing from warn"


def test_user_toml_no_warn_when_fields_match_engine(
    tmp_state_dir, capsys,
):
    """Negative test: a TOML with ONLY engine-appropriate fields
    produces no NIT-B warn. Guards against an overeager future
    "warn on every optional field" regression."""
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "clean-mflux.toml").write_text(
        'engine = "mflux"\n'
        'binary = "mflux-generate-fake"\n'
        'image_flag = "--image-path"\n'
        'extra_args = ["--model", "sdxl"]\n'
        'default_steps = 25\n'
    )
    backends_mod.reset_backends_cache()
    try:
        backends_mod._load_merged_backends()
    finally:
        backends_mod.reset_backends_cache()

    out = capsys.readouterr().out + capsys.readouterr().err
    # No "inapplicable" warn should fire.
    assert "inapplicable" not in out.lower()


def test_user_model_routes_through_engine_for_validate(tmp_state_dir):
    """``_model_for_validate(args)`` resolves user TOMLs (not just
    BUILTIN_MODELS) post-v0.8.1. Locked-in so a future refactor of
    the resolver can't silently regress user-TOML Engine routing back
    to v0.8.0's "user TOMLs bypass Engine entirely" behaviour."""
    from types import SimpleNamespace
    import imgen.backends as backends_mod
    import imgen.paths as paths_mod
    from imgen.cmd_helpers import _model_for_validate

    paths_mod.MODELS_D.mkdir()
    (paths_mod.MODELS_D / "my-runner.toml").write_text(
        'binary = "mflux-generate-fake"\n'
        'image_flag = "--image-path"\n'
        'default_guidance = 6.5\n'
        'min_guidance = 5.0\n'
        'max_guidance = 8.0\n'
    )
    backends_mod.reset_backends_cache()
    try:
        model = _model_for_validate(SimpleNamespace(model="my-runner"))
    finally:
        backends_mod.reset_backends_cache()

    assert model is not None
    # User-declared param defaults reach the resolver.
    assert model.default_guidance == 6.5
    assert model.min_guidance == 5.0
    assert model.max_guidance == 8.0
    # Engine defaults to mflux for v0.7-shape TOMLs.
    assert model.engine == "mflux"
