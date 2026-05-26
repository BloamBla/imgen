"""v0.8.0 commit 6 — DiffusersMpsEngine + _diffusers_runner lock-ins.

Per [[project-v080-design]] §E.1 + §Q commit 6. The locked
security-critical pattern is:

* STATIC argv — no user data interpolated into the command line.
* JSON-on-stdin transport — bounded read (65536 bytes), strict
  schema validation, deny-by-default for unknown top-level keys.
* Path resolution via IMGEN_INSTALL_ROOT, never cwd.
* HF-token redaction via the existing stderr-redaction wrapper
  (extended at commit 6 with a stdin_data kwarg).

Each test in this file locks one boundary of the security contract.
Reviewer findings folded in (architect commit-6 pre-vet C1–C3,
H1–H4, M1–M4; security-reviewer pre-vet CRITICAL + HIGH on payload
validation).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_diffusers_model(**overrides):
    """Build a Model row with engine='diffusers_mps' for unit tests.
    Built-in BUILTIN_MODELS has zero diffusers rows at commit 6, so
    the engine has no production caller — test fixtures construct
    Models directly per architect H3 (skip moving the qwen-bf16
    template forward; commit 10 owns it).
    """
    from imgen.models import Model
    defaults = dict(
        engine="diffusers_mps",
        repo="mlx-community/Qwen-Image-2512-4bit",
        cpu_offload_threshold_mp=2.0,
        ram_baseline_gb=10.0,
        ram_slope_gb_per_mp=5.0,
        encoder_ram_gb=7.0,
        param_overrides=(("true_cfg_scale", 4.0),),
    )
    defaults.update(overrides)
    return Model(**defaults)


def _make_genparams(**overrides):
    from imgen.engines.base import GenParams
    defaults = dict(
        prompt="a samurai on a misty mountain",
        negative="",
        width=1024,
        height=1024,
        steps=50,
        guidance=4.0,
        seed=42,
        quantize=4,
        strength=0.0,
        input_path=None,
        output_path=Path("/tmp/out.png"),
        loras=(),
    )
    defaults.update(overrides)
    return GenParams(**defaults)


# ── §Q lock-in 1: static runner argv shape ─────────────────────────────


class TestDiffusersMpsEngineStaticArgv:
    """Security-critical: argv must be STATIC and use `-m` invocation
    of the runner module. NEVER `-c "<script>"`, NEVER format strings
    of user data. Architect pre-vet N3: assert "-c" not in argv as a
    belt-and-suspenders check against future refactors."""

    def _capture_run_with_stderr_redaction(
        self, monkeypatch, tmp_path,
    ):
        """Mock IMGEN_INSTALL_ROOT to a tmp dir with a fake venv layout
        so DiffusersMpsEngine.run() finds a venv_python file (else it
        dies with the 'missing diffusers venv' hint, which is its own
        test below). Capture the argv + stdin_data that
        run_with_stderr_redaction is called with."""
        from imgen import paths
        from imgen.engines import diffusers_mps_engine
        from imgen.engines import (
            diffusers_mps_engine as engine_mod,
        )

        install_root = tmp_path / "install_root"
        venv_python = install_root / ".venv-diffusers" / "bin" / "python"
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("#!/usr/bin/env python3\n")
        venv_python.chmod(0o755)

        monkeypatch.setattr(paths, "IMGEN_INSTALL_ROOT", install_root)

        captured = {}

        def fake_run_with_stderr_redaction(*args, **kwargs):
            # Support both positional + keyword shapes
            if args:
                captured["argv"] = args[0]
                captured["env"] = args[1] if len(args) > 1 else None
            else:
                captured["argv"] = kwargs.get("argv") or kwargs.get("cmd")
                captured["env"] = kwargs.get("env")
            captured["stdin_data"] = kwargs.get("stdin_data")
            return 0

        from imgen import subprocess_helpers
        monkeypatch.setattr(
            subprocess_helpers,
            "run_with_stderr_redaction",
            fake_run_with_stderr_redaction,
        )
        return captured, venv_python

    def test_diffusers_mps_run_uses_static_runner_module(
        self, monkeypatch, tmp_path,
    ):
        """argv MUST end with `-m imgen.engines._diffusers_runner`.
        NEVER `-c "<dynamic-string>"`."""
        from imgen.engines import DiffusersMpsEngine

        captured, venv_python = self._capture_run_with_stderr_redaction(
            monkeypatch, tmp_path,
        )

        engine = DiffusersMpsEngine()
        rc = engine.run(_make_diffusers_model(), _make_genparams())
        assert rc == 0

        argv = captured["argv"]
        assert argv == [
            str(venv_python),
            "-m",
            "imgen.engines._diffusers_runner",
        ]
        # Architect pre-vet N3: belt-and-suspenders against future
        # refactors slipping in -c "<script>".
        assert "-c" not in argv

    def test_diffusers_mps_payload_serializes_via_stdin_json(
        self, monkeypatch, tmp_path,
    ):
        """stdin_data is JSON-encoded bytes; payload contains the
        resolved GenParams + Model fields the runner needs."""
        from imgen.engines import DiffusersMpsEngine

        captured, _ = self._capture_run_with_stderr_redaction(
            monkeypatch, tmp_path,
        )

        engine = DiffusersMpsEngine()
        engine.run(
            _make_diffusers_model(),
            _make_genparams(
                prompt="hello world", width=512, height=512, seed=99,
            ),
        )

        stdin_data = captured["stdin_data"]
        assert isinstance(stdin_data, bytes)
        payload = json.loads(stdin_data.decode("utf-8"))
        assert payload["prompt"] == "hello world"
        assert payload["width"] == 512
        assert payload["height"] == 512
        assert payload["seed"] == 99
        assert payload["repo"] == "mlx-community/Qwen-Image-2512-4bit"
        # param_overrides converted from tuple-of-tuples to dict
        assert payload["param_overrides"] == {"true_cfg_scale": 4.0}

    def test_diffusers_mps_routes_through_redaction_wrapper(
        self, monkeypatch, tmp_path,
    ):
        """The stderr-redaction wrapper is mandatory — diffusers'
        from_pretrained 401/403 includes auth headers in tracebacks."""
        from imgen.engines import DiffusersMpsEngine

        captured, _ = self._capture_run_with_stderr_redaction(
            monkeypatch, tmp_path,
        )
        engine = DiffusersMpsEngine()
        engine.run(_make_diffusers_model(), _make_genparams())
        # The captured dict is populated only when the wrapper is
        # called — so non-empty captured proves routing.
        assert captured.get("argv") is not None


# ── §Q lock-in 2: install-root path resolution (NOT cwd) ───────────────


def test_diffusers_mps_path_resolves_from_install_root_not_cwd(
    monkeypatch, tmp_path,
):
    """Security pre-vet C1 + memo §E.1: venv_python MUST resolve from
    ``IMGEN_INSTALL_ROOT``, never cwd. A ``cd /tmp/attacker && imgen
    draw ...`` invocation cannot exec a planted python via
    .venv-diffusers/bin/python.
    """
    from imgen import paths
    from imgen.engines import DiffusersMpsEngine

    # Set IMGEN_INSTALL_ROOT to a clean tmp dir; do NOT create the
    # .venv-diffusers/ inside it.
    install_root = tmp_path / "install_root"
    install_root.mkdir()
    monkeypatch.setattr(paths, "IMGEN_INSTALL_ROOT", install_root)

    # Plant a malicious .venv-diffusers in cwd. The engine MUST
    # ignore it.
    attacker_cwd = tmp_path / "attacker_cwd"
    attacker_cwd.mkdir()
    planted = attacker_cwd / ".venv-diffusers" / "bin" / "python"
    planted.parent.mkdir(parents=True)
    planted.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    planted.chmod(0o755)
    monkeypatch.chdir(attacker_cwd)

    engine = DiffusersMpsEngine()
    # No venv at IMGEN_INSTALL_ROOT/.venv-diffusers/ → die with
    # friendly hint. The planted attacker venv in cwd is irrelevant.
    with pytest.raises(SystemExit):
        engine.run(_make_diffusers_model(), _make_genparams())


# ── §Q lock-in 3: friendly missing-venv error ──────────────────────────


def test_diffusers_mps_dies_friendly_when_venv_missing(
    monkeypatch, tmp_path, capsys,
):
    """No .venv-diffusers/ → SystemExit with a hint pointing at
    bootstrap.sh and the IMGEN_INSTALL_DIFFUSERS=1 env override
    (architect pre-vet H4). Friendly, not a bare FileNotFoundError."""
    from imgen import paths
    from imgen.engines import DiffusersMpsEngine

    install_root = tmp_path / "install_root_no_diffusers"
    install_root.mkdir()
    monkeypatch.setattr(paths, "IMGEN_INSTALL_ROOT", install_root)

    engine = DiffusersMpsEngine()
    with pytest.raises(SystemExit):
        engine.run(_make_diffusers_model(), _make_genparams())

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert ".venv-diffusers" in combined
    assert "bootstrap.sh" in combined
    assert "IMGEN_INSTALL_DIFFUSERS=1" in combined


def test_diffusers_mps_refuses_symlinked_venv_python(
    monkeypatch, tmp_path, capsys,
):
    """v0.9 commit 7.1 (§R.2 security HIGH-1): a planted symlink at
    .venv-diffusers/bin/python must be refused BEFORE subprocess
    exec. Mirrors the install-path symlink guard in
    ``ensure_video_deps_or_die`` (commit 6 closed §R.1 HIGH-2 for
    the install path; this closes the execution path which was
    missed at the same time).
    """
    from imgen import paths
    from imgen.engines import DiffusersMpsEngine

    install_root = tmp_path / "install_root_planted"
    venv_bin = install_root / ".venv-diffusers" / "bin"
    venv_bin.mkdir(parents=True)
    target = tmp_path / "evil_python"
    target.write_text("#!/usr/bin/env python3\n")
    target.chmod(0o755)
    (venv_bin / "python").symlink_to(target)
    monkeypatch.setattr(paths, "IMGEN_INSTALL_ROOT", install_root)

    engine = DiffusersMpsEngine()
    with pytest.raises(SystemExit):
        engine.run(_make_diffusers_model(), _make_genparams())

    stderr = capsys.readouterr().err
    assert "symlink" in stderr.lower(), (
        f"refuse message should mention symlink; got: {stderr!r}"
    )


# ── Static runner payload validation (architect C2 + security CRITICAL+HIGH+M3) ─


class TestDiffusersRunnerPayloadValidation:
    """The runner is the LAST trust boundary before user data hits
    diffusers internals. Strict schema check on every payload field.
    Architect commit-6 pre-vet C2 + security HIGH + M3.
    """

    def _run_runner_with_payload(
        self, payload_bytes: bytes,
    ) -> subprocess.CompletedProcess:
        """Invoke the runner via subprocess so module-load
        side-effects (e.g. PYTORCH_ENABLE_MPS_FALLBACK env-set) are
        observable. Stdin = the payload bytes; stdout/stderr
        captured."""
        return subprocess.run(
            [sys.executable, "-m", "imgen.engines._diffusers_runner"],
            input=payload_bytes,
            capture_output=True,
            timeout=10,
        )

    def test_diffusers_runner_rejects_oversize_stdin(self):
        """Architect H1 + memo §E.1 round-2 security HIGH: payloads
        over 65536 bytes are rejected BEFORE json.loads runs (DoS
        guard)."""
        oversize = b"a" * (65_536 + 1)
        result = self._run_runner_with_payload(oversize)
        assert result.returncode == 64  # EX_USAGE
        assert b"exceeded" in result.stderr

    def test_diffusers_runner_rejects_malformed_json(self):
        """Non-JSON stdin → EX_USAGE before any pipeline import."""
        result = self._run_runner_with_payload(b"{not json")
        assert result.returncode == 64
        assert b"not valid JSON" in result.stderr

    def test_diffusers_runner_rejects_non_object_payload(self):
        """Payload root MUST be a JSON object, not a list/string/int."""
        result = self._run_runner_with_payload(b'["array", "payload"]')
        assert result.returncode == 64
        assert b"JSON object" in result.stderr

    def test_diffusers_runner_rejects_payload_with_unknown_keys(self):
        """Security pre-vet M3 + N2: deny-by-default for unknown
        top-level keys. Future parent-side bugs that accidentally add
        e.g. ``callback_on_step_end`` get caught here.
        """
        payload = {
            "repo": "org/model",
            "prompt": "hello",
            "negative": "",
            "steps": 20,
            "guidance": 4.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "output_path": "/tmp/out.png",
            "callback_on_step_end": "evil",  # unknown key
        }
        result = self._run_runner_with_payload(
            json.dumps(payload).encode("utf-8"),
        )
        assert result.returncode == 64
        assert b"unknown top-level keys" in result.stderr

    def test_diffusers_runner_rejects_repo_with_pathlike_chars(self):
        """Security HIGH: repo regex `^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$`
        rejects path-like values (``../etc/passwd``, ``/abs/path``).
        from_pretrained must never see attacker-controlled paths."""
        for bad_repo in [
            "../etc/passwd",
            "/etc/passwd",
            "..",
            "./local",
            "org name/spaces",
            "",
            "no-slash",
        ]:
            payload = {
                "repo": bad_repo,
                "prompt": "x",
                "negative": "",
                "steps": 20,
                "guidance": 4.0,
                "width": 1024,
                "height": 1024,
                "seed": 42,
                "output_path": "/tmp/out.png",
            }
            result = self._run_runner_with_payload(
                json.dumps(payload).encode("utf-8"),
            )
            assert result.returncode == 64, (
                f"repo={bad_repo!r} should reject"
            )
            assert b"repo" in result.stderr or b"unknown" in result.stderr

    def test_diffusers_runner_rejects_relative_output_path(self):
        """output_path MUST be absolute — defense-in-depth against
        cwd-relative writes."""
        payload = {
            "repo": "org/model",
            "prompt": "x",
            "negative": "",
            "steps": 20,
            "guidance": 4.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "output_path": "out.png",  # relative
        }
        result = self._run_runner_with_payload(
            json.dumps(payload).encode("utf-8"),
        )
        assert result.returncode == 64
        assert b"output_path" in result.stderr

    def test_diffusers_runner_rejects_unsafe_output_extension(self):
        """output_path extension MUST be in {.png .jpg .jpeg .webp}.
        Writing to /tmp/x.sh or /tmp/x.dylib would be a real risk if
        the parent or registry got compromised."""
        payload = {
            "repo": "org/model",
            "prompt": "x",
            "negative": "",
            "steps": 20,
            "guidance": 4.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "output_path": "/tmp/evil.sh",  # unsafe ext
        }
        result = self._run_runner_with_payload(
            json.dumps(payload).encode("utf-8"),
        )
        assert result.returncode == 64
        assert b"extension" in result.stderr

    def test_diffusers_runner_rejects_param_override_outside_allowlist(self):
        """Memo §E.1 round-2 security HIGH: param_overrides keys MUST
        be in {true_cfg_scale, cfg_normalization}. ``output_type`` →
        result shape, ``callback_on_step_end`` → callable injection,
        ``cross_attention_kwargs`` → bypasses safety."""
        for bad_key in ["output_type", "callback_on_step_end",
                        "cross_attention_kwargs", "scheduler"]:
            payload = {
                "repo": "org/model",
                "prompt": "x",
                "negative": "",
                "steps": 20,
                "guidance": 4.0,
                "width": 1024,
                "height": 1024,
                "seed": 42,
                "output_path": "/tmp/out.png",
                "param_overrides": {bad_key: "value"},
            }
            result = self._run_runner_with_payload(
                json.dumps(payload).encode("utf-8"),
            )
            assert result.returncode == 64, f"{bad_key=}"
            assert b"allowlist" in result.stderr

    def test_diffusers_runner_validates_int_ranges(self):
        """steps in [1, 500], width/height in [64, 8192], guidance in
        [0.0, 30.0]. Out-of-range values reject before pipeline load."""
        payload_base = {
            "repo": "org/model",
            "prompt": "x",
            "negative": "",
            "steps": 20,
            "guidance": 4.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "output_path": "/tmp/out.png",
        }
        for field, bad_value in [
            ("steps", 0), ("steps", 501),
            ("width", 32), ("width", 16384),
            ("height", 32), ("height", 16384),
            ("guidance", -0.1), ("guidance", 30.1),
        ]:
            payload = {**payload_base, field: bad_value}
            result = self._run_runner_with_payload(
                json.dumps(payload).encode("utf-8"),
            )
            assert result.returncode == 64, (
                f"{field}={bad_value} should reject"
            )

    def test_diffusers_runner_accepts_well_formed_payload(self):
        """Sanity: a fully-valid payload reaches the diffusers import
        path. Since diffusers isn't installed in the main test venv,
        the runner should fail at step 4 (import error) with exit
        code 3 (not 64 — that's input validation; not 0 — that's
        success). Locks the order: validation passes BEFORE the
        heavy imports.
        """
        payload = {
            "repo": "org/model",
            "prompt": "x",
            "negative": "",
            "steps": 20,
            "guidance": 4.0,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "output_path": "/tmp/out.png",
        }
        result = self._run_runner_with_payload(
            json.dumps(payload).encode("utf-8"),
        )
        # Either diffusers IS installed (rc 0 unlikely without weights
        # — would hit network/missing-model error) OR rc 3 (import
        # failed with the friendly bootstrap.sh hint). Critical
        # check: rc is NOT 64 (validation passed).
        assert result.returncode != 64, (
            f"valid payload should pass validation; "
            f"stderr={result.stderr.decode('utf-8', errors='replace')}"
        )


# ── Architect M3: circular-import smoke ────────────────────────────────


def test_diffusers_runner_imports_cleanly_in_isolated_subprocess():
    """Architect commit-6 pre-vet M3: the runner must be importable
    without dragging mflux into the diffusers venv. A subprocess
    that bypasses pytest's import cache catches accidental
    eager-submodule imports in ``imgen/__init__.py`` that would
    propagate to commands/* (which import mflux things).
    """
    result = subprocess.run(
        [sys.executable, "-c", "import imgen.engines._diffusers_runner"],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"runner import failed:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


# ── subprocess_helpers stdin_data extension (architect + security CRITICAL) ─


class TestRunWithStderrRedactionStdinData:
    """Architect + security CRITICAL: extended signature
    ``(cmd, env, log_file=None, *, stdin_data=None)`` — keyword-only
    so legacy positional callers stay binary-compatible.
    """

    def test_signature_back_compat_positional_callers(self):
        """Existing v0.7.x call sites use positional (cmd, env) or
        (cmd, env, log_file). Adding a keyword-only stdin_data MUST
        NOT break those callers — explicit lock-in."""
        from imgen.subprocess_helpers import run_with_stderr_redaction
        import inspect

        sig = inspect.signature(run_with_stderr_redaction)
        params = list(sig.parameters.values())
        # First three: cmd, env, log_file (positional or kw).
        assert params[0].name == "cmd"
        assert params[1].name == "env"
        assert params[2].name == "log_file"
        # stdin_data is keyword-only.
        stdin_param = sig.parameters["stdin_data"]
        assert stdin_param.kind == inspect.Parameter.KEYWORD_ONLY

    def test_stdin_data_written_before_stderr_read_loop(
        self, tmp_path,
    ):
        """End-to-end: send a JSON payload through stdin to a tiny
        child script that prints it back to stderr; verify the
        wrapper writes stdin BEFORE entering the stderr-read loop
        (else the child would block on stdin.read() and we'd
        deadlock)."""
        from imgen.subprocess_helpers import run_with_stderr_redaction

        # Child script: read stdin, write it back to stderr, exit.
        # Lives in a tmp file so we control its content.
        child_script = tmp_path / "echo_child.py"
        child_script.write_text(
            "import sys\n"
            "data = sys.stdin.read()\n"
            "sys.stderr.write(f'GOT: {data}\\n')\n"
            "sys.exit(0)\n"
        )

        payload = b'{"hello": "world"}'
        rc = run_with_stderr_redaction(
            cmd=[sys.executable, str(child_script)],
            env={"PATH": "/usr/bin"},
            stdin_data=payload,
        )
        assert rc == 0


# ── M-NEW-1 (v0.8.3): O_NOFOLLOW symlink-traversal guard ───────────────


class TestOpenOutputForSaveSymlinkGuard:
    """v0.8.3 M-NEW-1: ``_open_output_for_save`` refuses pre-existing
    symlinks via ``O_NOFOLLOW``. Tests the helper directly so they
    exercise the boundary without needing diffusers installed or a
    full pipeline run.
    """

    def test_open_output_for_save_writes_to_fresh_png(self, tmp_path):
        """Plain output path → returns wb file object + 'PNG' format."""
        from imgen.engines._diffusers_runner import _open_output_for_save
        out = tmp_path / "fresh.png"
        fp, fmt = _open_output_for_save(str(out))
        with fp:
            fp.write(b"\x89PNG\r\n\x1a\n")  # PNG magic
        assert fmt == "PNG"
        assert out.read_bytes().startswith(b"\x89PNG")

    def test_open_output_for_save_maps_jpeg_aliases(self, tmp_path):
        """Both .jpg and .jpeg → 'JPEG'; .webp → 'WEBP'."""
        from imgen.engines._diffusers_runner import _open_output_for_save
        for ext, expected in [(".jpg", "JPEG"), (".jpeg", "JPEG"),
                              (".webp", "WEBP")]:
            out = tmp_path / f"img{ext}"
            fp, fmt = _open_output_for_save(str(out))
            with fp:
                fp.write(b"x")
            assert fmt == expected

    def test_open_output_for_save_truncates_existing_file(self, tmp_path):
        """O_TRUNC: writing over a regular file is allowed (this is
        the normal re-run case; output filenames are stable per
        iteration)."""
        from imgen.engines._diffusers_runner import _open_output_for_save
        out = tmp_path / "stale.png"
        out.write_bytes(b"OLD CONTENT")
        fp, _ = _open_output_for_save(str(out))
        with fp:
            fp.write(b"NEW")
        assert out.read_bytes() == b"NEW"

    def test_open_output_for_save_refuses_symlink(self, tmp_path):
        """Pre-existing symlink at output_path → OSError with
        ELOOP. The symlink target MUST stay unchanged."""
        import errno
        from imgen.engines._diffusers_runner import _open_output_for_save
        target = tmp_path / "victim.txt"
        target.write_text("SENSITIVE — must not be overwritten\n")
        link = tmp_path / "out.png"
        link.symlink_to(target)

        with pytest.raises(OSError) as excinfo:
            _open_output_for_save(str(link))
        assert excinfo.value.errno == errno.ELOOP
        # Target untouched.
        assert target.read_text() == "SENSITIVE — must not be overwritten\n"

    def test_open_output_for_save_refuses_unsupported_extension(self, tmp_path):
        """Caller-bypass safety: even if validation is skipped, the
        helper itself rejects extensions outside the PIL format map."""
        from imgen.engines._diffusers_runner import _open_output_for_save
        with pytest.raises(ValueError, match="unsupported output extension"):
            _open_output_for_save(str(tmp_path / "x.gif"))


def test_diffusers_runner_main_refuses_symlink_at_output_path(tmp_path):
    """End-to-end: the runner subprocess hits the O_NOFOLLOW guard
    only after validation + import + pipeline-run, none of which we
    can exercise in CI without diffusers + weights.

    Instead, monkeypatch the pipeline call inside the runner module
    to a stub that returns a fake result, then drive ``main()`` in-
    process with a symlinked output_path. This locks the guard at the
    save call site (not just inside the helper)."""
    import errno

    from imgen.engines import _diffusers_runner as runner

    target = tmp_path / "victim.txt"
    target.write_text("MUST NOT BE OVERWRITTEN\n")
    link = tmp_path / "out.png"
    link.symlink_to(target)

    payload = {
        "repo": "org/model",
        "prompt": "x",
        "negative": "",
        "steps": 20,
        "guidance": 4.0,
        "width": 1024,
        "height": 1024,
        "seed": 42,
        "output_path": str(link),
    }

    # Stub out the heavy imports + pipeline so main() reaches the save
    # path. Inject a stub torch + diffusers into sys.modules; the
    # runner imports them lazily inside main().
    fake_torch = MagicMock()
    fake_torch.bfloat16 = "bf16"
    fake_torch.Generator.return_value.manual_seed.return_value = MagicMock()

    fake_image = MagicMock()
    fake_pipe_result = MagicMock()
    fake_pipe_result.images = [fake_image]

    fake_pipeline = MagicMock(return_value=fake_pipe_result)
    fake_pipeline_class = MagicMock()
    fake_pipeline_class.from_pretrained.return_value = fake_pipeline

    fake_diffusers = MagicMock()
    fake_diffusers.DiffusionPipeline = fake_pipeline_class
    fake_diffusers_utils = MagicMock()

    with patch.dict(sys.modules, {
        "torch": fake_torch,
        "diffusers": fake_diffusers,
        "diffusers.utils": fake_diffusers_utils,
    }), patch.object(
        sys, "stdin",
        MagicMock(buffer=MagicMock(
            read=lambda n: json.dumps(payload).encode("utf-8"),
        )),
    ), patch.object(sys, "stderr", new=MagicMock()) as stderr_mock:
        rc = runner.main()

    assert rc == 1, f"runner should return 1 on symlinked output, got {rc}"
    # Target file untouched — the save was refused before PIL could
    # dereference the symlink.
    assert target.read_text() == "MUST NOT BE OVERWRITTEN\n"
    # Static stderr message uses errno, not the user path.
    stderr_writes = "".join(
        call.args[0] for call in stderr_mock.write.call_args_list
        if call.args
    )
    assert f"errno {errno.ELOOP}" in stderr_writes
    assert "O_NOFOLLOW" in stderr_writes
    # No echo of the user-controlled path in stderr (control-byte
    # avoidance discipline).
    assert str(link) not in stderr_writes
