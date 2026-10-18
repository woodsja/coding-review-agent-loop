"""Trusted handoff metadata must be unreachable and tamper-evident for the editor.

The denial tests run commands under the REAL Codex workspace-write sandbox
(`codex sandbox`, no model call, no credits) with equivalent sandbox
configuration overrides. The production `codex exec` role additionally uses
`--ignore-user-config` and `--ignore-rules` (not accepted by the `sandbox`
subcommand). The permission arguments are separately checked by the public
role-grant tests, so a regression that puts trusted state back under /tmp fails here
with a genuine operating-system result rather than a mocked one.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from coding_review_agent_loop.git_handoff import (CommitIdentity, Handoff, HandoffError,
                                                  private_trusted_dir)
from test_git_handoff import case as case, git  # noqa: F401  (fixture)

EDITOR_FLAGS = ("-c", "permissions={}", "-c", 'sandbox_mode="workspace-write"', "-c", "sandbox_workspace_write.network_access=false", "-c", "sandbox_workspace_write.writable_roots=[]")


def sandboxed(cwd, script):
    """Run a shell script under the real Codex workspace-write sandbox."""
    # The suite patches shutil.which() to provide fake providers for config tests.
    # Resolve a real executable: mock placeholders are not a sandbox test.
    candidate = Path(os.environ.get("AGENTLOOP_TEST_REAL_CODEX",
                       str(Path.home() / ".local" / "bin" / "codex")))
    if not (candidate.is_file() and os.access(candidate, os.X_OK)):
        candidate = Path(shutil.which("codex") or "")
    if not (candidate.is_file() and os.access(candidate, os.X_OK)):
        pytest.skip("Real Codex sandbox CLI unavailable on this host; Ralph live gate required")
    codex = str(candidate)
    return subprocess.run([codex, "sandbox", *EDITOR_FLAGS, "--", "sh", "-c", script],
                          cwd=cwd, capture_output=True, text=True, timeout=60)


@pytest.fixture
def trusted_handoff(case):
    repo, checkout, _ = case
    base = git(repo, "rev-parse", "HEAD").decode()
    trusted = private_trusted_dir(checkout, "test-handoff")
    h = Handoff.begin(checkout, trusted, CommitIdentity("ReviewLoop", "loop@example.invalid", 1700000000,
                                                         "Implement candidate\n"), expected_base=base)
    return checkout, h


def test_trusted_state_is_outside_every_editor_writable_root(trusted_handoff):
    checkout, h = trusted_handoff
    for root in (Path("/tmp"), Path(tempfile.gettempdir()), checkout):
        assert not h.trusted.resolve().is_relative_to(root.resolve())
    assert (h.trusted.parent.stat().st_mode & 0o077) == 0
    assert h.trusted.parent.parent.stat().st_mode & 0o077 == 0


def test_real_codex_sandbox_cannot_modify_or_inject_trusted_git_metadata(trusted_handoff):
    checkout, h = trusted_handoff
    control = Path("/tmp") / f"reviewloop-control-{os.getpid()}"
    try:
        # Control: the same sandbox really can write /tmp and the checkout, so a
        # denial below is attributable to the trusted location, not a broken probe.
        ok = sandboxed(checkout, f"echo x > {control} && echo x > {checkout}/probe")
        assert ok.returncode == 0, ok.stderr
        before = {p: p.read_bytes() for p in (h.admin / "config", h.admin / "HEAD")}
        for attack in (f"printf '[core]\\n\\tfsmonitor = /bin/evil\\n' >> {h.admin}/config",
                       f"echo evil > {h.admin}/HEAD",
                       f"mkdir {h.admin}/hooks",
                       f"mkdir -p {h.admin}/info && echo '* filter=evil' > {h.admin}/info/attributes",
                       f"echo /tmp/evil > {h.admin}/objects/info/alternates",
                       f"rm -rf {h.trusted}",
                       f"mkdir {h.trusted.parent}/planted"):
            denied = sandboxed(checkout, attack)
            assert denied.returncode != 0, attack
            assert "Read-only file system" in denied.stderr or "Permission denied" in denied.stderr, denied.stderr
        assert {p: p.read_bytes() for p in before} == before
    finally:
        control.unlink(missing_ok=True)


@pytest.mark.parametrize("tamper", ["config", "head", "hooks", "attributes", "alternates", "base-index", "link"])
def test_tampered_trusted_metadata_is_rejected(trusted_handoff, tamper):
    checkout, h = trusted_handoff
    admin = h.admin
    if tamper == "config":
        (admin / "config").write_text((admin / "config").read_text() + "[core]\n\tfsmonitor = /bin/true\n")
    elif tamper == "head":
        (admin / "HEAD").write_text("0" * 40 + "\n")
    elif tamper == "hooks":
        (admin / "hooks").mkdir()
        (admin / "hooks" / "pre-commit").write_text("#!/bin/sh\n")
    elif tamper == "attributes":
        (admin / "info").mkdir()
        (admin / "info" / "attributes").write_text("* filter=evil\n")
    elif tamper == "alternates":
        (admin / "objects" / "info" / "alternates").write_text("/tmp/evil\n")
    elif tamper == "base-index":
        (admin / "base-index").write_bytes(b"DIRC")
    else:
        (admin / "refs" / "heads").symlink_to("/tmp")
    (checkout / "old").write_text("after\n")
    with pytest.raises(HandoffError, match="sealed"):
        h.capture(allowed_new=set())


def test_planted_loose_object_before_capture_is_rejected(trusted_handoff):
    checkout, h = trusted_handoff
    (h.objects / "ab").mkdir()
    (checkout / "old").write_text("after\n")
    with pytest.raises(HandoffError, match="sealed"):
        h.capture(allowed_new=set())


def test_untampered_metadata_still_captures_and_verifies_seal_afterwards(trusted_handoff):
    checkout, h = trusted_handoff
    (checkout / "old").write_text("after\n")
    h.capture(allowed_new=set())
    (h.admin / "config").write_text((h.admin / "config").read_text() + "[core]\n\thooksPath = /tmp\n")
    with pytest.raises(HandoffError, match="sealed"):
        h.verify("0" * 40)


@pytest.mark.parametrize("where", ["/tmp", "tmpdir", "checkout"])
def test_state_home_inside_editor_writable_root_is_refused(case, monkeypatch, tmp_path, where):
    _, checkout, _ = case
    if where == "/tmp":
        monkeypatch.setenv("XDG_STATE_HOME", "/tmp/reviewloop-state")
    elif where == "tmpdir":
        monkeypatch.setenv("TMPDIR", str(tmp_path))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    else:
        monkeypatch.setenv("XDG_STATE_HOME", str(checkout / "state"))
    with pytest.raises(HandoffError, match="editor-writable"):
        private_trusted_dir(checkout, "test-handoff")
    assert not Path("/tmp/reviewloop-state").exists()


def test_state_dir_must_be_private_and_not_group_writable_ancestor(case, monkeypatch):
    _, checkout, _ = case
    home = Path.home() / ".cache"
    shared = Path(tempfile.mkdtemp(prefix="reviewloop-perm-", dir=home))
    try:
        shared.chmod(0o777)
        monkeypatch.setenv("XDG_STATE_HOME", str(shared))
        with pytest.raises(HandoffError, match="writable ancestor"):
            private_trusted_dir(checkout, "test-handoff")
    finally:
        shutil.rmtree(shared, ignore_errors=True)


def test_real_codex_sandbox_cannot_alter_operator_session_or_private_area(trusted_handoff):
    checkout, h = trusted_handoff
    session = h.trusted.parent / "session" / ".credentials.json"
    session.parent.mkdir()
    session.write_text("synthetic-not-a-secret")
    (checkout / "old").write_text("after\n")
    h.capture(allowed_new=set())
    ctx = h.operator_context()
    for target in (session, ctx.private / "index"):
        denied = sandboxed(checkout, f"echo tampered >> {target}")
        assert denied.returncode != 0 and "Read-only file system" in denied.stderr
    assert session.read_text() == "synthetic-not-a-secret"


@pytest.mark.parametrize("config_source", ["user", "project"])
def test_hostile_codex_settings_cannot_add_trusted_state_as_writable_root(
        trusted_handoff, tmp_path, config_source):
    checkout, h = trusted_handoff
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    # Exercise a would-be permission expansion via either Codex's user config
    # or project-local settings. Our explicit overrides must take precedence.
    config = (codex_home / "config.toml" if config_source == "user"
              else checkout / ".codex" / "config.toml")
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[sandbox_workspace_write]\nwritable_roots = ["' +
                      str(h.trusted.parent.resolve()) + '"]\n')
    sentinel = h.trusted.parent / ("attempt-" + config_source)
    candidate = Path(os.environ.get("AGENTLOOP_TEST_REAL_CODEX",
                         str(Path.home() / ".local" / "bin" / "codex")))
    if not candidate.is_file():
        pytest.skip("Real Codex CLI unavailable; Ralph release gate required")
    env = {**os.environ, "CODEX_HOME": str(codex_home)}
    result = subprocess.run([str(candidate), "sandbox", *EDITOR_FLAGS,
                             "--", "sh", "-c", f"echo forbidden > {sentinel}"],
                            cwd=checkout, env=env, capture_output=True,
                            text=True, timeout=60)
    assert result.returncode != 0
    assert not sentinel.exists()
    assert "Read-only file system" in result.stderr or "Permission denied" in result.stderr


def test_production_codex_exec_is_compatible_with_explicit_hardening_flags():
    candidate = Path(os.environ.get("AGENTLOOP_TEST_REAL_CODEX",
                         str(Path.home() / ".local" / "bin" / "codex")))
    if not candidate.is_file():
        pytest.skip("Real Codex CLI unavailable; Ralph release gate required")
    result = subprocess.run([str(candidate), "exec", "--ignore-user-config",
                             "--ignore-rules", "--sandbox", "workspace-write",
                             "-c", "permissions={}",
                             "-c", "sandbox_workspace_write.writable_roots=[]",
                             "--help"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
