"""Synthetic checkpoint: selection and editor grants, without workflow execution."""

import dataclasses

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import agent_permissions as ap, cli, config as config_module
from coding_review_agent_loop.config import config_from_args
from coding_review_agent_loop.errors import AgentLoopError


@pytest.fixture
def split_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ap.sys, "platform", "linux")
    return make_config(
        tmp_path, coder="codex", reviewer=("claude",), git_operator="claude",
        agent_permissions="sandboxed", repair_backend="claude",
        repair_models=("repair",), semantic_followup_backend="claude",
    )


def argv(command="issue", *extra):
    return [command, "1", "--repo", "OWNER/REPO", "--coder", "codex",
            "--reviewer", "claude", "--agent-permissions", "sandboxed",
            "--repair-backend", "claude", "--repair-model", "repair",
            "--semantic-followup-backend", "claude", *extra]


@pytest.mark.parametrize("command", ["issue", "pr"])
def test_public_parser_opt_in_and_default(command):
    parser = cli.build_parser()
    assert parser.parse_args(argv(command)).git_operator is None
    assert parser.parse_args(argv(command, "--git-operator", "claude")).git_operator == "claude"
    with pytest.raises(SystemExit):
        parser.parse_args(argv(command, "--git-operator", "codex"))


@pytest.mark.parametrize("command", ["task", "discuss", "managed-pr"])
def test_option_unavailable_on_other_commands(command):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(argv(command, "--git-operator", "claude"))


def test_dataclass_default_and_replace(split_config):
    assert dataclasses.replace(split_config, git_operator=None).git_operator is None
    assert dataclasses.replace(split_config).git_operator == "claude"


@pytest.mark.parametrize("changes", [
    {"git_operator": "codex"}, {"git_operator": ""}, {"git_operator": False},
    {"coder": "claude"}, {"reviewer": ("codex",)},
    {"agent_permissions": "default"}, {"agent_permissions": "dangerous"},
    {"implementation_coder": "claude"}, {"materialize_split_issues": True},
    {"split_stage": 1}, {"plan_execution_mode": "implement-by-phase"},
    {"managed_ci": True}, {"auto_merge": True},
])
def test_invalid_config_table(split_config, changes):
    with pytest.raises(AgentLoopError, match="git-operator"):
        dataclasses.replace(split_config, **changes)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_linux_only(split_config, monkeypatch, platform):
    monkeypatch.setattr(ap.sys, "platform", platform)
    with pytest.raises(AgentLoopError, match="Linux"):
        dataclasses.replace(split_config)


@pytest.mark.parametrize("command,plan_first", [
    ("issue", True), ("task", False), ("discuss", False),
    ("managed-pr", False), ("unknown", False), ("pr", True),
])
def test_split_flow_rejects_unsupported_context(split_config, command, plan_first):
    with pytest.raises(AgentLoopError, match="git-operator"):
        ap.validate_sandboxed_flow(split_config, command=command, plan_first=plan_first)


@pytest.mark.parametrize("command", ["issue", "pr"])
def test_split_flow_accepts_only_explicit_opt_in(split_config, command):
    ap.validate_sandboxed_flow(split_config, command=command)
    with pytest.raises(AgentLoopError, match="committing Codex"):
        ap.validate_sandboxed_flow(dataclasses.replace(split_config, git_operator=None), command=command)


@pytest.mark.parametrize("extra", [
    ["--plan-first"], ["--coder", "claude"], ["--reviewer", "codex"],
    ["--agent-permissions", "default"], ["--agent-permissions", "dangerous"],
    ["--managed-ci"], ["--auto-merge"], ["--materialize-split-issues"],
    ["--plan-execution-mode", "decompose-only"],
])
def test_cli_rejects_before_preflight_or_claims(monkeypatch, capsys, extra):
    def forbidden(*a, **kw):
        pytest.fail("mutable workflow or command preflight reached")
    monkeypatch.setattr(config_module, "preflight_agent_commands", forbidden)
    monkeypatch.setattr(cli, "claim_agent_workdirs", forbidden)
    # Replace the original reviewer for this invalid case, rather than append.
    args = argv("issue", "--git-operator", "claude")
    if extra == ["--reviewer", "codex"]:
        args[args.index("--reviewer") + 1] = "codex"
        extra = []
    assert cli.main([*args, *extra]) == 1
    assert "git-operator" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["issue", "pr"])
def test_public_config_construction(monkeypatch, tmp_path, command):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(config_module, "preflight_agent_commands", lambda *a, **kw: None)
    args = cli.build_parser().parse_args(argv(command, "--git-operator", "claude"))
    config = config_from_args(args, FakeRunner())
    assert config.git_operator == "claude"


def test_editor_grant_is_narrow_and_reversible(split_config, monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "sandboxed_response_root", lambda config: tmp_path)
    assert ap.PERMISSION_CLASS_EDITOR == "editor"
    assert ap.permission_class_for_role("editor") == "read-only"
    assert ap.permission_class_for_role("editor", config=split_config, provider="codex") == "editor"
    grant = ap.role_permission_args(split_config, "codex", "editor")
    assert grant == ("--sandbox", "workspace-write", "-c", 'approval_policy="never"',
                     "-c", "sandbox_workspace_write.network_access=false")
    assert not any(".git" in arg or "gh" in arg or "add-dir" in arg for arg in grant)
    legacy = dataclasses.replace(split_config, git_operator=None)
    assert ap.role_permission_args(legacy, "codex", "editor")[:2] == ("--sandbox", "read-only")
    for role in (None, "reviewer", "planner", "EDITOR", "unknown"):
        assert ap.role_permission_args(split_config, "codex", role)[:2] == ("--sandbox", "read-only")
    for config in (split_config, legacy):
        with pytest.raises(AgentLoopError, match="committing coder"):
            ap.role_permission_args(config, "codex", "coder")
        with pytest.raises(AgentLoopError, match="committing coder"):
            ap.role_permission_env(config, "codex", "coder")


def test_editor_env_cannot_promote_permissions(split_config):
    env = ap.role_permission_env(split_config, "codex", "editor")
    assert env == ap.role_permission_env(split_config, "codex", None)
    assert env["GIT_CONFIG_COUNT"] == "0"
    assert all(env[key] == "0" for key in ap.GIT_TRACE_VARIABLES)


def test_claude_editor_keeps_read_only_grant(split_config, monkeypatch, tmp_path):
    monkeypatch.setattr(ap, "sandboxed_response_root", lambda config: tmp_path)
    monkeypatch.setattr(ap, "inspect_prefix", lambda config: "trusted-inspect")
    assert ap.role_permission_args(split_config, "claude", "editor") == ap.role_permission_args(
        split_config, "claude", "reviewer")


@pytest.mark.parametrize("command", ["issue", "pr"])
def test_checkpoint_stops_before_mutable_workflow(monkeypatch, capsys, split_config, command):
    monkeypatch.setattr(ap, "validate_split_role_capabilities", lambda config: None)
    monkeypatch.setattr(cli, "config_from_args", lambda *a, **kw: split_config)
    monkeypatch.setattr(cli, "claim_agent_workdirs", lambda *a: pytest.fail("checkout claimed"))
    monkeypatch.setattr(cli.tool_provenance, "capture_process_provenance",
                        lambda *a: pytest.fail("workflow started"))
    assert cli.main(argv(command, "--git-operator", "claude")) == 1
    assert "not yet wired" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["issue", "pr"])
def test_validated_selection_is_intentionally_blocked(split_config, monkeypatch, capsys, command):
    probes = []
    monkeypatch.setattr(cli, "config_from_args", lambda *a, **kw: split_config)
    monkeypatch.setattr(ap, "validate_split_role_capabilities", lambda c: probes.append(c))
    def forbidden(*a, **kw):
        pytest.fail("unwired execution reached")
    for name in ("claim_agent_workdirs", "ensure_agent_workdirs", "run_issue_loop", "run_pr_loop"):
        monkeypatch.setattr(cli, name, forbidden)
    assert cli.main(argv(command, "--git-operator", "claude")) == 1
    assert "split-role workflow not yet wired" in capsys.readouterr().err
    assert probes == [split_config]
    with pytest.raises(AgentLoopError, match="split-role workflow not yet wired"):
        ap.establish_sandboxed_run(split_config, command=command)


@pytest.mark.parametrize("failure", [None, "codex", "claude", "mount", "exception", "missing"])
def test_capability_probe_is_closed_and_credential_free(split_config, monkeypatch, failure):
    from types import SimpleNamespace
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setattr(ap.shutil, "which", lambda cmd: None if failure == "missing" else "/usr/bin/" + cmd)
    monkeypatch.setattr(ap, "executable_location_refusal", lambda *a: None)
    flags = ("--sandbox --config --output-last-message --restricted --tools --strict-mcp-config "
             "--mcp-config --permission-mode --permission-prompts --setting-sources --settings "
             "--allowedTools --disallowedTools --no-session-persistence")
    calls = []
    def run(command, **kw):
        calls.append(command)
        assert "ANTHROPIC_API_KEY" not in kw["env"]
        assert kw["env"]["HOME"] == "/nonexistent"
        assert kw["cwd"] == "/"
        if failure == "exception":
            raise OSError("private diagnostic")
        return SimpleNamespace(
            returncode=int(failure == "mount" and "bwrap" in command[0]),
            stdout="" if failure and failure in command[0] else flags,
            stderr="private diagnostic")
    monkeypatch.setattr(ap.subprocess, "run", run)
    if failure:
        with pytest.raises(AgentLoopError, match="capability/mount checks failed") as exc:
            ap.validate_split_role_capabilities(split_config)
        assert "private diagnostic" not in str(exc.value)
    else:
        ap.validate_split_role_capabilities(split_config)
        assert len(calls) == 3
        assert "--ro-bind" in calls[-1] and "--remount-ro" in calls[-1]


@pytest.mark.parametrize("option", ["claude_args", "codex_args", "gemini_args", "antigravity_args"])
def test_split_rejects_all_passthrough(split_config, option):
    with pytest.raises(AgentLoopError, match="not allowed"):
        dataclasses.replace(split_config, **{option: ("--override",)})


def test_named_claude_reviewer_is_supported(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(config_module, "preflight_agent_commands", lambda *a, **kw: None)
    args = argv("pr", "--git-operator", "claude")
    index = args.index("--reviewer")
    args[index:index + 2] = ["--reviewer-seat", "independent=claude", "--seat-model", "independent=model"]
    assert config_from_args(cli.build_parser().parse_args(args), FakeRunner()).git_operator == "claude"


def test_operator_cannot_use_generic_claude_grant(split_config):
    with pytest.raises(AgentLoopError, match="dedicated"):
        ap.role_permission_args(split_config, "claude", "git-operator")
    with pytest.raises(AgentLoopError, match="dedicated"):
        ap.role_permission_env(split_config, "claude", "git-operator")


@pytest.mark.parametrize("command", ["issue", "pr"])
def test_help_describes_checkpoint_and_constraints(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([command, "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for text in ("--git-operator {claude}", "Linux", "Claude reviewer", "not yet wired"):
        assert text in help_text


def test_editor_env_revalidates_selection(split_config, monkeypatch):
    # An invocation must not retain editor privileges after the supported host changes.
    monkeypatch.setattr(ap.sys, "platform", "darwin")
    with pytest.raises(AgentLoopError, match="Linux"):
        ap.role_permission_env(split_config, "codex", "editor")


@pytest.fixture(autouse=True)
def no_real_checkpoint_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("checkpoint tests must stub every subprocess")
    monkeypatch.setattr(ap.subprocess, "run", forbidden)


def test_owner_operator_strings_are_preserved_without_discovery(monkeypatch, tmp_path):
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    monkeypatch.setattr(config_module, 'preflight_agent_commands', lambda *a, **k: None)
    args = cli.build_parser().parse_args(argv('issue', '--git-operator', 'claude',
        '--git-operator-cli-sha256', 'a' * 64,
        '--git-operator-session-file', '/synthetic/.credentials.json'))
    config = config_from_args(args, FakeRunner())
    assert config.git_operator_cli_sha256 == 'a' * 64
    assert config.git_operator_session_file == '/synthetic/.credentials.json'


@pytest.mark.parametrize('changes', [
    {'git_operator_cli_sha256': 'not-a-digest'},
    {'git_operator_session_file': '/synthetic/../.credentials.json'},
    {'git_operator_session_file': 'relative/.credentials.json'},
    {'git_operator_session_file': '/synthetic/\n.credentials.json'},
])
def test_operator_owner_options_reject_unsafe_strings(split_config, changes):
    with pytest.raises(AgentLoopError, match='owner-supplied safe'):
        dataclasses.replace(split_config, **changes)


def test_attested_capability_probe_uses_pinned_cli_without_credentials(split_config, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from coding_review_agent_loop.claude_git_operator import _PinnedFile
    config = dataclasses.replace(split_config, claude_cmd='/synthetic/claude',
        git_operator_cli_sha256='a' * 64,
        git_operator_session_file='/synthetic/.credentials.json')
    events = []
    class Pin:
        fd = 99
        def verify(self): events.append('verify')
        def close(self): events.append('close')
    def pin(path, *, digest):
        assert path == Path('/synthetic/claude') and digest == 'a' * 64
        events.append('pin')
        return Pin()
    monkeypatch.setattr(_PinnedFile, 'open', pin)
    monkeypatch.setattr(ap.shutil, 'which', lambda cmd: '/usr/bin/codex')
    monkeypatch.setattr(ap, 'executable_location_refusal', lambda *a: None)
    flags = ('--sandbox --config --output-last-message --restricted --tools --strict-mcp-config '
             '--mcp-config --permission-mode --permission-prompts --setting-sources --settings '
             '--allowedTools --disallowedTools --no-session-persistence')
    def probe(command, **kwargs):
        assert kwargs['env'] == {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent', 'LC_ALL': 'C'}
        if command[0] == '/proc/self/fd/99':
            assert kwargs['pass_fds'] == (99,)
            events.append('help')
        return SimpleNamespace(returncode=0, stdout=flags)
    monkeypatch.setattr(ap.subprocess, 'run', probe)
    ap.validate_split_role_capabilities(config)
    assert events == ['pin', 'help', 'verify', 'close']
