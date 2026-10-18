"""Direct issue public seam, using only disposable Git and fake providers."""
import json
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_issue_implementation
from test_git_handoff import case as case, git
from coding_review_agent_loop import split_issue_handoff as split

from coding_review_agent_loop import issue_loop as loop
from coding_review_agent_loop.errors import AgentLoopError

REAL_PREFLIGHT = split.preflight


def editor_text(**changes):
    text = structured_issue_implementation(pr_number=None, tests_run=['python -m pytest tests'],
                                          reviewer='OpenAI Codex')
    raw, tail = text.split('\n<!--', 1)
    payload = json.loads(raw)
    payload['editor_handoff'] = dict(version=1, outcome='ready', new_paths=[])
    payload.update(changes)
    return json.dumps(payload) + '\n<!--' + tail


@pytest.fixture
def direct(case, tmp_path, monkeypatch):
    _, root, h = case
    config = make_config(tmp_path / 'run', coder='codex', reviewer=('claude',),
                         git_operator='claude', agent_permissions='sandboxed',
                         repair_backend='claude', repair_models=('repair',),
                         semantic_followup_backend='claude', codex_dir=root,
                         git_operator_cli_sha256='a' * 64,
                         git_operator_session_file='/synthetic/.credentials.json')
    runner = FakeRunner(git_head=h.base, advance_git_head_on_pr=False)
    for name in ('ensure_agent_workdirs', 'sync_coder_base_before_implementation'):
        monkeypatch.setattr(loop, name, lambda *a, **k: None)
    monkeypatch.setattr(loop, '_freeze_prompt_architecture', lambda r, c: c)
    monkeypatch.setattr(loop, '_read_assigned_workdir_head', lambda *a: git(root, 'rev-parse', 'HEAD').decode())
    monkeypatch.setattr(loop, 'prepare_agent_memory', lambda *a: None)
    monkeypatch.setattr(split, 'preflight', lambda c: SimpleNamespace(run=lambda h: pytest.fail('operator launched')))
    monkeypatch.setattr(split, 'OrdinaryPrApi', lambda *a: lambda *a: {'id': 42})
    return runner, config, root, h


@pytest.mark.parametrize('fault,diagnostic', [('no-change', 'No candidate changes'),
                                            ('head', 'shared Git metadata changed')])
def test_direct_issue_refuses_no_change_or_editor_git_mutation(direct, monkeypatch, fault, diagnostic):
    runner, config, root, h = direct
    def edit(*a, **kw):
        assert kw['role'] == 'editor'
        if fault == 'head':
            foreign = git(root, 'commit-tree', h.base_tree, '-p', h.base, data=b'Foreign\n').decode()
            git(root, 'update-ref', 'HEAD', foreign)
        parsed = kw['validate'](editor_text())
        return SimpleNamespace(text=editor_text(), session_id=None, marker_value=parsed, model_used=None)
    monkeypatch.setattr(loop, '_run_validated_agent', edit)
    with pytest.raises(AgentLoopError, match=diagnostic):
        loop.run_issue_loop(runner, issue_number=1, config=config)
    assert not runner.comments
    assert not runner.issue_comments


@pytest.mark.parametrize("additional_issues", [(), (2, 3)])
def test_direct_issue_publishes_exact_candidate_then_independent_review(direct, tmp_path, monkeypatch, additional_issues):
    import dataclasses
    import shutil
    from agent_loop_helpers import structured_pr_review
    from test_git_handoff import operator_commit
    runner, config, root, old = direct
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    config = dataclasses.replace(config, trusted_local_origin=origin, expected_closing_issue_ids=additional_issues)
    runner.git_remote = str(origin)
    phases = []

    def commit(h):
        phases.append('operator')
        assert (root / 'old').read_text() == 'after\n'
        assert (root / 'new').read_text() == 'new\n'
        shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
        sha = operator_commit(h)
        assert git(root, 'rev-parse', 'HEAD').decode() == old.base
        return sha
    monkeypatch.setattr(split, 'preflight', lambda c: SimpleNamespace(run=commit))

    class Api:
        pr = None
        creates = 0
        def __call__(self, method, endpoint, payload=None):
            if endpoint == 'user':
                return {'id': 42}
            if '/pulls?' in endpoint:
                return [self.pr] if self.pr else []
            if method == 'POST':
                self.creates += 1
                sha = git(origin, 'rev-parse', 'refs/heads/' + payload['head']).decode()
                self.pr = dict(number=77, state='open', body=payload['body'], user={'id': 42},
                    head=dict(sha=sha, ref=payload['head'], repo=dict(full_name='OWNER/REPO')),
                    base=dict(ref='main', repo=dict(full_name='OWNER/REPO')))
                runner.pr_payload.update(body=payload['body'], headRefOid=sha, headRefName=payload['head'])
                runner.git_head = sha
                phases.append('PR')
                return self.pr
            return self.pr
    api = Api()
    from coding_review_agent_loop import git_transport
    def import_for_fake_reviewer(destination, source_ref, tracking_ref, **kwargs):
        assert destination in (config.claude_dir, root)
        return (git(origin, 'rev-parse', 'refs/heads/' + api.pr['head']['ref']).decode()
                if 'pull/' in source_ref else old.base)
    monkeypatch.setattr(git_transport, 'import_ref', import_for_fake_reviewer)
    monkeypatch.setattr(split, 'OrdinaryPrApi', lambda *a: api)
    real_run = runner.run_with_log
    def invoke(args, **kwargs):
        if args[0] == 'codex':
            phases.append('editor')
            assert git(root, 'rev-parse', 'HEAD').decode() == old.base
            (root / 'old').write_text('after\n')
            (root / 'new').write_text('new\n')
            assert '--sandbox' in args and 'workspace-write' in args
        elif args[0] == 'claude':
            phases.append('reviewer')
            assert api.pr is not None
            assert git(root, 'rev-parse', 'HEAD').decode() == api.pr['head']['sha']
            assert 'operator' not in kwargs.get('input_text', '').lower()[:100]
        return real_run(args, **kwargs)
    monkeypatch.setattr(runner, 'run_with_log', invoke)
    runner.codex_outputs = [editor_text(editor_handoff=dict(version=1, outcome='ready', new_paths=['new']))]
    runner.claude_outputs = [structured_pr_review(reviewer='Anthropic Claude')]
    # The permission wrapper is not launched by FakeRunner. Its inspector path
    # is a synthetic executable string, not a live capability probe.
    from coding_review_agent_loop import agent_permissions as ap
    monkeypatch.setattr(ap, 'inspect_prefix', lambda c: 'synthetic-inspect')
    def probes(c):
        def probe(args, cwd):
            if cwd == root:
                return SimpleNamespace(returncode=0, stdout=split._run(None, root, args), stderr=b'')
            return runner.run_binary(('git', *args), cwd=cwd, check=False)
        return probe
    monkeypatch.setattr(ap, 'hardened_git_probe_runner_bytes', probes)
    def text_probes(c):
        def probe(args, cwd):
            result = probes(c)(args, cwd)
            return SimpleNamespace(returncode=result.returncode, stdout=result.stdout.decode(), stderr='')
        return probe
    monkeypatch.setattr(ap, 'hardened_git_probe_runner', text_probes)
    assert loop.run_issue_loop(runner, issue_number=1, config=config) == 0
    assert phases == ['editor', 'operator', 'PR', 'reviewer']
    assert api.creates == 1
    sha = api.pr['head']['sha']
    assert git(origin, 'show', sha + ':old') == b'after'
    assert git(origin, 'show', sha + ':new') == b'new'
    assert git(origin, 'rev-parse', sha + '^').decode() == old.base
    assert all(f'Closes #{i}' in api.pr['body'] for i in (1, *additional_issues))
    assert 'AGENT_LOOP_MANAGED' not in api.pr['body']
    assert any('AGENT_ISSUE_PR_HANDOFF' in c['body'] for c in runner.issue_comments)


@pytest.mark.parametrize('fault', ['missing', 'blocked', 'tests', 'architecture', 'undeclared',
                                  'unsupported', 'clarification', 'wrong-base', 'changed-source', 'wrong-tree',
                                  'foreign-closing'])
def test_direct_issue_rejected_editor_never_publishes(direct, monkeypatch, fault):
    import shutil
    from test_git_handoff import operator_commit
    from coding_review_agent_loop.agent_failure import ValidatedAgentResponse
    runner, config, root, old = direct
    text = editor_text()
    payload, tail = text.split('\n<!--', 1)
    data = json.loads(payload)
    if fault == 'foreign-closing':
        data['summary'] = 'Closes #999'
    if fault == 'missing':
        del data['editor_handoff']
    elif fault == 'blocked':
        data['editor_handoff']['outcome'] = 'blocked'
    elif fault == 'tests':
        data['tests_run'] = []
    elif fault == 'architecture':
        del data['architecture_impact']
    elif fault == 'unsupported':
        data['editor_handoff']['version'] = 2
    text = json.dumps(data) + '\n<!--' + tail
    if fault == 'clarification':
        text = 'Please clarify.\n<!-- AGENT_STATE: clarification -->\n-- OpenAI Codex'
    if fault == 'wrong-base':
        monkeypatch.setattr(loop, '_read_assigned_workdir_head', lambda *a: '0' * 40)
    def edit(*a, **kw):
        assert fault != 'wrong-base', 'unexpected base must refuse before editor'
        (root / 'old').write_text('after\n')
        if fault == 'undeclared':
            (root / 'new').write_text('undeclared\n')
        parsed = kw['validate'](text)
        return ValidatedAgentResponse(text=text, session_id=None, marker_value=parsed)
    monkeypatch.setattr(loop, '_run_validated_agent', edit)
    def hostile(h):
        assert fault in ('changed-source', 'wrong-tree')
        shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
        sha = operator_commit(h, tree=h.base_tree if fault == 'wrong-tree' else None)
        if fault == 'changed-source':
            (root / 'old').write_text('operator changed it\n')
        return sha
    monkeypatch.setattr(split, 'preflight', lambda c: SimpleNamespace(run=hostile))
    monkeypatch.setattr(split, 'publish_verified', lambda *a, **k: pytest.fail('unchecked publication'))
    with pytest.raises(AgentLoopError):
        loop.run_issue_loop(runner, issue_number=1, config=config)
    assert not runner.comments and not runner.issue_comments


def test_missing_owner_auth_stops_before_editor(direct, monkeypatch):
    import dataclasses
    runner, config, _, _ = direct
    monkeypatch.setattr(split, "preflight", REAL_PREFLIGHT)
    monkeypatch.setattr(loop, "_run_validated_agent", lambda *a, **k: pytest.fail("editor started"))
    config = dataclasses.replace(config, git_operator_cli_sha256=None, git_operator_session_file=None)
    with pytest.raises(AgentLoopError, match='Supply --git-operator-cli-sha256'):
        loop.run_issue_loop(runner, issue_number=1, config=config)


def test_extension_is_opt_in_and_legacy_null_stays_terminal():
    from coding_review_agent_loop.protocol import validate_structured_issue_implementation
    with pytest.raises(AgentLoopError):
        validate_structured_issue_implementation(editor_text())
    parsed = validate_structured_issue_implementation(editor_text(), allow_editor_handoff=True)
    assert parsed.pr_number is None and parsed.editor_new_paths == () and parsed.editor_outcome == 'ready'
    legacy = validate_structured_issue_implementation(structured_issue_implementation(pr_number=None),
                                                       allow_editor_handoff=True)
    assert legacy.editor_outcome is None and legacy.editor_new_paths is None


def test_host_auth_failure_precedes_editor(direct, monkeypatch):
    runner, config, _, _ = direct
    monkeypatch.setattr(split, 'OrdinaryPrApi', lambda *a: lambda *a: {})
    monkeypatch.setattr(loop, '_run_validated_agent', lambda *a, **k: pytest.fail('editor started'))
    with pytest.raises(AgentLoopError, match='authentication.*before editing'):
        loop.run_issue_loop(runner, issue_number=1, config=config)
    assert not runner.comments and not runner.issue_comments


def test_editor_prompt_has_no_coder_git_or_publication_grant(direct):
    from coding_review_agent_loop.prompts import build_issue_prompt
    _, config, _, _ = direct
    prompt = build_issue_prompt(1, config)
    assert 'NEVER branch, commit, push, open a PR, or change Git metadata' in prompt
    assert 'workspace-write' in prompt and 'network access is disabled' in prompt
    assert 'Create a branch' not in prompt
    assert 'commits, and pushes' not in prompt
    assert 'gh subcommands this prompt names' not in prompt
    assert '"editor_handoff"' in prompt and '"new_paths": []' in prompt
    assert '"pr_number": null' in prompt
