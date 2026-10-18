"""Ordinary correction transaction; all Git mutations use disposable fixtures."""
import json

import pytest

from agent_loop_helpers import structured_coder_followup
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol import validate_structured_coder_followup


def editor_text(**changes):
    raw, tail = structured_coder_followup(addressed_items=['item-1'],
        tests_run=['python -m pytest tests'], reviewer='OpenAI Codex').split('\n<!--', 1)
    payload = json.loads(raw)
    payload['editor_handoff'] = dict(version=1, outcome='ready', new_paths=[])
    payload.update(changes)
    return json.dumps(payload) + '\n<!--' + tail


def test_followup_extension_only_available_in_opt_in_mode():
    with pytest.raises(AgentLoopError):
        validate_structured_coder_followup(editor_text())
    parsed = validate_structured_coder_followup(editor_text(), allow_editor_handoff=True)
    assert parsed.editor_outcome == 'ready' and parsed.editor_new_paths == ()
    legacy = structured_coder_followup()
    assert validate_structured_coder_followup(legacy).editor_outcome is None


@pytest.mark.parametrize('declaration', [None, {}, {'version': 2, 'outcome': 'ready', 'new_paths': []},
    {'version': 1, 'outcome': 'ready', 'new_paths': ['../escape']},
    {'version': 1, 'outcome': 'ready', 'new_paths': ['x', 'x']}])
def test_invalid_editor_declaration(declaration):
    with pytest.raises(AgentLoopError):
        validate_structured_coder_followup(editor_text(editor_handoff=declaration), allow_editor_handoff=True)

from types import SimpleNamespace
import dataclasses
import shutil
from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from test_git_handoff import case as case, git, operator_commit
from coding_review_agent_loop import split_issue_handoff as split, pr_loop as loop


@pytest.fixture
def correction(case, tmp_path, monkeypatch):
    _, root, old = case
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    git(root, 'push', str(origin), 'HEAD:refs/heads/feature/review-context')
    config = make_config(tmp_path / 'run', coder='codex', reviewer=('claude',),
        git_operator='claude', agent_permissions='sandboxed', repair_backend='claude',
        repair_models=('repair',), semantic_followup_backend='claude', codex_dir=root,
        trusted_local_origin=origin, git_operator_cli_sha256='a' * 64,
        git_operator_session_file='/synthetic/.credentials.json')
    runner = FakeRunner(git_head=old.base, advance_git_head_on_pr=False)
    runner.pr_payload['headRefOid'] = old.base
    runner.git_remote = str(origin)
    phases = []
    for name in ('ensure_agent_workdirs', 'prepare_agent_memory'):
        monkeypatch.setattr(loop, name, lambda *a, **k: None)
    monkeypatch.setattr(loop, '_freeze_prompt_architecture', lambda r, c, **kw: c)
    monkeypatch.setattr(loop, '_read_assigned_workdir_head', lambda *a: git(root, 'rev-parse', 'HEAD').decode())

    def api(method, endpoint, payload=None):
        assert method == 'GET', 'corrections must never create another PR'
        if endpoint == 'user':
            return {'id': 42}
        assert endpoint == 'repos/OWNER/REPO/pulls/77'
        sha = git(origin, 'rev-parse', 'refs/heads/feature/review-context').decode()
        runner.pr_payload['headRefOid'] = sha
        runner.git_head = sha
        return dict(number=77, state='open', head=dict(sha=sha, ref='feature/review-context',
                    repo=dict(full_name='OWNER/REPO')), base=dict(ref='main', repo=dict(full_name='OWNER/REPO')))
    monkeypatch.setattr(split, 'OrdinaryPrApi', lambda *a: api)
    def commit(h):
        phases.append('operator')
        assert git(root, 'rev-parse', 'HEAD').decode() == old.base
        shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
        return operator_commit(h)
    monkeypatch.setattr(split, 'preflight', lambda c: SimpleNamespace(run=commit))
    from coding_review_agent_loop import git_transport, agent_permissions as ap
    def import_ref(destination, source_ref, tracking_ref, **kwargs):
        return (git(origin, 'rev-parse', 'refs/heads/feature/review-context').decode()
                if 'pull/' in source_ref else old.base)
    monkeypatch.setattr(git_transport, 'import_ref', import_ref)
    monkeypatch.setattr(ap, 'inspect_prefix', lambda c: 'synthetic-inspect')
    def probe(args, cwd):
        if cwd == root:
            return SimpleNamespace(returncode=0, stdout=split._run(None, root, args), stderr=b'')
        return runner.run_binary(('git', *args), cwd=cwd, check=False)
    monkeypatch.setattr(ap, 'hardened_git_probe_runner_bytes', lambda c: probe)
    def text_probe(args, cwd):
        r = probe(args, cwd)
        return SimpleNamespace(returncode=r.returncode, stdout=r.stdout.decode(), stderr='')
    monkeypatch.setattr(ap, 'hardened_git_probe_runner', lambda c: text_probe)
    runner.claude_outputs = [structured_pr_review(state='blocking', blocking_items=['Fix old.'],
        reviewer='Anthropic Claude'), structured_pr_review(reviewer='Anthropic Claude',
        prior_item_dispositions=[dict(item_id='item-1', disposition='resolved')])]
    runner.codex_outputs = [editor_text()]
    real_invoke = runner.run_with_log
    def invoke(args, **kwargs):
        if args[0] == 'codex':
            phases.append('editor')
            assert '--sandbox' in args and 'workspace-write' in args
            assert git(root, 'rev-parse', 'HEAD').decode() == old.base
            (root / 'old').write_text('after\n')
        elif args[0] == 'claude':
            phases.append('reviewer')
        return real_invoke(args, **kwargs)
    monkeypatch.setattr(runner, 'run_with_log', invoke)
    # Synthetic host test receipt attached at the validated acquisition seam;
    # model prose alone is deliberately insufficient to publish.
    from coding_review_agent_loop.local_test_evidence import LocalTestObservation
    real_turn = loop._run_validated_agent
    def turn(*a, **kw):
        result = real_turn(*a, **kw)
        if kw.get('role') == 'editor':
            result = dataclasses.replace(result, acquisition_test_observations=(LocalTestObservation(
                command=('python', '-m', 'pytest', 'tests'), outcome='passed',
                provenance='parent-observed', cwd=str(root), normalized_command='python -m pytest tests'),))
        return result
    monkeypatch.setattr(loop, '_run_validated_agent', turn)
    return runner, config, root, old, origin, phases


def test_public_pr_blocking_editor_operator_then_independent_approval(correction):
    runner, config, root, old, origin, phases = correction
    assert loop.run_pr_loop(runner, pr_number=77, config=config) == 0
    assert phases == ['reviewer', 'editor', 'operator', 'reviewer']
    sha = git(origin, 'rev-parse', 'refs/heads/feature/review-context').decode()
    assert sha != old.base
    assert git(origin, 'rev-parse', sha + '^').decode() == old.base
    assert git(origin, 'show', sha + ':old') == b'after'
    assert git(root, 'rev-parse', 'HEAD').decode() == sha
    assert all(c['pr_number'] == 77 for c in runner.comment_calls) if hasattr(runner, 'comment_calls') else runner.comments


# Adapter-level refusals: each must leave the remote PR branch exactly as found.
REF = 'refs/heads/feature/review-context'


def begin_adapter(correction, monkeypatch):
    runner, config, root, old, origin, phases = correction
    owner = SimpleNamespace(command='pr', number=77)
    monkeypatch.setattr(split, 'current_claim_owner', lambda: owner)
    metadata = SimpleNamespace(number=77, repo=config.repo, head_branch='feature/review-context',
                               base_branch='main', head_sha=old.base)
    return split.SplitPrHandoff.begin(runner, config=config, pr_number=77, metadata=metadata)


def ready(**changes):
    return validate_structured_coder_followup(editor_text(**changes), allow_editor_handoff=True)


def receipt(adapter):
    from coding_review_agent_loop.local_test_evidence import LocalTestObservation
    return (LocalTestObservation(command=('python', '-m', 'pytest', 'tests'), outcome='passed',
        provenance='parent-observed', cwd=str(adapter.handoff.checkout),
        normalized_command='python -m pytest tests'),)


def publish(adapter, parsed=None, *, head_change_expected=True, observations=None):
    return adapter.publish(parsed or ready(), head_change_expected=head_change_expected,
                           test_observations=receipt(adapter) if observations is None else observations)


def test_changed_source_head_refused_before_publication(correction, monkeypatch):
    _, _, root, old, origin, _ = correction
    adapter = begin_adapter(correction, monkeypatch)
    (root / 'old').write_text('after\n')
    moved = git(origin, 'commit-tree', old.base + '^{tree}', '-p', old.base, '-m', 'concurrent').decode()
    git(origin, 'update-ref', REF, moved)
    with pytest.raises(split.HandoffError):
        publish(adapter)
    assert git(origin, 'rev-parse', REF).decode() == moved


def test_no_change_requires_host_classified_evidence_only_turn(correction, monkeypatch):
    _, _, root, old, origin, _ = correction
    adapter = begin_adapter(correction, monkeypatch)
    with pytest.raises(split.HandoffError):
        publish(adapter, head_change_expected=True)
    assert git(origin, 'rev-parse', REF).decode() == old.base
    adapter = begin_adapter(correction, monkeypatch)
    assert publish(adapter, head_change_expected=False) == old.base
    assert git(origin, 'rev-parse', REF).decode() == old.base


def test_unauthorized_new_file_refused(correction, monkeypatch):
    _, _, root, old, origin, _ = correction
    adapter = begin_adapter(correction, monkeypatch)
    (root / 'old').write_text('after\n')
    (root / 'surprise.txt').write_text('undeclared\n')
    with pytest.raises(split.HandoffError):
        publish(adapter)
    assert git(origin, 'rev-parse', REF).decode() == old.base


@pytest.mark.parametrize('fault', ['parent', 'tree'])
def test_wrong_parent_or_tree_from_operator_refused(correction, monkeypatch, fault):
    _, _, root, old, origin, _ = correction
    def bad(h):
        shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
        if fault == 'parent':
            return operator_commit(h, parent='1' * 40)
        return operator_commit(h, tree=git(root, 'rev-parse', old.base + '^{tree}').decode())
    monkeypatch.setattr(split, 'preflight', lambda c: SimpleNamespace(run=bad))
    adapter = begin_adapter(correction, monkeypatch)
    (root / 'old').write_text('after\n')
    with pytest.raises(split.HandoffError):
        publish(adapter)
    assert git(origin, 'rev-parse', REF).decode() == old.base


def test_race_after_commit_before_remote_publication_refused(correction, monkeypatch):
    _, _, root, old, origin, _ = correction
    raced = git(origin, 'commit-tree', old.base + '^{tree}', '-p', old.base, '-m', 'race').decode()
    def commit_then_race(h):
        shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
        sha = operator_commit(h)
        git(origin, 'update-ref', REF, raced)
        return sha
    monkeypatch.setattr(split, 'preflight', lambda c: SimpleNamespace(run=commit_then_race))
    adapter = begin_adapter(correction, monkeypatch)
    (root / 'old').write_text('after\n')
    with pytest.raises(split.HandoffError):
        publish(adapter)
    assert git(origin, 'rev-parse', REF).decode() == raced
