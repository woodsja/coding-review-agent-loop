"""Durable restart/recovery of EDITED, COMMITTED, PUSHED and PR_BOUND.

A "crash" is a BaseException injected at a phase boundary; the restart is a NEW
adapter in a NEW clean linked worktree, as a real rerun would get. All Git is
disposable and local; no provider, network or GitHub is used.
"""
import dataclasses
import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_coder_followup, structured_issue_implementation
from test_git_handoff import case as case, git, operator_commit  # noqa: F401
from test_handoff_trusted_state import sandboxed
from coding_review_agent_loop import split_issue_handoff as split
from coding_review_agent_loop.git_handoff import HandoffError, state_base
from coding_review_agent_loop.handoff_journal import HandoffJournal
from coding_review_agent_loop.protocol import (validate_structured_coder_followup,
                                               validate_structured_issue_implementation)

BRANCH = 'feature/review-context'


class Crash(BaseException):
    """Simulated process death: nothing in the adapter may catch or absorb it."""


def issue_text(**changes):
    text = structured_issue_implementation(pr_number=None, tests_run=['python -m pytest tests'],
                                          reviewer='OpenAI Codex')
    raw, tail = text.split('\n<!--', 1)
    payload = json.loads(raw)
    payload['editor_handoff'] = dict(version=1, outcome='ready', new_paths=[])
    payload.update(changes)
    return json.dumps(payload) + '\n<!--' + tail


def pr_text():
    raw, tail = structured_coder_followup(addressed_items=['item-1'],
        tests_run=['python -m pytest tests'], reviewer='OpenAI Codex').split('\n<!--', 1)
    payload = json.loads(raw)
    payload['editor_handoff'] = dict(version=1, outcome='ready', new_paths=[])
    return json.dumps(payload) + '\n<!--' + tail


class Api:
    """Idempotent in-memory GitHub; counts creations and reads the real bare origin."""
    def __init__(self, origin):
        self.origin, self.pr, self.creates = origin, None, 0

    def __call__(self, method, endpoint, payload=None):
        if endpoint == 'user':
            return {'id': 42}
        if '/pulls?' in endpoint:
            return [self.pr] if self.pr else []
        if method == 'POST':
            self.creates += 1
            sha = git(self.origin, 'rev-parse', 'refs/heads/' + payload['head']).decode()
            self.pr = dict(number=77, state='open', body=payload['body'], user={'id': 42},
                           head=dict(sha=sha, ref=payload['head'], repo=dict(full_name='OWNER/REPO')),
                           base=dict(ref='main', repo=dict(full_name='OWNER/REPO')))
            return self.pr
        if self.pr is not None and endpoint.endswith('/pulls/77'):
            return self.pr
        # Existing-PR correction target.
        sha = git(self.origin, 'rev-parse', 'refs/heads/' + BRANCH).decode()
        return dict(number=77, state='open', head=dict(sha=sha, ref=BRANCH, repo=dict(full_name='OWNER/REPO')),
                    base=dict(ref='main', repo=dict(full_name='OWNER/REPO')))


@pytest.fixture
def env(case, tmp_path, monkeypatch):
    repo, root, old = case
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    git(root, 'push', str(origin), 'HEAD:refs/heads/' + BRANCH)
    config = make_config(tmp_path / 'run', coder='codex', reviewer=('claude',),
        git_operator='claude', agent_permissions='sandboxed', repair_backend='claude',
        repair_models=('repair',), semantic_followup_backend='claude', codex_dir=root,
        trusted_local_origin=origin, git_operator_cli_sha256='a' * 64,
        git_operator_session_file='/synthetic/.credentials.json', expected_closing_issue_ids=(1,))
    api = Api(origin)
    state = SimpleNamespace(operator_runs=0, crash=None, owner=None)
    runner = FakeRunner(git_head=old.base, advance_git_head_on_pr=False)
    monkeypatch.setattr(split, 'OrdinaryPrApi', lambda *a: api)

    def operator(h):
        state.operator_runs += 1
        if state.crash == 'operator':
            raise Crash
        shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
        return operator_commit(h)
    monkeypatch.setattr(split, 'preflight', lambda c: SimpleNamespace(run=operator))
    real_push, real_bind = split.SplitIssueHandoff._push, split.bind_ordinary_pr

    def push(self, **kw):
        if state.crash == 'push':
            raise Crash
        return real_push(self, **kw)
    monkeypatch.setattr(split.SplitIssueHandoff, '_push', push)

    def bind(**kw):
        if state.crash == 'bind':
            raise Crash
        return real_bind(**kw)
    monkeypatch.setattr(split, 'bind_ordinary_pr', bind)

    def fresh_worktree(name):
        path = tmp_path / name
        git(repo, 'worktree', 'add', '--detach', str(path), old.base)
        return path

    def configured(path):
        return dataclasses.replace(config, codex_dir=path)

    def claim(command, number):
        state.owner = SimpleNamespace(command=command, number=number, run_id='run' + name_counter())
        monkeypatch.setattr(split, 'current_claim_owner', lambda: state.owner)

    counter = iter(range(1, 100))
    name_counter = lambda: str(next(counter))
    return SimpleNamespace(repo=repo, root=root, old=old, origin=origin, config=config, api=api, state=state,
                           runner=runner, fresh=fresh_worktree, configured=configured, claim=claim)


def begin_issue(env, root, config=None):
    env.claim('issue', 1)
    config = config or env.configured(root)
    return split.SplitIssueHandoff.begin(env.runner, config=config, issue_number=1, expected_base=env.old.base)


def edit(root):
    (root / 'old').write_text('after\n')


def publish_issue(adapter, text=None):
    text = text or issue_text()
    parsed = validate_structured_issue_implementation(text, allow_editor_handoff=True)
    return adapter.publish(parsed, editor_text=text)


def journal(env, kind='issue', number=1, root=None):
    return HandoffJournal('OWNER/REPO', kind, number, root or env.root)


@pytest.mark.parametrize('crash,phase', [('operator', 'EDITED'), ('push', 'COMMITTED'), ('bind', 'PUSHED')])
def test_issue_crash_then_restart_completes_once(env, crash, phase):
    edit(env.root)
    env.state.crash = crash
    adapter = begin_issue(env, env.root)
    with pytest.raises(Crash):
        publish_issue(adapter)
    assert journal(env).load().phase == phase
    runs_before = env.state.operator_runs
    env.state.crash = None
    # Restart: a different run in a different CLEAN worktree at the same base.
    second = env.fresh('restart')
    adapter = begin_issue(env, second)
    assert adapter.recovered.phase == phase
    assert (second / 'old').read_text() == 'after\n'   # candidate restored from frozen objects
    assert publish_issue(adapter) == 77
    assert env.api.creates == 1
    assert env.state.operator_runs == runs_before + 1
    sha = git(env.origin, 'rev-parse', 'refs/heads/' + adapter.branch).decode()
    assert git(env.origin, 'show', sha + ':old') == b'after'
    assert git(env.origin, 'rev-parse', sha + '^').decode() == env.old.base
    assert adapter.record.phase == 'PR_BOUND'
    adapter.complete()
    assert journal(env, root=second).load() is None


def test_issue_restart_after_pr_bound_reuses_pr_and_branch(env):
    edit(env.root)
    adapter = begin_issue(env, env.root)
    assert publish_issue(adapter) == 77
    assert journal(env).load().phase == 'PR_BOUND'
    branch, remote = adapter.branch, git(env.origin, 'rev-parse', 'refs/heads/' + adapter.branch)
    again = begin_issue(env, env.fresh('restart'))
    assert (again.branch, again.handoff_id) == (branch, adapter.handoff_id)
    assert publish_issue(again) == 77
    assert env.api.creates == 1
    assert git(env.origin, 'rev-parse', 'refs/heads/' + branch) == remote


def test_recovered_commit_must_be_the_journaled_commit(env):
    edit(env.root)
    env.state.crash = 'push'
    with pytest.raises(Crash):
        publish_issue(begin_issue(env, env.root))
    path = journal(env).path
    data = json.loads(path.read_text())
    data['sha'] = '1' * 40
    path.write_text(json.dumps(data))
    env.state.crash = None
    adapter = begin_issue(env, env.fresh('restart'))
    with pytest.raises(HandoffError, match='journaled commit'):
        publish_issue(adapter)
    assert subprocess.run(['git', '-C', str(env.origin), 'rev-parse', '--verify', 'refs/heads/' + adapter.branch],
                          capture_output=True).returncode != 0


def test_tampered_frozen_candidate_object_is_rejected_on_recovery(env):
    edit(env.root)
    env.state.crash = 'operator'
    with pytest.raises(Crash):
        publish_issue(begin_issue(env, env.root))
    record = journal(env).load()
    blob = git(env.root, 'hash-object', '--stdin', data=b'after\n').decode()
    from pathlib import Path
    loose = Path(record.trusted) / 'git' / 'objects' / blob[:2] / blob[2:]
    assert loose.exists()
    loose.chmod(0o644)
    loose.write_bytes(b'not a zlib stream')
    env.state.crash = None
    with pytest.raises(HandoffError):
        begin_issue(env, env.fresh('restart'))


def test_recovery_refuses_dirty_or_moved_base(env):
    edit(env.root)
    env.state.crash = 'operator'
    with pytest.raises(Crash):
        publish_issue(begin_issue(env, env.root))
    env.state.crash = None
    dirty = env.fresh('dirty')
    (dirty / 'old').write_text('somebody else\n')
    with pytest.raises(HandoffError, match='clean checkout'):
        begin_issue(env, dirty)
    git(env.repo, 'commit', '--allow-empty', '-qm', 'base moved')
    moved_base = git(env.repo, 'rev-parse', 'HEAD').decode()
    moved = env.fresh('moved')
    git(moved, 'checkout', '-q', '--detach', moved_base)
    env.claim('issue', 1)
    with pytest.raises(HandoffError, match='different base'):
        split.SplitIssueHandoff.begin(env.runner, config=env.configured(moved), issue_number=1,
                                     expected_base=moved_base)


def test_journal_is_host_private_and_unwritable_by_the_real_codex_sandbox(env):
    edit(env.root)
    env.state.crash = 'operator'
    with pytest.raises(Crash):
        publish_issue(begin_issue(env, env.root))
    path = journal(env).path
    assert path.is_relative_to(state_base()) and (path.stat().st_mode & 0o077) == 0
    before = path.read_bytes()
    for attack in (f"echo '{{}}' > {path}", f"rm -f {path}", f"mv {path} {path}.x"):
        denied = sandboxed(env.root, attack)
        assert denied.returncode != 0 and ('Read-only' in denied.stderr or 'Permission denied' in denied.stderr)
    assert path.read_bytes() == before


def test_corrupt_or_foreign_journal_fails_closed(env):
    edit(env.root)
    env.state.crash = 'operator'
    with pytest.raises(Crash):
        publish_issue(begin_issue(env, env.root))
    path = journal(env).path
    good = path.read_text()
    for bad in ('{', json.dumps({**json.loads(good), 'phase': 'NOPE'}),
                json.dumps({**json.loads(good), 'branch': '../escape'}),
                json.dumps({**json.loads(good), 'number': 2}),
                json.dumps({**json.loads(good), 'phase': 'PUSHED'})):
        path.write_text(bad)
        with pytest.raises(HandoffError):
            journal(env).load()
    path.write_text(good)
    path.chmod(0o666)
    with pytest.raises(HandoffError, match='Unsafe'):
        journal(env).load()


def test_phase_cannot_move_backwards(env):
    edit(env.root)
    adapter = begin_issue(env, env.root)
    publish_issue(adapter)
    with pytest.raises(HandoffError, match='earlier phase'):
        journal(env).write(adapter.record.at('EDITED', sha=None, pr_number=None))


# ----------------------------------------------------------------- PR correction


def begin_pr(env, root, head=None):
    env.claim('pr', 77)
    metadata = SimpleNamespace(number=77, repo=env.config.repo, head_branch=BRANCH, base_branch='main',
                               head_sha=head or env.old.base)
    return split.SplitPrHandoff.begin(env.runner, config=env.configured(root), pr_number=77, metadata=metadata)


def publish_pr(adapter, text=None):
    text = text or pr_text()
    parsed = validate_structured_coder_followup(text, allow_editor_handoff=True)
    from coding_review_agent_loop.local_test_evidence import LocalTestObservation
    receipt = (LocalTestObservation(command=('python', '-m', 'pytest', 'tests'), outcome='passed',
        provenance='parent-observed', cwd=str(adapter.handoff.checkout),
        normalized_command='python -m pytest tests'),) if adapter.recovered is None else ()
    return adapter.publish(parsed, head_change_expected=True, test_observations=receipt, editor_text=text)


@pytest.mark.parametrize('crash,phase', [('operator', 'EDITED'), ('push', 'COMMITTED')])
def test_pr_correction_crash_then_restart_completes_on_same_pr(env, crash, phase):
    edit(env.root)
    env.state.crash = crash
    with pytest.raises(Crash):
        publish_pr(begin_pr(env, env.root))
    assert journal(env, 'pr', 77).load().phase == phase
    env.state.crash = None
    second = env.fresh('restart')
    adapter = begin_pr(env, second)
    assert adapter.recovered.phase == phase and (second / 'old').read_text() == 'after\n'
    sha = publish_pr(adapter)
    assert env.api.creates == 0, 'a correction must never create a PR'
    assert git(env.origin, 'rev-parse', 'refs/heads/' + BRANCH).decode() == sha
    assert git(env.origin, 'rev-parse', sha + '^').decode() == env.old.base
    assert adapter.record.phase == 'PUSHED'
    adapter.complete()
    assert journal(env, 'pr', 77, second).load() is None


def test_pr_restart_after_push_retires_journal_and_does_not_republish(env):
    edit(env.root)
    adapter = begin_pr(env, env.root)
    sha = publish_pr(adapter)
    assert journal(env, 'pr', 77).load().phase == 'PUSHED'
    # Crash before complete(): the PR head is already the journaled commit.
    second = env.fresh('restart')
    git(second, 'fetch', '-q', str(env.origin), 'refs/heads/' + BRANCH)
    git(second, 'checkout', '-q', '--detach', sha)
    env.claim('pr', 77)
    metadata = SimpleNamespace(number=77, repo=env.config.repo, head_branch=BRANCH, base_branch='main', head_sha=sha)
    fresh = split.SplitPrHandoff.begin(env.runner, config=env.configured(second), pr_number=77, metadata=metadata)
    assert fresh.recovered is None
    assert journal(env, 'pr', 77, second).load() is None


def test_pr_journal_for_unrelated_head_is_refused(env):
    edit(env.root)
    env.state.crash = 'operator'
    with pytest.raises(Crash):
        publish_pr(begin_pr(env, env.root))
    env.state.crash = None
    git(env.repo, 'commit', '--allow-empty', '-qm', 'someone pushed')
    other = git(env.repo, 'rev-parse', 'HEAD').decode()
    git(env.repo, 'push', '-q', str(env.origin), f'{other}:refs/heads/{BRANCH}', '--force')
    second = env.fresh('moved')
    git(second, 'checkout', '-q', '--detach', other)
    with pytest.raises(HandoffError, match='different base'):
        begin_pr(env, second, head=other)
