"""Host-only publication against disposable local origins and stub GitHub."""
import json
import shutil

import pytest

from test_git_handoff import case as case, git, operator_commit
from coding_review_agent_loop.git_handoff import HandoffError
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.git_publication import publish_verified, bind_ordinary_pr


def candidate(case):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
    return h, operator_commit(h)


def test_pinned_absent_push_and_retry(case, tmp_path):
    h, sha = candidate(case)
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    args = dict(repo='owner/repo', origin=str(origin), branch='issue-1',
                expected_head=None, local_origin=origin)
    assert publish_verified(h, sha, **args) == sha
    assert publish_verified(h, sha, **args) == sha
    assert git(origin, 'rev-parse', 'refs/heads/issue-1').decode() == sha
    # The application-facing seam never advances the checkout's own branch.
    assert git(h.checkout, 'rev-parse', 'HEAD').decode() == h.base


def test_foreign_head_and_invalid_candidate_never_push(case, tmp_path):
    h, sha = candidate(case)
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    git(h.checkout, 'push', str(origin), f'{h.base}:refs/heads/issue-1')
    with pytest.raises(HandoffError, match='foreign'):
        publish_verified(h, sha, repo='owner/repo', origin=str(origin), branch='issue-1',
                         expected_head=None, local_origin=origin)
    with pytest.raises(HandoffError):
        publish_verified(h, operator_commit(h, tree=h.base_tree), repo='owner/repo',
                         origin=str(origin), branch='new', expected_head=None, local_origin=origin)
    assert git(origin, 'rev-parse', 'refs/heads/issue-1').decode() == h.base


def test_correction_and_unsafe_endpoint(case, tmp_path):
    h, sha = candidate(case)
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    git(h.checkout, 'push', str(origin), f'{h.base}:refs/heads/fix')
    assert publish_verified(h, sha, repo='owner/repo', origin=str(origin), branch='fix',
                            expected_head=h.base, local_origin=origin) == sha
    with pytest.raises(AgentLoopError, match='endpoint'):
        publish_verified(h, sha, repo='owner/repo', origin='ext::sh -c bad',
                         branch='fix', expected_head=h.base)


class GithubStub:
    def __init__(self):
        self.prs = []
        self.creates = 0
    def __call__(self, method, endpoint, payload=None):
        if endpoint == 'user':
            return {'id': 42}
        if method == 'GET' and '/pulls?' in endpoint:
            return self.prs
        if method == 'POST':
            self.creates += 1
            pr = dict(number=7, state='open', body=payload['body'], user={'id': 42},
                      head=dict(sha='a' * 40, ref='issue-1', repo=dict(full_name='owner/repo')),
                      base=dict(ref='main', repo=dict(full_name='owner/repo')))
            self.prs = [pr]
            return pr
        return self.prs[0]


def test_ordinary_pr_retry_has_no_waiver_or_duplicate():
    api = GithubStub()
    args = dict(repo='owner/repo', branch='issue-1', base='main', sha='a' * 40,
                issue_ids=(1,), title='Fix issue', body='Implemented.', handoff_id='run-1', api=api)
    assert bind_ordinary_pr(**args) == 7
    assert bind_ordinary_pr(**args) == 7
    assert api.creates == 1
    assert 'Closes #1' in api.prs[0]['body']
    assert 'MANAGED' not in json.dumps(api.prs)
    api.prs[0]['head']['sha'] = 'b' * 40
    with pytest.raises(HandoffError):
        bind_ordinary_pr(**args)


def test_duplicate_and_foreign_pr_block():
    api = GithubStub()
    args = dict(repo='owner/repo', branch='issue-1', base='main', sha='a' * 40,
                issue_ids=(1,), title='Fix issue', body='', handoff_id='run-1', api=api)
    bind_ordinary_pr(**args)
    api.prs *= 2
    with pytest.raises(HandoffError):
        bind_ordinary_pr(**args)


def test_lease_race_cannot_overwrite_foreign_update(case, tmp_path, monkeypatch):
    from coding_review_agent_loop import git_transport as transport
    h, sha = candidate(case)
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    original = transport._private_git
    def racing(args, **kwargs):
        if args[0] == 'push':
            git(h.checkout, 'push', str(origin), f'{h.base}:refs/heads/issue-1')
        return original(args, **kwargs)
    monkeypatch.setattr(transport, '_private_git', racing)
    with pytest.raises(transport.TransportFailure):
        publish_verified(h, sha, repo='owner/repo', origin=str(origin), branch='issue-1',
                         expected_head=None, local_origin=origin)
    assert git(origin, 'rev-parse', 'refs/heads/issue-1').decode() == h.base


def test_missing_host_auth_blocks_before_network(case, monkeypatch):
    from coding_review_agent_loop import git_transport as transport
    h, sha = candidate(case)
    monkeypatch.setattr(transport, '_private_git', lambda *a, **k: pytest.fail('network must not run'))
    with pytest.raises(HandoffError, match='authentication'):
        publish_verified(h, sha, repo='owner/repo', origin='https://github.com/owner/repo.git',
                         branch='issue-1', expected_head=None)


def test_pr_create_timeout_reconciles_and_wrong_contract_blocks():
    from coding_review_agent_loop.errors import AgentLoopError
    class Interrupted(GithubStub):
        def __call__(self, method, endpoint, payload=None):
            value = super().__call__(method, endpoint, payload)
            if method == 'POST':
                raise AgentLoopError('synthetic uncertain write')
            return value
    api = Interrupted()
    args = dict(repo='owner/repo', branch='issue-1', base='main', sha='a' * 40,
                issue_ids=(1,), title='Fix issue', body='', handoff_id='run-1', api=api)
    assert bind_ordinary_pr(**args) == 7
    assert api.creates == 1
    for field, value in [('body', 'Closes #2\nReviewLoop handoff: run-1\n'), ('state', 'closed')]:
        saved = api.prs[0][field]
        api.prs[0][field] = value
        with pytest.raises(HandoffError):
            bind_ordinary_pr(**args)
        api.prs[0][field] = saved


def test_unsafe_config_hooks_and_filters_never_execute(case, tmp_path):
    from coding_review_agent_loop.git_handoff import Handoff, CommitIdentity
    repo, root, old = case
    canary = tmp_path / 'executed'
    git(repo, 'config', 'core.fsmonitor', f'touch {canary}')
    git(repo, 'config', 'filter.canary.clean', f'touch {canary}')
    hook = repo / '.git' / 'hooks' / 'pre-push'
    hook.write_text(f'#!/bin/sh\ntouch {canary}\n')
    hook.chmod(0o755)
    h = Handoff.begin(root, tmp_path / 'new-trusted',
                      CommitIdentity('ReviewLoop', 'loop@example.invalid', 1700000000, 'Candidate\n'),
                      expected_base=old.base)
    (root / 'old').write_text('after')
    (root / '.gitattributes').write_text('old filter=canary\n')
    h.capture(allowed_new={'.gitattributes'})
    shutil.copytree(h.objects, h.operator_context().objects, dirs_exist_ok=True)
    sha = operator_commit(h)
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    publish_verified(h, sha, repo='owner/repo', origin=str(origin), branch='issue-1',
                     expected_head=None, local_origin=origin)
    assert not canary.exists()


def test_ordinary_gh_adapter_uses_structured_input_and_sanitizes_failures(tmp_path):
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.git_publication import OrdinaryPrApi
    from coding_review_agent_loop.runner import CommandResult
    class RunnerStub:
        fail = False
        def run(self, args, *, cwd, input_text=None, check=True):
            assert args[1:4] == ['api', '--method', 'POST']
            assert args[-2:] == ['--input', '-']
            assert json.loads(input_text) == {'body': 'line one\nline two'}
            return CommandResult(args, cwd, '{}' if not self.fail else '',
                                 'synthetic-private-error', 1 if self.fail else 0)
    runner = RunnerStub()
    api = OrdinaryPrApi(runner, make_config(tmp_path))
    assert api('POST', 'repos/owner/repo/pulls', {'body': 'line one\nline two'}) == {}
    runner.fail = True
    with pytest.raises(HandoffError) as exc:
        api('POST', 'repos/owner/repo/pulls', {'body': 'line one\nline two'})
    assert 'synthetic-private-error' not in str(exc.value)


def test_pr_missing_auth_fails_without_create():
    calls = []
    def api(method, endpoint, payload=None):
        calls.append((method, endpoint))
        return {}
    with pytest.raises(HandoffError, match='authentication'):
        bind_ordinary_pr(repo='owner/repo', branch='issue-1', base='main', sha='a' * 40,
                         issue_ids=(1,), title='Fix', body='', handoff_id='run-1', api=api)
    assert calls == [('GET', 'user')]


def test_token_bearing_push_store_cannot_lie_in_editor_writable_tmp(case, tmp_path, monkeypatch):
    from coding_review_agent_loop import git_publication as publication
    from coding_review_agent_loop.git_handoff import ensure_state_base
    from tempfile import TemporaryDirectory
    h, sha = candidate(case)
    origin = tmp_path / 'origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    seen = []

    def observed_store(*args, **kwargs):
        assert kwargs['dir'] == ensure_state_base(h.checkout)
        assert not kwargs['dir'].is_relative_to('/tmp')
        assert not kwargs['dir'].is_relative_to(h.checkout)
        seen.append(kwargs['dir'])
        return TemporaryDirectory(*args, **kwargs)

    monkeypatch.setattr(publication, 'TemporaryDirectory', observed_store)
    publication.publish_verified(h, sha, repo='owner/repo', origin=str(origin),
                                 branch='verified', expected_head=None, local_origin=origin)
    assert seen
    assert git(origin, 'rev-parse', 'refs/heads/verified').decode() == sha
