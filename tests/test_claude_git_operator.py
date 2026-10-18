"""Synthetic role tests: never launch Claude or use provider credentials."""
import json
import subprocess

import pytest

from test_git_handoff import case as case, git
from coding_review_agent_loop.git_handoff import HandoffError
from coding_review_agent_loop.claude_git_operator import ClaudeGitOperator


def test_missing_auth_fails_before_launch(case, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(HandoffError, match='authentication'):
        ClaudeGitOperator('/usr/bin/claude').run(h, auth={})


def test_role_layout_and_fixed_wrapper(case, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    launch = ClaudeGitOperator('/usr/bin/claude').prepare(h, auth={'ANTHROPIC_API_KEY': 'synthetic'})
    assert '--unshare-net' not in launch.argv
    assert '--restricted' in launch.argv and '--no-session-persistence' in launch.argv
    assert launch.argv[launch.argv.index('--tools') + 1] == ''
    assert launch.env.get('GH_TOKEN') is None
    assert launch.env['HOME'] == '/tmp/operator-home'
    assert '--ro-bind' in launch.argv and '--unshare-pid' in launch.argv
    assert '--ro-bind' in launch.argv[:launch.argv.index('--')]
    wrapper = launch.wrapper
    # Execute only the trusted wrapper on the synthetic checkout, outside the
    # model. The actual mount enforcement has a separate OS test.
    for action in ('stage', 'commit'):
        result = subprocess.run([str(wrapper), action], capture_output=True)
        assert result.returncode == 0, result.stderr
    sha = (h.operator_context().private / 'result').read_text().strip()
    assert h.verify(sha) == sha
    assert git(root, 'rev-parse', 'HEAD').decode() == h.base
    for args in [('push',), ('stage', 'old'), ('config',), ('commit', ';', 'touch', 'canary')]:
        assert subprocess.run([str(wrapper), *args], capture_output=True).returncode != 0
    settings = json.loads(launch.settings.read_text())
    assert settings['disableAllHooks'] is True


def test_failed_confinement_has_no_fallback(case, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    calls = []
    def failed(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 1, b'', b'bwrap: denied')
    monkeypatch.setattr(subprocess, 'run', failed)
    with pytest.raises(HandoffError, match='confined'):
        ClaudeGitOperator('/usr/bin/claude').run(h, auth={'ANTHROPIC_API_KEY': 'synthetic'})
    assert len(calls) == 1


def test_mcp_only_accepts_fixed_zero_argument_tools(case, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    launch = ClaudeGitOperator('/usr/bin/claude').prepare(h, auth={'ANTHROPIC_API_KEY': 'synthetic'})
    messages = [dict(jsonrpc='2.0', id=1, method='initialize'),
                dict(jsonrpc='2.0', id=2, method='tools/list'),
                dict(jsonrpc='2.0', id=3, method='tools/call', params={'name': 'push'}),
                dict(jsonrpc='2.0', id=4, method='tools/call', params={'name': 'stage', 'arguments': {'path': 'old'}}),
                dict(jsonrpc='2.0', id=5, method='tools/call', params={'name': 'stage'}),
                dict(jsonrpc='2.0', id=6, method='tools/call', params={'name': 'commit'})]
    result = subprocess.run([str(launch.wrapper), 'mcp'], input=''.join(json.dumps(m) + '\n' for m in messages),
                            capture_output=True, text=True, check=True)
    replies = [json.loads(line) for line in result.stdout.splitlines()]
    assert {t['name'] for t in replies[1]['result']['tools']} == {'status', 'diff', 'stage', 'commit'}
    assert 'error' in replies[2] and 'error' in replies[3]
    assert not replies[4]['result']['isError'] and not replies[5]['result']['isError']
    assert h.verify((h.operator_context().private / 'result').read_text().strip())


def test_live_role_mounts_deny_source_shared_git_and_host_home(case, monkeypatch):
    # Runs Python in the exact role mount layout; does not invoke Claude.
    repo, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    launch = ClaudeGitOperator('/usr/bin/claude').prepare(h, auth={'ANTHROPIC_API_KEY': 'synthetic'})
    mounts = launch.argv[:launch.argv.index('--') + 1]
    script = '''
import pathlib, sys
root, common, admin, private = map(pathlib.Path, sys.argv[1:])
for operation in (
    lambda: (root / 'old').write_text('bad'),
    lambda: (root / 'old').chmod(0o755),
    lambda: (root / 'old').rename(root / 'bad'),
    lambda: (root / 'old').unlink(),
    lambda: (root / 'new').write_text('bad'),
    lambda: (root / 'new').symlink_to('old'),
    lambda: (root / '.git').write_text('bad'),
    lambda: (common / 'config').write_text('bad'),
    lambda: (common / 'objects' / 'bad').write_text('bad'),
    lambda: (admin / 'config').write_text('bad'),
):
    try: operation()
    except OSError: pass
    else: raise SystemExit('write permitted by: ' + str(operation))
assert (common / 'config').is_file()
assert not pathlib.Path('/home/woodsja/.ssh').exists()
(private / 'probe').write_text('ok')
pathlib.Path('/tmp/scratch').write_text('ok')
'''
    result = subprocess.run([*mounts, '/usr/bin/python3', '-I', '-S', '-c', script,
                             str(root), str(repo / '.git'), str(h.admin), str(h.operator_context().private)],
                            capture_output=True, env=launch.env)
    if result.returncode and result.stderr.startswith(b'bwrap:'):
        pytest.skip('Host disallows bubblewrap namespaces; no live role confinement claim')
    assert result.returncode == 0, result.stderr.decode()
