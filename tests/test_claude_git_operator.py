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
    h.operator_context()  # Host preparation precedes the isolated launch/read probe.
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


def local_claude(tmp_path):
    """Synthetic native install, never the host Claude executable or login."""
    import hashlib
    binary = tmp_path / 'home' / '.local' / 'share' / 'claude' / 'versions' / 'synthetic'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'\x7fELFsynthetic-test-cli')
    binary.chmod(0o700)
    return binary, hashlib.sha256(binary.read_bytes()).hexdigest()


def test_attested_local_install_and_readonly_session_mount(case, tmp_path, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, digest = local_claude(tmp_path)
    session = tmp_path / 'login' / '.credentials.json'
    session.parent.mkdir()
    session.write_text('synthetic-session-not-a-token')
    session.chmod(0o600)
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    operator = ClaudeGitOperator(str(binary), local_executable_sha256=digest, session_file=session)
    with operator.prepare(h) as launch:
        assert launch.env['CLAUDE_CONFIG_DIR'] == '/tmp/operator-home/.claude'
        assert set(launch.env) == {'PATH', 'HOME', 'CLAUDE_CONFIG_DIR', 'TMPDIR', 'LC_ALL'}
        assert str(binary.parent) not in launch.argv
        assert str(session.parent) not in launch.argv
        assert 'synthetic-session-not-a-token' not in repr(launch)
        assert '/opt/reviewloop/claude' in launch.argv
        assert '/tmp/operator-home/.claude/.credentials.json' in launch.argv
        assert len(launch.pass_fds) == 2
        for fd in launch.pass_fds:
            assert f'/proc/self/fd/{fd}' in launch.argv
        launch.verify_pins()
    import os
    for fd in launch.pass_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize('mutation', ['digest', 'symlink', 'writable', 'script'])
def test_local_executable_attestation_refuses_unsafe_install(case, tmp_path, monkeypatch, mutation):
    import hashlib
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, digest = local_claude(tmp_path)
    if mutation == 'digest':
        digest = '0' * 64
    elif mutation == 'symlink':
        link = binary.parent / 'alias'
        link.symlink_to(binary.name)
        binary = link
    elif mutation == 'writable':
        binary.chmod(0o777)
    else:
        binary.write_text('#!/bin/sh\nexit 0\n')
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(HandoffError, match='attest|executable'):
        ClaudeGitOperator(str(binary), local_executable_sha256=digest).prepare(
            h, auth={'ANTHROPIC_API_KEY': 'synthetic'})


@pytest.mark.parametrize('replace', [False, True])
def test_executable_pin_detects_changes_after_preparation(case, tmp_path, monkeypatch, replace):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, digest = local_claude(tmp_path)
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    with ClaudeGitOperator(str(binary), local_executable_sha256=digest).prepare(
            h, auth={'ANTHROPIC_API_KEY': 'synthetic'}) as launch:
        if replace:
            binary.unlink()
        binary.write_bytes(b'\x7fELFchanged-executable')
        binary.chmod(0o700)
        with pytest.raises(HandoffError, match='changed'):
            launch.verify_pins()


@pytest.mark.parametrize('unsafe', ['symlink', 'public', 'directory', 'hardlink'])
def test_session_mount_refuses_unsafe_files(case, tmp_path, monkeypatch, unsafe):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    session = tmp_path / '.credentials.json'
    session.write_text('synthetic')
    session.chmod(0o600)
    if unsafe == 'symlink':
        session.rename(tmp_path / 'actual')
        session.symlink_to('actual')
    elif unsafe == 'public':
        session.chmod(0o644)
    elif unsafe == 'directory':
        session.unlink()
        session.mkdir()
    else:
        import os
        os.link(session, tmp_path / 'alias')
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    with pytest.raises(HandoffError, match='session'):
        ClaudeGitOperator('/usr/bin/claude', session_file=session).prepare(h)


def test_attested_executable_inside_editor_checkout_is_refused(case, monkeypatch):
    import hashlib
    _, root, h = case
    binary = root / 'claude'
    binary.write_bytes(b'\x7fELFsynthetic')
    binary.chmod(0o700)
    (root / 'old').write_text('after')
    h.capture(allowed_new={'claude'})
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    with pytest.raises(HandoffError, match='outside'):
        ClaudeGitOperator(str(binary), local_executable_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()).prepare(
            h, auth={'ANTHROPIC_API_KEY': 'synthetic'})


def test_session_pin_never_reads_session_contents(case, tmp_path, monkeypatch):
    import os
    from coding_review_agent_loop import claude_git_operator as module
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    h.operator_context()  # Host preparation precedes the isolated launch/read probe.
    session = tmp_path / '.credentials.json'
    session.write_text('synthetic')
    session.chmod(0o600)
    monkeypatch.setattr(module, '_trusted_executable', str)
    # O_PATH is not readable; even accidental host reads must fail. Guard the
    # Python read APIs too so a future implementation cannot extract tokens.
    def refuse(*args, **kwargs):
        pytest.fail('session contents must not be read by host Python')
    monkeypatch.setattr(os, 'read', refuse)
    monkeypatch.setattr(os, 'pread', refuse)
    # _unchanged reads the worktree pointer; it is already checked above and
    # separately tested. This test isolates session preparation/verification.
    monkeypatch.setattr(h, '_unchanged', lambda: None)
    with ClaudeGitOperator('/usr/bin/claude', session_file=session).prepare(h) as launch:
        launch.verify_pins()


def test_local_install_is_never_implicitly_trusted(case, tmp_path):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, _ = local_claude(tmp_path)
    from coding_review_agent_loop.errors import AgentLoopError
    with pytest.raises(AgentLoopError, match='root-owned'):
        ClaudeGitOperator(str(binary)).prepare(h, auth={'ANTHROPIC_API_KEY': 'synthetic'})


def test_run_checks_post_launch_pin_before_raw_commit_verification(case, tmp_path, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    h.operator_context()  # Host preparation precedes the isolated launch/read probe.
    binary, digest = local_claude(tmp_path)
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    seen_fds = []
    def changed(argv, **kwargs):
        seen_fds.extend(kwargs['pass_fds'])
        binary.write_bytes(b'\x7fELFtampered')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, 'run', changed)
    monkeypatch.setattr(h, 'verify', lambda sha: pytest.fail('changed pin must stop verification'))
    with pytest.raises(HandoffError, match='changed'):
        ClaudeGitOperator(str(binary), local_executable_sha256=digest).run(h, auth={'ANTHROPIC_API_KEY': 'synthetic'})
    import os
    assert seen_fds
    for fd in seen_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_session_replacement_during_launch_is_refused(case, tmp_path, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    h.operator_context()  # Host preparation precedes the isolated launch/read probe.
    session = tmp_path / '.credentials.json'
    session.write_text('synthetic')
    session.chmod(0o600)
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    def replaced(argv, **kwargs):
        session.unlink()
        session.write_text('synthetic-other')
        session.chmod(0o600)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, 'run', replaced)
    with pytest.raises(HandoffError, match='changed'):
        ClaudeGitOperator('/usr/bin/claude', session_file=session).run(h)


def test_attested_operator_commit_through_verified_local_publication(case, tmp_path, monkeypatch):
    """Internal vertical slice; no workflow/reviewer integration claim.

    Fake ONLY the model/confinement launch, executing the real fixed wrapper,
    raw verification and host publication against a disposable bare origin.
    The separate OS test covers mounts when namespaces are available.
    """
    from coding_review_agent_loop.git_publication import publish_verified, bind_ordinary_pr
    from test_git_publication import GithubStub
    _, root, h = case
    (root / 'old').write_text('ready for publication')
    (root / 'new').write_text('declared new file')
    h.capture(allowed_new={'new'})
    binary, digest = local_claude(tmp_path)
    session = tmp_path / '.credentials.json'
    session.write_text('synthetic')
    session.chmod(0o600)
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    real_run = subprocess.run
    launches = []
    def fake_model(argv, **kwargs):
        if argv[0] != '/usr/bin/bwrap':
            return real_run(argv, **kwargs)
        launches.append(argv)
        cfg = json.loads(argv[argv.index('--mcp-config') + 1])['mcpServers']['git_operator']
        wrapper = cfg['args'][1]
        for action in ('stage', 'commit'):
            real_run([cfg['command'], '-I', wrapper, action], capture_output=True, check=True)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, 'run', fake_model)
    sha = ClaudeGitOperator(str(binary), local_executable_sha256=digest, session_file=session).run(h)
    assert len(launches) == 1
    assert h.verify(sha) == sha
    assert git(root, 'rev-parse', 'HEAD').decode() == h.base
    origin = tmp_path / 'bare-origin'
    origin.mkdir()
    git(origin, 'init', '--bare', '-q')
    publish_verified(h, sha, repo='owner/repo', origin=str(origin), branch='issue-1',
                     expected_head=None, local_origin=origin)
    assert git(origin, 'rev-parse', 'refs/heads/issue-1').decode() == sha
    assert git(origin, 'show', f'{sha}:new') == b'declared new file'
    class Api(GithubStub):
        def __call__(self, method, endpoint, payload=None):
            result = super().__call__(method, endpoint, payload)
            if method == 'POST':
                result['head']['sha'] = sha
            return result
    api = Api()
    args = dict(repo='owner/repo', branch='issue-1', base='main', sha=sha, issue_ids=(1,),
                title='Fix issue', body='Code ready.', handoff_id='synthetic-1', api=api)
    assert bind_ordinary_pr(**args) == 7
    assert bind_ordinary_pr(**args) == 7
    assert api.creates == 1
    assert git(root, 'rev-parse', 'HEAD').decode() == h.base


def test_live_attested_inode_and_readonly_session_in_private_runtime(case, tmp_path, monkeypatch):
    import hashlib
    import os
    import shutil
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, _ = local_claude(tmp_path)
    # A synthetic local installation of a harmless native executable. This
    # probes exec/bind behavior without starting Claude or authenticating.
    shutil.copyfile('/usr/bin/true', binary)
    binary.chmod(0o700)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    home = tmp_path / 'host-home'
    session = home / '.claude' / '.credentials.json'
    session.parent.mkdir(parents=True)
    session.write_text('synthetic-session')
    session.chmod(0o600)
    (session.parent / 'settings.json').write_text('host-settings-canary')
    (home / '.ssh').mkdir()
    (home / '.ssh' / 'id_ed25519').write_text('synthetic-ssh-canary')
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    script = '''
import pathlib, subprocess, sys
session = pathlib.Path('/tmp/operator-home/.claude/.credentials.json')
assert session.read_text() == 'synthetic-session'
for path in (session, pathlib.Path('/opt/reviewloop/claude')):
    for operation in (lambda: path.write_text('bad'), lambda: path.chmod(0o777), lambda: path.unlink()):
        try: operation()
        except OSError: pass
        else: raise SystemExit('readonly mount was writable')
for path in sys.argv[1:]:
    assert not pathlib.Path(path).exists()
assert not pathlib.Path('/tmp/operator-home/.claude/settings.json').exists()
pathlib.Path('/tmp/operator-home/.claude/runtime').write_text('scratch only')
assert subprocess.run(['/opt/reviewloop/claude']).returncode == 0
'''
    with ClaudeGitOperator(str(binary), local_executable_sha256=digest, session_file=session).prepare(h) as launch:
        mounts = launch.argv[:launch.argv.index('--') + 1]
        result = subprocess.run([*mounts, '/usr/bin/python3', '-I', '-S', '-c', script,
                                 str(home), str(binary.parent)],
                                capture_output=True, env=launch.env, pass_fds=launch.pass_fds)
        if result.returncode and any(message in result.stderr for message in (
                b'No permissions to create new namespace',
                b'Creating new namespace failed: Operation not permitted')):
            pytest.skip('Host disallows bubblewrap namespaces; no local-install runtime claim')
        assert result.returncode == 0, result.stderr.decode()
        launch.verify_pins()
    assert session.read_text() == 'synthetic-session'
    assert (session.parent / 'settings.json').read_text() == 'host-settings-canary'


def test_failed_preparation_closes_attested_descriptors(case, tmp_path, monkeypatch):
    import os
    from coding_review_agent_loop import claude_git_operator as module
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, digest = local_claude(tmp_path)
    pinned = []
    original_open = module._PinnedFile.open
    def record(path, **kwargs):
        pin = original_open(path, **kwargs)
        pinned.append(pin.fd)
        return pin
    monkeypatch.setattr(module._PinnedFile, 'open', record)
    def unavailable(path):
        raise HandoffError('system runtime unavailable')
    monkeypatch.setattr(module, '_trusted_executable', unavailable)
    with pytest.raises(HandoffError, match='runtime unavailable'):
        ClaudeGitOperator(str(binary), local_executable_sha256=digest).prepare(
            h, auth={'ANTHROPIC_API_KEY': 'synthetic'})
    assert pinned
    for fd in pinned:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_session_auth_cannot_be_combined_with_token(case, tmp_path, monkeypatch):
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(HandoffError, match='one explicit'):
        ClaudeGitOperator('/usr/bin/claude', session_file=tmp_path / '.credentials.json').run(
            h, auth={'ANTHROPIC_API_KEY': 'synthetic'})


def test_model_bootstrap_closes_host_mount_descriptors(case, tmp_path, monkeypatch):
    import os
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, digest = local_claude(tmp_path)
    session = tmp_path / '.credentials.json'
    session.write_text('synthetic')
    session.chmod(0o600)
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    with ClaudeGitOperator(str(binary), local_executable_sha256=digest, session_file=session).prepare(h) as launch:
        command = launch.argv[launch.argv.index('--') + 1:]
        assert command[:4] == ('/usr/bin/python3', '-I', '-S', '-c')
        # Exercise the actual trusted bootstrap outside the mount namespace,
        # substituting a probe for the Claude executable, with synthetic fds.
        probe = '''
import os, sys
for fd in map(int, sys.argv[1:]):
    try: os.fstat(fd)
    except OSError: pass
    else: raise SystemExit('Host bind descriptor leaked into model process')
'''
        result = subprocess.run([*command[:5], '/usr/bin/python3', '-I', '-S', '-c', probe,
                                 *(str(fd) for fd in launch.pass_fds)],
                                env=launch.env, pass_fds=launch.pass_fds, capture_output=True)
        assert result.returncode == 0, result.stderr.decode()
        for fd in launch.pass_fds:
            os.fstat(fd)  # Only the child descriptors were closed.


@pytest.mark.parametrize('source_field,source_path', [
    ('checkout', '/opt'), ('common', '/opt/reviewloop'),
    ('trusted', '/tmp'), ('checkout', '/tmp/operator-home'),
])
def test_source_mount_cannot_cover_attested_runtime(case, tmp_path, monkeypatch, source_field, source_path):
    from pathlib import Path
    _, root, h = case
    (root / 'old').write_text('after')
    h.capture(allowed_new=set())
    binary, digest = local_claude(tmp_path)
    # Isolate mount-layout validation: no host source or credential access.
    monkeypatch.setattr(h, source_field, Path(source_path))
    monkeypatch.setattr(h, '_unchanged', lambda: None)
    monkeypatch.setattr('coding_review_agent_loop.claude_git_operator._trusted_executable', str)
    with pytest.raises(HandoffError, match='overlap'):
        ClaudeGitOperator(str(binary), local_executable_sha256=digest).prepare(
            h, auth={'ANTHROPIC_API_KEY': 'synthetic'})
