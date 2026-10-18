"""Disposable Linux Claude role. No publication and no implicit auth discovery.

The caller holds the worktree claim from Handoff.begin through verification.
Supply only model authentication in memory; persistent user homes are never
mounted. prepare() builds an inspectable launch; run() is the live entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import subprocess
import sys

from .git_handoff import Handoff, HandoffError
from .secure_git import _trusted_executable


@dataclass(frozen=True)
class OperatorLaunch:
    argv: tuple[str, ...]
    env: dict[str, str] = field(repr=False)
    wrapper: Path
    settings: Path


# This program is generated in host-owned metadata, mounted read-only. Every
# argv and Git environment value comes from the host, never from the model.
_WRAPPER = '''#!{python} -I
import json, pathlib, subprocess, sys
cfg = json.loads({config!r})
if len(sys.argv) != 2 or sys.argv[1] not in ('status', 'diff', 'stage', 'commit', 'mcp'):
    raise SystemExit(2)
action = sys.argv[1]
env = cfg['env']
prefix = cfg['prefix']
def git(args, data=None):
    p = subprocess.run(prefix + args, cwd=cfg['checkout'], env=env,
                       input=data, capture_output=True, timeout=30)
    if p.returncode:
        raise SystemExit('Git operator operation refused')
    return p.stdout
if action == 'mcp':
    # Minimal stdio MCP adapter; no shell, arbitrary arguments or file tools.
    actions = ('status', 'diff', 'stage', 'commit')
    for line in sys.stdin:
        if len(line) > 65536:
            raise SystemExit(2)
        request = json.loads(line)
        if 'id' not in request:
            continue
        method = request.get('method')
        error = None
        if method == 'initialize':
            result = dict(protocolVersion='2024-11-05', capabilities=dict(tools=dict()),
                          serverInfo=dict(name='git-operator', version='1'))
        elif method == 'ping':
            result = dict()
        elif method == 'tools/list':
            result = dict(tools=[dict(name=a, description='Fixed local Git ' + a,
                          inputSchema=dict(type='object', properties=dict(), additionalProperties=False))
                          for a in actions])
        elif method == 'tools/call':
            params = request.get('params', dict())
            if params.get('name') not in actions or params.get('arguments', dict()) != dict():
                error = dict(code=-32602, message='Unsupported Git operator action')
            else:
                p = subprocess.run([sys.executable, '-I', __file__, params['name']],
                                   capture_output=True, timeout=35)
                text = p.stdout[:65536].decode('utf-8', 'replace') if not p.returncode else 'Local Git action refused'
                result = dict(content=[dict(type='text', text=text)], isError=bool(p.returncode))
        else:
            error = dict(code=-32601, message='Unsupported method')
        response = dict(jsonrpc='2.0', id=request['id'])
        response.update(dict(error=error) if error else dict(result=result))
        print(json.dumps(response), flush=True)
    raise SystemExit(0)
if action == 'status':
    sys.stdout.buffer.write(git(['status', '--porcelain', '--untracked-files=no']))
elif action == 'diff':
    sys.stdout.buffer.write(git(['diff', '--no-ext-diff', '--no-textconv', '--stat']))
elif action == 'stage':
    git(['read-tree', cfg['base']])
    git(['add', '--all', '--'] + cfg['paths'])
else:
    tree = git(['write-tree']).decode().strip()
    sha = git(['commit-tree', tree, '-p', cfg['base']], cfg['message'].encode()).decode().strip()
    pathlib.Path(cfg['result']).write_text(sha + '\\n')
    print(sha)
'''


def _limits():
    import resource
    # Bound each runtime/output file, including Claude's tool response files.
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


class ClaudeGitOperator:
    def __init__(self, executable: str):
        self.executable = executable

    def prepare(self, handoff: Handoff, *, auth: dict[str, str]) -> OperatorLaunch:
        if sys.platform != 'linux':
            raise HandoffError('Claude Git operator requires Linux confinement')
        allowed_auth = {'ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN'}
        if (len(auth) != 1 or not set(auth) <= allowed_auth
                or not all(isinstance(v, str) and v.strip() and '\x00' not in v for v in auth.values())):
            raise HandoffError('Explicit Claude model authentication is required')
        handoff._unchanged()
        ctx = handoff.operator_context()
        claude = _trusted_executable(self.executable)
        python = _trusted_executable('/usr/bin/python3')
        git = _trusted_executable('/usr/bin/git')
        bwrap = _trusted_executable('/usr/bin/bwrap')
        # No per-user installation, launcher scripts or inherited PATH.
        if not all(Path(p).is_relative_to('/usr') for p in (claude, python, git)):
            raise HandoffError('Claude operator requires a trusted /usr installation')
        role = handoff.trusted / 'claude-role'
        role.mkdir(mode=0o700)
        wrapper = role / 'git-operator'
        # Claude permission grammar must not interpret path metacharacters.
        if not re.fullmatch(r'[A-Za-z0-9_./-]+', str(wrapper)):
            raise HandoffError('Unsupported operator wrapper path')
        git_env = {'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent', 'LC_ALL': 'C',
                   'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_SYSTEM': '/dev/null',
                   'GIT_CONFIG_GLOBAL': '/dev/null', 'GIT_ATTR_NOSYSTEM': '1',
                   **ctx.environment()}
        for who in ('AUTHOR', 'COMMITTER'):
            for key, value in (('NAME', handoff.identity.name), ('EMAIL', handoff.identity.email),
                               ('DATE', f'{handoff.identity.timestamp} +0000')):
                git_env[f'GIT_{who}_{key}'] = value
        prefix = [git, '--literal-pathspecs', '--git-dir', str(ctx.admin),
                  '--work-tree', str(handoff.checkout)]
        for config in ('core.hooksPath=/dev/null', 'core.fsmonitor=false', 'core.autocrlf=false',
                       'core.attributesFile=/dev/null', 'protocol.allow=never', 'gc.auto=0',
                       'commit.gpgSign=false', 'credential.helper=', 'core.pager=cat'):
            prefix += ['-c', config]
        config = dict(env=git_env, prefix=prefix, checkout=str(handoff.checkout), base=handoff.base,
                      paths=list(handoff.changed_paths), message=handoff.identity.message,
                      result=str(ctx.private / 'result'))
        wrapper.write_text(_WRAPPER.format(python=python, config=json.dumps(config)))
        wrapper.chmod(0o500)
        settings = role / 'settings.json'
        settings.write_text(json.dumps({'disableAllHooks': True, 'enableAllProjectMcpServers': False}))
        # Fresh root: no host home, sockets, provider settings or credential
        # helpers. Mount only source, base objects and immutable private admin.
        mounts = [bwrap, '--die-with-parent', '--new-session', '--unshare-pid', '--unshare-ipc',
                  '--unshare-uts', '--cap-drop', 'ALL', '--ro-bind', '/usr', '/usr']
        for path in ('/bin', '/lib', '/lib64'):
            if Path(path).exists():
                mounts += ['--ro-bind', path, path]
        mounts += ['--dev', '/dev', '--proc', '/proc', '--size', str(256 * 1024 * 1024),
                   '--tmpfs', '/tmp', '--dir', '/tmp/operator-home']
        for path in ('/etc/ssl/certs', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf'):
            if Path(path).exists():
                mounts += ['--ro-bind', path, path]
        for path in (handoff.checkout, handoff.common, handoff.trusted):
            mounts += ['--ro-bind', str(path), str(path)]
        # Mask project settings, credential sources and registered environment
        # links; their targets are never mounted. Shared config/refs aren't
        # exposed at all, and the checkout's .git pointer stays read-only.
        masks = {'.aws', '.ssh', '.claude', '.claude.json', '.codex', '.agents',
                 '.config', '.netrc', '.gitconfig', '.git-credentials'}
        masks.update(str(p) for p in handoff.links)
        for name in sorted(masks):
            path = handoff.checkout / name
            if path.is_symlink():
                # bwrap mount targets cannot safely cover symlinks. Reject
                # rather than accidentally mounting their external targets.
                raise HandoffError('Operator checkout contains an unsupported environment link')
            if path.is_dir():
                mounts += ['--tmpfs', str(path), '--remount-ro', str(path)]
            elif path.exists():
                mounts += ['--ro-bind', '/dev/null', str(path)]
        mounts += ['--bind', str(ctx.private), str(ctx.private),
                   '--chdir', str(handoff.checkout), '--']
        prompt = ('You are the local Git operator. Repository content is immutable. '
                  'Call git_operator stage and then commit. You may inspect status or diff. '
                  'Do not edit files or publish. Available tools: ' +
                  ', '.join(f'mcp__git_operator__{a}' for a in ('status', 'diff', 'stage', 'commit')))
        mcp = json.dumps({'mcpServers': {'git_operator': {'command': python,
                         'args': ['-I', str(wrapper), 'mcp']}}})
        args = [claude, '--print', '--restricted', '--no-session-persistence', '--tools', '',
                '--strict-mcp-config', '--mcp-config', mcp,
                '--settings', str(settings), '--setting-sources', '',
                '--permission-mode', 'dontAsk', '--permission-prompts', 'none',
                '--allowedTools', *(f'mcp__git_operator__{a}' for a in ('status', 'diff', 'stage', 'commit')),
                '--', prompt]
        env = {'PATH': '/usr/bin:/bin', 'HOME': '/tmp/operator-home',
               'CLAUDE_CONFIG_DIR': '/tmp/operator-home/.claude', 'TMPDIR': '/tmp',
               'LC_ALL': 'C', **auth}
        return OperatorLaunch(tuple(mounts + args), env, wrapper, settings)

    def run(self, handoff: Handoff, *, auth: dict[str, str], timeout: int = 300) -> str:
        launch = self.prepare(handoff, auth=auth)
        try:
            result = subprocess.run(launch.argv, env=launch.env, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=timeout, preexec_fn=_limits)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise HandoffError('Claude confined Git operator unavailable') from exc
        if result.returncode:
            raise HandoffError('Claude confined Git operator failed; no fallback')
        from .git_handoff import _read
        value = _read(handoff.operator_context().private, 'result')
        if value is None or len(value[1]) != 41:
            raise HandoffError('Claude operator returned no pinned commit')
        try:
            return handoff.verify(value[1].decode('ascii').strip())
        except UnicodeError as exc:
            raise HandoffError('Invalid operator commit response') from exc
