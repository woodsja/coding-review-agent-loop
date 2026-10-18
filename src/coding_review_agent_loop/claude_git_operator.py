"""Disposable Linux Claude role. No publication and no implicit auth discovery.

The caller holds the worktree claim from Handoff.begin through verification.
Model authentication is explicit: an in-memory model token OR a read-only
session-file mount. Persistent user homes/settings are never mounted.
prepare() returns a context manager owning pinned file descriptors; run() is
the live entry point. No implicit executable or credential discovery occurs.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
import re
import stat
import subprocess
import sys

from .git_handoff import Handoff, HandoffError
from .secure_git import _trusted_executable


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_canonical(path: Path, *, metadata_only: bool) -> int:
    """Open each component without following symlinks, including ancestors.

    The returned descriptor, rather than a re-resolved host path, is the bind
    source. O_PATH for sessions means host Python never reads their contents.
    """
    if not path.is_absolute() or '..' in path.parts:
        raise OSError('Noncanonical path')
    parent = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        flags = os.O_PATH if metadata_only else os.O_RDONLY | os.O_NONBLOCK
        return os.open(path.name, flags | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    finally:
        os.close(parent)


def _digest(fd: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while True:
        block = os.pread(fd, 1024 * 1024, offset)
        if not block:
            return digest.hexdigest()
        offset += len(block)
        if offset > 512 * 1024 * 1024:
            raise HandoffError('Local Claude executable exceeds attestation bound')
        digest.update(block)


@dataclass
class _PinnedFile:
    path: Path = field(repr=False)
    fd: int
    identity: tuple
    digest: str | None = field(repr=False)
    closed: bool = False

    @classmethod
    def open(cls, path: Path, *, digest: str | None = None):
        label = 'session file' if digest is None else 'executable attestation'
        fd = None
        try:
            fd = _open_canonical(path, metadata_only=digest is None)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or info.st_uid not in (0, os.getuid())
                    or info.st_mode & (0o022 | stat.S_ISUID | stat.S_ISGID)):
                raise ValueError('Unsafe file')
            if digest is None:
                if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 1024 * 1024:
                    raise ValueError('Unsafe session')
            elif (not re.fullmatch(r'[0-9a-f]{64}', digest)
                  or not info.st_mode & 0o111 or os.pread(fd, 4, 0) != b'\x7fELF'
                  or _digest(fd) != digest):
                raise ValueError('Unattested executable')
            pin = cls(path, fd, _file_identity(info), digest)
            pin.verify()
            return pin
        except (OSError, ValueError):
            if fd is not None:
                os.close(fd)
            raise HandoffError(f'Unsafe Claude {label}') from None
        except BaseException:
            if fd is not None:
                os.close(fd)
            raise

    def verify(self):
        try:
            current = _open_canonical(self.path, metadata_only=True)
            try:
                valid = (_file_identity(os.fstat(current)) == self.identity
                         and _file_identity(os.fstat(self.fd)) == self.identity)
                if self.digest is not None:
                    valid = valid and _digest(self.fd) == self.digest
                valid = valid and _file_identity(os.fstat(self.fd)) == self.identity
            finally:
                os.close(current)
            if self.closed or not valid:
                raise ValueError('Changed pin')
        except (OSError, ValueError):
            raise HandoffError('Pinned Claude executable or session file changed') from None

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True


@dataclass(frozen=True)
class OperatorLaunch:
    argv: tuple[str, ...]
    env: dict[str, str] = field(repr=False)
    wrapper: Path
    settings: Path
    pins: tuple[_PinnedFile, ...] = field(default=(), repr=False)

    @property
    def pass_fds(self):
        return tuple(pin.fd for pin in self.pins)

    def verify_pins(self):
        for pin in self.pins:
            pin.verify()

    def close(self):
        for pin in self.pins:
            pin.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# Bind-source descriptors refer to the host mounts, not the read-only
# namespace mounts. Never let the model inherit them (including O_PATH fds).
# This fixed trusted bootstrap uses no shell and closes every non-stdio fd
# before exec, independently of bubblewrap's own descriptor handling.
_EXEC = ('import os, sys; os.closerange(3, os.sysconf("SC_OPEN_MAX")); '
         'os.execv(sys.argv[1], sys.argv[1:])')


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
    def __init__(self, executable: str, *, local_executable_sha256: str | None = None,
                 session_file: Path | None = None):
        # Explicit host attestation only, never model-supplied values. Select
        # the canonical native versions/... file rather than a launcher link.
        self.executable = executable
        self.local_executable_sha256 = local_executable_sha256
        self.session_file = session_file

    def prepare(self, handoff: Handoff, *, auth: dict[str, str] | None = None) -> OperatorLaunch:
        if sys.platform != 'linux':
            raise HandoffError('Claude Git operator requires Linux confinement')
        pins = []
        try:
            return self._prepare(handoff, auth=auth or {}, pins=pins)
        except BaseException:
            for pin in pins:
                pin.close()
            raise

    def _prepare(self, handoff, *, auth, pins):
        allowed_auth = {'ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN'}
        if self.session_file is not None:
            if auth:
                raise HandoffError('Choose one explicit Claude authentication source')
        elif (len(auth) != 1 or not set(auth) <= allowed_auth
                or not all(isinstance(v, str) and v.strip() and '\x00' not in v for v in auth.values())):
            raise HandoffError('Explicit Claude model authentication is required')
        # Later source mounts must not hide the attested executable/session
        # or expose host files in the isolated settings/runtime directory.
        runtime = (Path('/opt/reviewloop'), Path('/tmp/operator-home'))
        if any(root.is_relative_to(reserved) or reserved.is_relative_to(root)
               for root in (handoff.checkout, handoff.common, handoff.trusted)
               for reserved in runtime):
            raise HandoffError('Handoff source mounts overlap the isolated Claude runtime')
        handoff._unchanged()
        private = handoff.trusted.parent / (handoff.trusted.name + '-operator')
        forbidden = (handoff.checkout, handoff.common, handoff.trusted, private)
        executable_pin = None
        if self.local_executable_sha256 is not None:
            if any(Path(self.executable).is_relative_to(p) for p in forbidden):
                raise HandoffError('Attested Claude executable must be outside handoff storage')
            executable_pin = _PinnedFile.open(Path(self.executable), digest=self.local_executable_sha256)
            pins.append(executable_pin)
            claude = '/opt/reviewloop/claude'
        else:
            claude = _trusted_executable(self.executable)
            if not Path(claude).is_relative_to('/usr'):
                raise HandoffError('Claude operator requires a trusted /usr executable or explicit local attestation')
        session_pin = None
        if self.session_file is not None:
            session = Path(self.session_file)
            if session.name != '.credentials.json' or any(session.is_relative_to(p) for p in forbidden):
                raise HandoffError('Claude session file must be outside handoff storage')
            session_pin = _PinnedFile.open(session)
            pins.append(session_pin)
        python = _trusted_executable('/usr/bin/python3')
        git = _trusted_executable('/usr/bin/git')
        bwrap = _trusted_executable('/usr/bin/bwrap')
        if not all(Path(p).is_relative_to('/usr') for p in (python, git, bwrap)):
            raise HandoffError('Claude operator requires a trusted /usr installation')
        ctx = handoff.operator_context()
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
                   '--tmpfs', '/tmp', '--dir', '/tmp/operator-home',
                   '--dir', '/tmp/operator-home/.claude']
        for path in ('/etc/ssl/certs', '/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf'):
            if Path(path).exists():
                mounts += ['--ro-bind', path, path]
        # Descriptor bind sources pin exact inodes across path replacement.
        # Neither the local-install parent nor the real Claude config/home is
        # mounted. Session contents are never read by the host or put in argv.
        if executable_pin is not None:
            mounts += ['--ro-bind', f'/proc/self/fd/{executable_pin.fd}', claude]
        if session_pin is not None:
            mounts += ['--ro-bind', f'/proc/self/fd/{session_pin.fd}',
                       '/tmp/operator-home/.claude/.credentials.json']
        for path in (handoff.checkout, handoff.common, handoff.trusted):
            mounts += ['--ro-bind', str(path), str(path)]
        # Mask project settings, credential sources and registered environment
        # links; their targets are never mounted. Shared config/refs and the
        # checkout's .git pointer stay read-only.
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
        launch = OperatorLaunch(tuple(mounts + [python, '-I', '-S', '-c', _EXEC] + args),
                                env, wrapper, settings, tuple(pins))
        launch.verify_pins()
        return launch

    def run(self, handoff: Handoff, *, auth: dict[str, str] | None = None, timeout: int = 300) -> str:
        with self.prepare(handoff, auth=auth) as launch:
            launch.verify_pins()
            try:
                result = subprocess.run(launch.argv, env=launch.env, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        timeout=timeout, preexec_fn=_limits, pass_fds=launch.pass_fds)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise HandoffError('Claude confined Git operator unavailable') from exc
            finally:
                launch.verify_pins()
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
