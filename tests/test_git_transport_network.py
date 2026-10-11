"""Real Git transport regressions against local TLS Git HTTP and OpenSSH backends."""

from __future__ import annotations

import base64
import getpass
import os
import socket
import shlex
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import git_transport
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(("git", "-C", str(path), *args), text=True).strip()


@pytest.fixture
def https_git(tmp_path, monkeypatch):
    bare = tmp_path / "OWNER" / "REPO.git"
    bare.parent.mkdir()
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(("git", "init", "-q", "--bare", "-b", "main", str(bare)), check=True)
    subprocess.run(("git", "init", "-q", "-b", "main", str(seed)), check=True)
    (seed / "file.txt").write_text("transport fixture\n")
    _git(seed, "add", "file.txt")
    _git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    _git(seed, "push", "-q", str(bare), "main")
    subprocess.run(("git", "clone", "-q", str(bare), str(checkout)), check=True)
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(("openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                    "-keyout", str(key), "-out", str(cert), "-days", "1",
                    "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost"),
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    requests: list[tuple[str, str | None]] = []
    redirected_requests: list[str | None] = []
    required_auth = {"value": None}
    redirect = {"value": "", "enabled": False}

    class RedirectHandler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            redirected_requests.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()

        do_POST = do_GET

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.serve_git()

        def do_POST(self):
            self.serve_git()

        def serve_git(self):
            auth = self.headers.get("Authorization")
            requests.append((self.path, auth))
            if redirect["enabled"]:
                self.send_response(302)
                self.send_header("Location", redirect["value"])
                self.end_headers()
                return
            if required_auth["value"] and (not auth or auth.lower() != required_auth["value"].lower()):
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Basic realm="Git"')
                self.end_headers()
                return
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            path, _, query = self.path.partition("?")
            env = dict(os.environ)
            env.update({"GIT_PROJECT_ROOT": str(tmp_path), "GIT_HTTP_EXPORT_ALL": "1",
                        "PATH_INFO": path, "QUERY_STRING": query,
                        "REQUEST_METHOD": self.command,
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": str(len(body))})
            result = subprocess.run(("git", "http-backend"), input=body, env=env,
                                    capture_output=True, check=True)
            header, payload = result.stdout.split(b"\r\n\r\n", 1)
            lines = header.decode("latin1").split("\r\n")
            status = 200
            for line in lines:
                if line.lower().startswith("status:"):
                    status = int(line.split(":", 1)[1].strip().split()[0])
            self.send_response(status)
            for line in lines:
                if ":" in line and not line.lower().startswith("status:"):
                    name, value = line.split(":", 1)
                    self.send_header(name, value.strip())
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    redirected_server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    redirected_server.socket = context.wrap_socket(redirected_server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    redirected_thread = threading.Thread(target=redirected_server.serve_forever, daemon=True)
    thread.start()
    redirected_thread.start()
    redirect["value"] = f"https://localhost:{redirected_server.server_port}/redirected"
    url = f"https://localhost:{server.server_port}/OWNER/REPO.git"
    repo = f"localhost:{server.server_port}/OWNER/REPO"
    monkeypatch.setenv("SSL_CERT_FILE", str(cert))
    original_env = git_transport._transport_env

    def fixture_env(endpoint, gh_cmd, **kwargs):
        env = original_env(endpoint, gh_cmd, **kwargs)
        env["GIT_SSL_CAINFO"] = str(cert)
        return env

    monkeypatch.setattr(git_transport, "_transport_env", fixture_env)
    _git(checkout, "config", "remote.origin.url", url)
    try:
        yield checkout, url, repo, requests, redirected_requests, required_auth, redirect
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        redirected_server.shutdown()
        redirected_server.server_close()
        redirected_thread.join(timeout=5)


@pytest.mark.parametrize("mode", ["public", "private", "redirect", "rewrite"])
def test_real_https_transport_policy(https_git, monkeypatch, tmp_path, mode):
    checkout, url, repo, requests, redirected_requests, required_auth, redirect = https_git
    marker = tmp_path / "helper-ran"
    _git(checkout, "config", "credential.helper", f"!touch {marker}; echo password=planted")
    if mode == "rewrite":
        _git(checkout, "config", "url.https://127.0.0.1:1/.insteadOf", url)
    if mode in {"private", "redirect"}:
        token = "fixture-secret-token"
        required_auth["value"] = "Basic " + base64.b64encode(f"x-access-token:{token}".encode()).decode()
        monkeypatch.setattr(git_transport, "_token_from_gh", lambda *_args: token)
    if mode == "redirect":
        redirect["enabled"] = True
    kwargs = {"repo": repo, "runner": Runner(), "gh_cmd": "gh" if mode in {"private", "redirect"} else None}
    if mode == "redirect":
        with pytest.raises(AgentLoopError, match="Trusted Git transport failed"):
            git_transport.import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main", **kwargs)
        assert all(path != "/redirected" for path, _ in requests)
        assert all(path.startswith("/OWNER/REPO.git/") for path, _ in requests)
        assert any(auth and auth.lower() == required_auth["value"].lower() for _, auth in requests)
        assert redirected_requests == []
    else:
        sha = git_transport.import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main", **kwargs)
        assert sha == _git(checkout, "rev-parse", "HEAD")
        assert _git(checkout, "cat-file", "-t", sha) == "commit"
        assert requests
        if mode == "private":
            assert any(auth and auth.lower() == required_auth["value"].lower() for _, auth in requests)
            assert all(path.startswith("/OWNER/REPO.git/") for path, _ in requests)
    assert not marker.exists()


_SSHD = Path("/usr/sbin/sshd")
_SSH = Path("/usr/bin/ssh")
requires_ssh = pytest.mark.skipif(os.name != "posix" or not _SSH.exists(),
                                  reason="OpenSSH client is unavailable")
requires_sshd = pytest.mark.skipif(os.name != "posix" or not _SSHD.exists() or not _SSH.exists(),
                                   reason="local OpenSSH server is unavailable")


def _private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)
    return path


def _ssh_home(tmp_path: Path, monkeypatch) -> Path:
    """A trusted 0700 HOME with no agent socket and no default-named key."""
    tmp_path.chmod(0o700)
    home = _private_dir(tmp_path / "home")
    _private_dir(home / ".ssh")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("SSH_AUTH_SOCK", raising=False)
    monkeypatch.setattr(git_transport, "_EMITTED_WARNINGS", set())
    return home


def _write_private(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o600)
    return path


def _seed_origin(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "OWNER" / "REPO.git"
    bare.parent.mkdir()
    seed = tmp_path / "seed"
    checkout = tmp_path / "checkout"
    subprocess.run(("git", "init", "-q", "--bare", "-b", "main", str(bare)), check=True)
    subprocess.run(("git", "init", "-q", "-b", "main", str(seed)), check=True)
    (seed / "file.txt").write_text("ssh fixture\n")
    _git(seed, "add", "file.txt")
    _git(seed, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    _git(seed, "push", "-q", str(bare), "main")
    subprocess.run(("git", "clone", "-q", str(bare), str(checkout)), check=True)
    return bare, checkout


@pytest.fixture
def ssh_origin(tmp_path, monkeypatch):
    """A local sshd serving one bare repository, plus a trusted operator HOME.

    The client key has a non-default name and is reachable only through
    ~/.ssh/config or the explicit override; there is no agent socket.
    """
    bare, checkout = _seed_origin(tmp_path)
    home = _ssh_home(tmp_path, monkeypatch)
    key = home / ".ssh" / "github_key"
    unauthorized = home / ".ssh" / "unauthorized_key"
    host_key = tmp_path / "host_key"
    for path in (key, unauthorized, host_key):
        subprocess.run(("ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)), check=True)
    forced_command = tmp_path / "serve_git.sh"
    forced_command.write_text(f"#!/bin/sh\nexec /usr/bin/git-upload-pack {bare}\n")
    forced_command.chmod(0o700)
    authorized = tmp_path / "authorized_keys"
    authorized.write_text(f'command="{forced_command}" ' + key.with_suffix(".pub").read_text())
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server_config = tmp_path / "sshd_config"
    server_config.write_text(f"Port {port}\nListenAddress 127.0.0.1\nHostKey {host_key}\n"
                             f"AuthorizedKeysFile {authorized}\nStrictModes no\nUsePAM no\n"
                             "PasswordAuthentication no\nPubkeyAuthentication yes\n"
                             "PermitRootLogin no\nLogLevel QUIET\n")
    host_public = host_key.with_suffix(".pub").read_text().split()[:2]
    known_hosts = _write_private(
        home / ".ssh" / "known_hosts_test",
        f"[localhost]:{port},[127.0.0.1]:{port} {' '.join(host_public)}\n",
    )
    server = subprocess.Popen((str(_SSHD), "-D", "-e", "-f", str(server_config)),
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(50):
            if server.poll() is not None:
                pytest.skip("local sshd could not start")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.02)
        else:
            pytest.skip("local sshd did not accept connections")
        repo = f"localhost:{port}/OWNER/REPO"
        url = git_transport.default_origin(repo, protocol="ssh")
        _git(checkout, "config", "remote.origin.url", url)
        commands: list[str] = []
        original_env = git_transport._transport_env

        def fixture_env(endpoint, gh_cmd, **kwargs):
            env = original_env(endpoint, gh_cmd, **kwargs)
            commands.append(env["GIT_SSH_COMMAND"])
            # Labelled test convenience only: the local sshd authenticates the
            # current OS user rather than GitHub's pinned ``git`` user.  No
            # identity, host-key or known-hosts option is injected here.
            env["GIT_SSH_COMMAND"] += f" -l {getpass.getuser()}"
            return env

        monkeypatch.setattr(git_transport, "_transport_env", fixture_env)
        yield SimpleNamespace(bare=bare, checkout=checkout, home=home, key=key,
                              unauthorized=unauthorized, known_hosts=known_hosts,
                              port=port, repo=repo, url=url, commands=commands,
                              marker=tmp_path / "proxy-ran")
    finally:
        server.terminate()
        server.communicate(timeout=5)


def _hostile_lines(marker: Path) -> str:
    return (f"Host *\n  ProxyCommand touch {marker}\n  LocalCommand touch {marker}\n"
            f'Match exec "touch {marker}"\n  IdentityFile /dev/null\n')


def _assert_no_test_injection(command: str) -> None:
    assert "StrictHostKeyChecking=no" not in command
    assert "UserKnownHostsFile=/dev/null" not in command
    argv = shlex.split(command)
    assert argv[:3] == ["/usr/bin/ssh", "-F", os.devnull]
    strict = argv.index("StrictHostKeyChecking=yes")
    assert all(index > strict for index, token in enumerate(argv) if token == "-i")


@requires_sshd
def test_real_ssh_fetch_uses_config_selected_identity(ssh_origin, tmp_path):
    from agent_loop_helpers import make_config
    from coding_review_agent_loop.config import ensure_temp_checkout

    origin = ssh_origin
    _write_private(origin.home / ".ssh" / "config", (
        "Host localhost\n"
        "  IdentityFile ~/.ssh/github_key\n"
        "  IdentitiesOnly yes\n"
        "  UserKnownHostsFile ~/.ssh/known_hosts_test\n"
        + _hostile_lines(origin.marker)
    ))
    checkout = origin.checkout
    config = make_config(tmp_path, create_dirs=False, repo=origin.repo, claude_dir=checkout,
                         trusted_origin_protocol="ssh")
    checkout.rename(tmp_path / "fixture-clone")
    checkout_marker = tmp_path / "checkout-helper-ran"

    class PlantCheckoutConfig(Runner):
        def run(self, args, *, cwd, **kwargs):
            result = super().run(args, cwd=cwd, **kwargs)
            if tuple(args[:4]) == ("git", "remote", "add", "origin"):
                _git(checkout, "config", "core.sshCommand", f"touch {checkout_marker}")
                _git(checkout, "config", "url.ssh://evil.example/.insteadOf", origin.url)
            return result

    ensure_temp_checkout(checkout, agent="claude", config=config, runner=PlantCheckoutConfig())
    sha = _git(checkout, "rev-parse", "HEAD")
    assert _git(checkout, "config", "--local", "--get", "remote.origin.url") == origin.url
    assert sha == _git(origin.bare, "rev-parse", "refs/heads/main")
    assert (checkout / "file.txt").read_text() == "ssh fixture\n"
    assert origin.commands
    for command in origin.commands:
        _assert_no_test_injection(command)
        assert f"-i {origin.key}" in command
        assert f"UserKnownHostsFile={origin.known_hosts}" in command
    assert not origin.marker.exists() and not checkout_marker.exists()


@requires_sshd
def test_real_ssh_fetch_uses_explicit_identity_override(ssh_origin):
    origin = ssh_origin
    _write_private(origin.home / ".ssh" / "config", (
        "Host localhost\n  UserKnownHostsFile ~/.ssh/known_hosts_test\n" + _hostile_lines(origin.marker)
    ))
    sha = git_transport.import_ref(origin.checkout, "refs/heads/main", "refs/remotes/origin/main",
                                   repo=origin.repo, runner=Runner(), ssh_identity=origin.key)
    assert sha == _git(origin.bare, "rev-parse", "refs/heads/main")
    _assert_no_test_injection(origin.commands[-1])
    assert shlex.split(origin.commands[-1]).count("-i") == 1
    assert "IdentitiesOnly=yes" in origin.commands[-1]
    assert not origin.marker.exists()


@requires_sshd
def test_real_ssh_without_usable_key_reports_authentication_class(ssh_origin):
    origin = ssh_origin
    _write_private(origin.home / ".ssh" / "config", (
        "Host localhost\n  IdentityFile ~/.ssh/unauthorized_key\n  IdentitiesOnly yes\n"
        "  UserKnownHostsFile ~/.ssh/known_hosts_test\n"
    ))
    with pytest.raises(AgentLoopError) as raised:
        git_transport.import_ref(origin.checkout, "refs/heads/main", "refs/remotes/origin/main",
                                 repo=origin.repo, runner=Runner())
    message = str(raised.value)
    assert "(authentication failure)." in message
    assert "--git-ssh-identity" in message
    assert "Permission denied" not in message and str(origin.unauthorized) not in message


@requires_sshd
def test_real_ssh_unknown_host_fails_closed_without_prompt(ssh_origin):
    origin = ssh_origin
    missing = origin.home / ".ssh" / "known_hosts_missing"
    _write_private(origin.home / ".ssh" / "config", (
        "Host localhost\n  IdentityFile ~/.ssh/github_key\n  IdentitiesOnly yes\n"
        "  UserKnownHostsFile ~/.ssh/known_hosts_missing\n"
    ))
    assert not (origin.home / ".ssh" / "known_hosts").exists()
    started = time.monotonic()
    with pytest.raises(AgentLoopError) as raised:
        git_transport.import_ref(origin.checkout, "refs/heads/main", "refs/remotes/origin/main",
                                 repo=origin.repo, runner=Runner())
    assert time.monotonic() - started < 60
    message = str(raised.value)
    assert "(host-key failure)." in message and "known_hosts" in message
    assert "--git-ssh-identity" not in message
    assert not missing.exists()
    assert not (origin.home / ".ssh" / "known_hosts").exists()
    command = origin.commands[-1]
    assert "StrictHostKeyChecking=yes" in command and "BatchMode" not in command


@requires_ssh
def test_real_ssh_connection_refused_reports_network_class(tmp_path, monkeypatch):
    _, checkout = _seed_origin(tmp_path)
    _ssh_home(tmp_path, monkeypatch)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    repo = f"127.0.0.1:{port}/OWNER/REPO"
    _git(checkout, "config", "remote.origin.url", git_transport.default_origin(repo, protocol="ssh"))
    with pytest.raises(AgentLoopError) as raised:
        git_transport.import_ref(checkout, "refs/heads/main", "refs/remotes/origin/main",
                                 repo=repo, runner=Runner())
    message = str(raised.value)
    assert "(network or timeout)." in message
    assert "--git-ssh-identity" not in message and "refused" not in message


@requires_ssh
def test_pinned_ssh_consumes_exactly_the_validated_paths(tmp_path, monkeypatch):
    home = _ssh_home(tmp_path, monkeypatch)
    key = _write_private(home / ".ssh" / "github_key", "placeholder\n")
    known_hosts = _write_private(home / ".ssh" / "known_hosts_github", "")
    _write_private(home / ".ssh" / "config", (
        "Host localhost\n  IdentityFile ~/.ssh/github_key\n"
        "  UserKnownHostsFile ~/.ssh/known_hosts_github\n"
    ))
    command = git_transport._transport_env("ssh://git@localhost:2222/OWNER/REPO.git", None)["GIT_SSH_COMMAND"]
    # Test-only option dump; production code never runs ssh -G.  The
    # generated command carries -F /dev/null, so no user config is evaluated.
    result = subprocess.run((*shlex.split(command), "-G", "localhost"), capture_output=True,
                            text=True, check=True, stdin=subprocess.DEVNULL, timeout=30)
    values: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        name, _, value = line.partition(" ")
        values.setdefault(name, []).append(value)
    assert values["identityfile"] == [str(key)]
    assert values["userknownhostsfile"] == [str(known_hosts)]
    assert values["stricthostkeychecking"] == ["true"]
    assert values.get("proxycommand", ["none"]) == ["none"]
