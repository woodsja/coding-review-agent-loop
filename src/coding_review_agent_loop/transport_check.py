"""``agent-loop transport-check``: one real trusted-transport fetch (#1383).

The check fetches the target repository's default branch through the
production :func:`git_transport.import_ref` path into a fresh scratch
repository, using the operator's real remote URL and credentials, and prints
exactly one line with a fixed, non-secret class::

    transport-check: pass
    transport-check: fail (<class>)

optionally followed by `` [ssh-config-warning]`` when the transport ignored a
refused ``~/.ssh/config`` value during this run.  Nothing else reaches stdout
or stderr: argument errors, production warnings, subprocess output and
exception text are all suppressed, because any of them may name a key path,
URL credential or token.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

from . import git_transport
from .errors import AgentLoopError
from .git_transport import TransportFailure

PREFIX = "transport-check"
PASS = "pass"
CONFIGURATION = "configuration"
LOCAL_IMPORT = "local import"
UNCLASSIFIED = "unclassified"
WARNING_SUFFIX = " [ssh-config-warning]"
_CHAIN_DEPTH = 16


class _ArgumentError(Exception):
    """A parse error.  Deliberately carries no argument content."""


class _ConfigurationError(Exception):
    """A local preflight or branch-discovery failure."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str):  # type: ignore[override]
        # Never print usage or the offending token: it may be a secret.
        raise _ArgumentError()


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="agent-loop transport-check",
        exit_on_error=False,
        description=(
            "Perform one trusted-transport fetch of the repository's default branch "
            "into a private scratch repository, using the operator's real remote URL "
            "and credentials, and print only a fixed pass or fail class. Required "
            "before merging changes to git_transport.py, secure_git.py or the "
            "checkout sync path."
        ),
    )
    parser.add_argument("--repo", required=True, help="OWNER/REPO (or HOST/OWNER/REPO).")
    parser.add_argument(
        "--base", default=None,
        help="Branch to fetch. Omit to look up the default branch through the GitHub CLI.",
    )
    parser.add_argument(
        "--trusted-origin-protocol", choices=("https", "ssh"), default="https",
        help="Protocol of the trusted origin (default: https).",
    )
    parser.add_argument(
        "--git-ssh-identity", type=Path, default=None, metavar="PATH",
        help="Absolute SSH private key path (default: AGENT_LOOP_GIT_SSH_IDENTITY, then ~/.ssh/config).",
    )
    parser.add_argument("--gh-cmd", default="gh")
    return parser


def classify_failure(exc: BaseException) -> str:
    """Map an exception to a fixed class without reading its text.

    A non-transport exception raised while a transport failure was being
    handled (for example the HTTPS token retry being unavailable after an
    authentication failure) reports that transport failure's class.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending and len(seen) < _CHAIN_DEPTH:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, TransportFailure):
            return current.label
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
    if isinstance(exc, (AgentLoopError, OSError, subprocess.TimeoutExpired)):
        return LOCAL_IMPORT
    return UNCLASSIFIED


@contextlib.contextmanager
def _isolated_output() -> Iterator[SimpleNamespace]:
    """Silence Python- and fd-level output; isolate SSH warning dedup state.

    ``state.warned`` is set on exit when any SSH config warning was emitted
    inside the block.  The process-wide warning set object is restored
    unchanged, so later loop-side calls still print their warnings.
    """
    state = SimpleNamespace(warned=False)
    original_warnings = git_transport._EMITTED_WARNINGS
    fresh: set[str] = set()
    git_transport._EMITTED_WARNINGS = fresh
    old_stdout, old_stderr = sys.stdout, sys.stderr
    for stream in (old_stdout, old_stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    saved = (os.dup(1), os.dup(2))
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    sink = open(os.devnull, "w", encoding="utf-8")
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        sys.stdout = sys.stderr = sink
        yield state
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr
        for stream in (old_stdout, old_stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull_fd):
            os.close(fd)
        sink.close()
        state.warned = bool(fresh)
        git_transport._EMITTED_WARNINGS = original_warnings


def _safe_branch(name: str | None) -> str:
    if not name or not git_transport._REF.fullmatch(name) or ".." in name:
        raise _ConfigurationError()
    if not git_transport._REF.fullmatch(f"refs/heads/{name}"):
        raise _ConfigurationError()
    return name


def _preflight(repo: str, *, base: str | None, protocol: str, ssh_identity_arg: Path | None,
               gh_cmd: str, runner: Any, scratch: Path) -> Path | None:
    """Phase 1: local configuration only; no network access."""
    from .config import resolve_git_ssh_identity

    try:
        url = git_transport.default_origin(repo, protocol=protocol)
        identity = resolve_git_ssh_identity(
            SimpleNamespace(git_ssh_identity=ssh_identity_arg), local_origin=None,
        )
        if base is not None:
            _safe_branch(base)
        scratch.mkdir(mode=0o700)
        runner.run(("git", "init", "-q"), cwd=scratch)
        runner.run(("git", "remote", "add", "origin", url), cwd=scratch)
        git_transport._transport_env(url, gh_cmd, ssh_identity=identity,
                                     forbidden_roots=(scratch,))
    except (AgentLoopError, OSError, ValueError, _ConfigurationError) as exc:
        raise _ConfigurationError() from exc
    return identity


def _default_branch(repo: str, *, gh_cmd: str, runner: Any, cwd: Path) -> str:
    """Phase 2: one GitHub metadata request; never a transport fetch."""
    from .github import get_repo_default_branch

    try:
        branch = get_repo_default_branch(
            runner, config=SimpleNamespace(gh_cmd=gh_cmd, repo=repo), cwd=cwd,
        )
    except (AgentLoopError, OSError, ValueError) as exc:
        raise _ConfigurationError() from exc
    return _safe_branch(branch)


def _check(repo: str, *, base: str | None, protocol: str, ssh_identity_arg: Path | None,
           gh_cmd: str, runner: Any) -> str:
    root = Path(tempfile.mkdtemp(prefix="agent-loop-transport-check-"))
    try:
        scratch = root / "repo"
        try:
            identity = _preflight(repo, base=base, protocol=protocol,
                                  ssh_identity_arg=ssh_identity_arg, gh_cmd=gh_cmd,
                                  runner=runner, scratch=scratch)
            branch = base if base is not None else _default_branch(
                repo, gh_cmd=gh_cmd, runner=runner, cwd=root)
        except _ConfigurationError:
            return CONFIGURATION
        try:
            git_transport.import_ref(
                scratch, f"refs/heads/{branch}", f"refs/remotes/origin/{branch}",
                repo=repo, runner=runner, gh_cmd=gh_cmd, ssh_identity=identity,
                forbidden_roots=(scratch,),
            )
        except Exception as exc:  # noqa: BLE001 - every failure maps to a fixed class
            return classify_failure(exc)
        return PASS
    except Exception:  # noqa: BLE001
        return UNCLASSIFIED
    finally:
        shutil.rmtree(root, ignore_errors=True)


def run_transport_check(repo: str, *, base: str | None = None, protocol: str = "https",
                        ssh_identity_arg: Path | None = None, gh_cmd: str = "gh",
                        runner: Any = None) -> tuple[str, bool]:
    """Run all phases with output silenced; return ``(class, warned)``."""
    if runner is None:
        from .runner import Runner

        runner = Runner()
    with _isolated_output() as state:
        result = _check(repo, base=base, protocol=protocol, ssh_identity_arg=ssh_identity_arg,
                        gh_cmd=gh_cmd, runner=runner)
    return result, state.warned


def render(result: str, warned: bool = False) -> str:
    line = f"{PREFIX}: {PASS}" if result == PASS else f"{PREFIX}: fail ({result})"
    return line + (WARNING_SUFFIX if warned else "")


def main(argv: Sequence[str], *, runner: Any = None) -> int:
    try:
        args = build_parser().parse_args(list(argv))
    except (_ArgumentError, argparse.ArgumentError):
        print(render(CONFIGURATION))
        return 1
    result, warned = run_transport_check(
        args.repo, base=args.base, protocol=args.trusted_origin_protocol,
        ssh_identity_arg=args.git_ssh_identity, gh_cmd=args.gh_cmd, runner=runner,
    )
    print(render(result, warned))
    return 0 if result == PASS else 1
