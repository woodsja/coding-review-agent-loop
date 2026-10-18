"""Durable host-private phase journal for the split editor/Git-operator handoff.

One record per (repository, kind, number) lives in the per-user state directory
that no editor sandbox can write (see ``private_trusted_dir``). It is written
atomically after each phase so a crashed run can complete publication without a
second commit, a duplicate push or a duplicate PR:

``EDITED``     the Codex candidate is frozen; its objects are in ``trusted``.
``COMMITTED``  a locally verified commit ``sha`` exists.
``PUSHED``     ``sha`` is the verified remote branch head.
``PR_BOUND``   the ordinary PR ``pr_number`` is bound to ``sha``.

The record is host evidence, not authority: recovery re-derives and re-verifies
the tree, the commit and the remote/PR state instead of trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

from .git_handoff import HandoffError, SHA, _branch_name, ensure_state_base, state_base

PHASES = ("EDITED", "COMMITTED", "PUSHED", "PR_BOUND")
KINDS = ("issue", "pr")
VERSION = 1
MAX_RECORD = 1024 * 1024


@dataclass(frozen=True)
class JournalRecord:
    repo: str
    kind: str
    number: int
    handoff_id: str
    branch: str
    base_branch: str
    base: str
    phase: str
    tree: str
    new_paths: tuple[str, ...]
    timestamp: int
    message: str
    editor_text: str
    trusted: str
    sha: str | None = None
    pr_number: int | None = None
    title: str | None = None
    model_used: str | None = None

    def validated(self):
        text_ok = all(isinstance(v, str) for v in (self.repo, self.handoff_id, self.branch,
                      self.base_branch, self.message, self.editor_text, self.trusted))
        if (not text_ok or self.kind not in KINDS or self.phase not in PHASES
                or type(self.number) is not int or self.number < 1
                or type(self.timestamp) is not int
                or len(self.repo.split("/")) != 2
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.handoff_id)
                or not SHA.fullmatch(self.base) or not SHA.fullmatch(self.tree)
                or not isinstance(self.new_paths, tuple)
                or any(not isinstance(p, str) for p in self.new_paths)
                or not os.path.isabs(self.trusted)):
            raise HandoffError("Invalid handoff journal record")
        _branch_name(self.branch)
        _branch_name(self.base_branch)
        index = PHASES.index(self.phase)
        if (index >= 1) != (self.sha is not None) or (self.sha is not None and not SHA.fullmatch(self.sha)):
            raise HandoffError("Handoff journal commit does not match its phase")
        if (index == 3) != (self.pr_number is not None) or (
                self.pr_number is not None and (type(self.pr_number) is not int or self.pr_number < 1)):
            raise HandoffError("Handoff journal PR does not match its phase")
        for optional in (self.title, self.model_used):
            if optional is not None and not isinstance(optional, str):
                raise HandoffError("Invalid handoff journal record")
        return self

    def at(self, phase: str, **changes) -> "JournalRecord":
        return JournalRecord(**{**asdict(self), "new_paths": self.new_paths, "phase": phase, **changes}).validated()


class HandoffJournal:
    def __init__(self, repo: str, kind: str, number: int, checkout: Path):
        if kind not in KINDS or type(number) is not int or number < 1:
            raise HandoffError("Invalid handoff journal identity")
        self.repo, self.kind, self.number = repo, kind, number
        self.root = ensure_state_base(checkout) / "journals"
        key = hashlib.sha256(f"{repo.casefold()}|{kind}|{number}".encode()).hexdigest()
        self.path = self.root / f"{key}.json"

    def _directory(self) -> Path:
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.root.lstat()
        if self.root.is_symlink() or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise HandoffError("Unsafe handoff journal directory")
        return self.root

    def load(self) -> JournalRecord | None:
        self._directory()
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise HandoffError("Handoff journal is unreadable") from exc
        try:
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_mode & 0o077 or info.st_size > MAX_RECORD):
                raise HandoffError("Unsafe handoff journal file")
            raw = os.read(fd, MAX_RECORD + 1)
        finally:
            os.close(fd)
        try:
            data = json.loads(raw)
            data["new_paths"] = tuple(data["new_paths"])
            if data.pop("version") != VERSION:
                raise ValueError
            record = JournalRecord(**data).validated()
        except (ValueError, KeyError, TypeError) as exc:
            raise HandoffError("Corrupt handoff journal; inspect it before retrying") from exc
        if (record.repo.casefold(), record.kind, record.number) != (self.repo.casefold(), self.kind, self.number):
            raise HandoffError("Handoff journal belongs to a different target")
        return record

    def write(self, record: JournalRecord) -> JournalRecord:
        record.validated()
        if (record.repo.casefold(), record.kind, record.number) != (self.repo.casefold(), self.kind, self.number):
            raise HandoffError("Handoff journal belongs to a different target")
        previous = self.load()
        if previous is not None and PHASES.index(record.phase) < PHASES.index(previous.phase):
            raise HandoffError("Handoff journal cannot move to an earlier phase")
        directory = self._directory()
        payload = json.dumps({"version": VERSION, **asdict(record)}, sort_keys=True).encode()
        scratch = directory / (self.path.name + f".{os.getpid()}.tmp")
        fd = os.open(scratch, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(scratch, self.path)
        except BaseException:
            scratch.unlink(missing_ok=True)
            raise
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return record

    def retire(self, *keep: str):
        """Drop the record and the retained host evidence of a finished handoff."""
        record = self.load()
        self.path.unlink(missing_ok=True)
        base = state_base().resolve()
        for raw in (*(() if record is None else (record.trusted,)), *keep):
            path = Path(raw).parent
            if path.parent == base and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)


def retire_published_issue_journal(config, issue_number: int) -> None:
    """Drop an issue journal whose push has already produced the resumed PR.

    A journal still before ``PUSHED`` describes unpublished work for an issue that
    already has an open PR; it is left in place for the owner rather than guessed at.
    """
    from .workdirs import active_workdir

    journal = HandoffJournal(config.repo, "issue", issue_number, active_workdir(config))
    record = journal.load()
    if record is not None and PHASES.index(record.phase) >= PHASES.index("PUSHED"):
        journal.retire()
