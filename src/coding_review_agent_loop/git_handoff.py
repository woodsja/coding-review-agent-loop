"""Verified editor candidate and raw-object validation primitive.

Call begin BEFORE editing, capture AFTER editor exit under the caller's worktree
claim, then run_local_commit and verify. Only the disposable operator area may
be mounted writable. Claude invocation and host publication live in separate modules.
Unsupported inputs fail closed. Trusted state must never be operator writable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import zlib

from .checkout_verification import registered_links, link_intact
from .errors import AgentLoopError
from .protocol_markers import MARKER_BY_TOKEN, scan_reserved_markers
from .secure_git import local_command, _trusted_executable

MAX_FILE = 16 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
MAX_PATHS = 10000
# The Claude Git-operator hides credential/project-setting paths inside its
# namespace. Changed entries under one of these paths cannot be staged
# faithfully and must be rejected *before* handing off to the model.
OPERATOR_MASKED_PATHS = frozenset({
    '.aws', '.ssh', '.claude', '.claude.json', '.codex', '.agents', '.config',
    '.netrc', '.gitconfig', '.git-credentials',
})
MAX_METADATA_FILE = 64 * 1024 * 1024
SHA = re.compile(r"[0-9a-f]{40}\Z")


class HandoffError(AgentLoopError):
    """No verified commit is available; never retry with weaker permissions."""


@dataclass(frozen=True)
class CommitIdentity:
    name: str
    email: str
    timestamp: int
    message: str

    def headers(self) -> bytes:
        if (not re.fullmatch(r"[A-Za-z0-9 ._-]{1,80}", self.name)
                or not re.fullmatch(r"[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+", self.email)
                or len(self.email) > 120 or type(self.timestamp) is not int
                or not 0 <= self.timestamp <= 4102444800):
            raise HandoffError("Invalid trusted commit identity")
        message = self.message
        if (not message.strip() or not message.endswith("\n")
                or len(message.encode()) > 4096 or scan_reserved_markers(message)
                or any(token.casefold() in message.casefold() for token in MARKER_BY_TOKEN)
                or any(ord(c) < 32 and c != "\n" or ord(c) == 127 for c in message)):
            raise HandoffError("Invalid trusted commit message")
        value = f"{self.name} <{self.email}> {self.timestamp} +0000\n"
        return ("author " + value + "committer " + value).encode()


def _branch_name(branch: str):
    if (not isinstance(branch, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_./-]{0,199}", branch)
            or any(x in branch for x in ("..", "//", "@{"))
            or any(p.startswith(".") or p.endswith((".", ".lock")) for p in branch.split("/"))
            or branch.endswith("/") or branch.startswith("refs/")):
        raise HandoffError("Unsafe publication branch")


def state_base() -> Path:
    """The per-user directory for trusted handoff state and journals."""
    state = os.environ.get("XDG_STATE_HOME")
    if not state or not os.path.isabs(state):
        state = os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(os.path.realpath(state)) / "coding-review-agent-loop-handoff"


def ensure_state_base(checkout: Path) -> Path:
    """Create/validate the private state root; it must be outside editor-writable roots."""
    base = state_base()
    writable = {Path(os.path.realpath(p)) for p in (
        "/tmp", "/var/tmp", "/dev/shm", tempfile.gettempdir(), os.environ.get("TMPDIR") or "/tmp")}
    writable.add(checkout.resolve(strict=True))
    if any(base.is_relative_to(root) for root in writable):
        raise HandoffError("Trusted handoff state must not be inside an editor-writable root")
    try:
        base.mkdir(mode=0o700, parents=True, exist_ok=True)
        for directory in (base, *base.parents):
            info = directory.lstat()
            if directory.is_symlink() or not stat.S_ISDIR(info.st_mode):
                raise HandoffError("Unsafe trusted handoff state directory")
            if directory == base and (info.st_uid != os.getuid() or info.st_mode & 0o077):
                raise HandoffError("Trusted handoff state must be private to the current user")
            if info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
                raise HandoffError("Trusted handoff state has a group- or world-writable ancestor")
    except OSError as exc:
        raise HandoffError("Trusted handoff state directory unavailable") from exc
    return base


def private_trusted_dir(checkout: Path, label: str) -> Path:
    """Create host metadata outside every root a workspace-write editor can modify.

    Codex workspace-write can write the checkout, /tmp and $TMPDIR, so the
    trusted Git administration data lives in a 0700 per-user state directory
    that is none of those. Creation happens before the editor runs; the editor
    sandbox mounts it read-only (verified against the real Codex sandbox).
    """
    base = ensure_state_base(checkout)
    try:
        root = Path(tempfile.mkdtemp(prefix=label + "-", dir=base))
    except OSError as exc:
        raise HandoffError("Trusted handoff state directory unavailable") from exc
    return root / "trusted"


def _run(admin: Path | None, checkout: Path, args, *, data=None, objects=None, index=None, alternates=None):
    prefix = ["--literal-pathspecs", "--no-optional-locks"]
    if admin is not None:
        prefix += ["--git-dir", str(admin)]
    command, env, fds = local_command([*prefix, *args], environ={"HOME": "/nonexistent", "LC_ALL": "C"}, checkout=checkout)
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_CONFIG_SYSTEM="/dev/null", GIT_ATTR_NOSYSTEM="1")
    if objects is not None:
        env["GIT_OBJECT_DIRECTORY"] = str(objects)
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    if alternates is not None:
        env["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(alternates)
    try:
        result = subprocess.run(command, env=env, pass_fds=fds, input=data,
                                capture_output=True, timeout=30, cwd=checkout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HandoffError("Local handoff Git unavailable") from exc
    if result.returncode:
        raise HandoffError("Local handoff Git refused operation")
    return result.stdout


def _admin(path: Path, base: str):
    path.mkdir(parents=True)
    (path / "objects").mkdir()
    (path / "refs").mkdir()
    (path / "HEAD").write_text(base + "\n")
    (path / "config").write_text("[core]\n repositoryformatversion = 0\n bare = false\n logallrefupdates = false\n")
    (path / "objects" / "info").mkdir()


def _path(value: str):
    p = PurePosixPath(value)
    if (not value or p.is_absolute() or str(p) != value or ".." in p.parts
            or any(x.lower() == ".git" for x in p.parts) or "\x00" in value):
        raise HandoffError("Unsafe candidate path")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _pointer(root):
    """Pin the linked-worktree pointer without following links or blocking."""
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(".git", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 4096:
                raise HandoffError("A linked worktree pointer is required")
            raw = os.read(fd, 4097)
            if _identity(os.fstat(fd)) != _identity(before):
                raise HandoffError("Worktree pointer changed during capture")
            return _identity(before), raw
        finally:
            os.close(fd)
    except OSError as exc:
        raise HandoffError("Unsafe worktree pointer") from exc
    finally:
        os.close(root_fd)


def _read(root: Path, name: str):
    """Read bytes/mode through descriptors; never follow an ancestor symlink."""
    _path(name)
    opened = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened.append(fd)
        parts = name.split("/")
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            opened.append(fd)
        before = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            target = os.readlink(parts[-1], dir_fd=fd)
            # Deliberately narrow first slice: links may point only inside root.
            if not ((root / name).parent / target).resolve().is_relative_to(root):
                raise HandoffError("Escaping symlink")
            if _identity(os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)) != _identity(before):
                raise HandoffError("Symlink changed during capture")
            return "120000", os.fsencode(target)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE:
            raise HandoffError("Unsupported or oversized candidate file")
        handle = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        opened.append(handle)
        if _identity(os.fstat(handle)) != _identity(before):
            raise HandoffError("Candidate changed during capture")
        with os.fdopen(os.dup(handle), "rb") as stream:
            content = stream.read(MAX_FILE + 1)
        if _identity(os.fstat(handle)) != _identity(before) or len(content) > MAX_FILE:
            raise HandoffError("Candidate changed during capture")
        return ("100755" if before.st_mode & 0o111 else "100644"), content
    except FileNotFoundError:
        return None
    except (OSError, RuntimeError) as exc:
        raise HandoffError("Unsafe candidate filesystem entry") from exc
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _stamp(root: Path):
    """Detect shared administration mutations without reading secret config."""
    entries = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        for name in sorted(dirs + files):
            path = Path(directory) / name
            info = path.lstat()
            entries.append((str(path.relative_to(root)), info.st_mode, info.st_ino,
                            info.st_size, info.st_mtime_ns, info.st_ctime_ns))
            if len(entries) > 100000:
                raise HandoffError("Shared Git metadata exceeds inspection bound")
    return tuple(entries)


@dataclass(frozen=True)
class OperatorContext:
    # admin is host-owned/read-only; private holds index and loose objects only.
    admin: Path
    private: Path
    objects: Path
    shared: Path

    def environment(self):
        return {"GIT_DIR": str(self.admin), "GIT_INDEX_FILE": str(self.private / "index"),
                "GIT_OBJECT_DIRECTORY": str(self.objects),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(self.shared),
                "GIT_NO_REPLACE_OBJECTS": "1"}


_LOOSE_DIR = re.compile(r"[0-9a-f]{2}\Z")


def _seal(admin: Path):
    """Digest every trusted administration entry the host never rewrites.

    Host-written loose objects and the scratch index are exempt; anything else
    (config, hooks, attributes, alternates, packs, links, extra files) is part
    of the seal, so editor-planted Git metadata cannot go unnoticed.
    """
    entries = []
    for directory, dirs, files in os.walk(admin, followlinks=False):
        rel = Path(directory).relative_to(admin)
        dirs.sort()
        if rel.parts[:1] == ("objects",) and len(rel.parts) >= 2 and _LOOSE_DIR.match(rel.parts[1]):
            dirs[:] = []
            continue
        for name in sorted(dirs + files):
            path = Path(directory) / name
            relative = rel / name
            if relative == Path("index") or (relative.parts[:1] == ("objects",)
                                            and len(relative.parts) == 2 and _LOOSE_DIR.match(name)):
                continue
            info = path.lstat()
            kind = stat.S_IFMT(info.st_mode)
            if stat.S_ISREG(kind):
                if info.st_size > MAX_METADATA_FILE:
                    raise HandoffError("Trusted Git metadata exceeds inspection bound")
                content = hashlib.sha256(path.read_bytes()).hexdigest()
            elif stat.S_ISLNK(kind):
                content = os.readlink(path)
            elif stat.S_ISDIR(kind):
                content = ""
            else:
                raise HandoffError("Unexpected trusted Git metadata entry")
            entries.append((str(relative), stat.S_IMODE(info.st_mode), kind, content))
            if len(entries) > 100000:
                raise HandoffError("Trusted Git metadata exceeds inspection bound")
    return tuple(entries)


def _loose(objects: Path, oid: str):
    """Read one loose object with no Git process, config or alternates."""
    if not SHA.fullmatch(oid):
        raise HandoffError("Invalid raw object identity")
    value = _read(objects, oid[:2] + "/" + oid[2:])
    if value is None or value[0] == "120000":
        raise HandoffError("Journaled candidate object is missing")
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(value[1], MAX_FILE + 4097)
        if not decoder.eof or decoder.unused_data or len(raw) > MAX_FILE + 4096:
            raise HandoffError("Unsupported raw object encoding")
        if hashlib.sha1(raw).hexdigest() != oid:
            raise HandoffError("Raw object hash mismatch")
        header, content = raw.split(b"\0", 1)
        kind, size = header.split(b" ")
        if size != str(len(content)).encode():
            raise HandoffError("Invalid raw object size")
    except (ValueError, zlib.error) as exc:
        raise HandoffError("Invalid journaled candidate object") from exc
    return kind, content


def _flatten(objects: Path, tree: str, prefix: str = "", out=None, depth: int = 0):
    out = {} if out is None else out
    kind, raw = _loose(objects, tree)
    if kind != b"tree" or depth > 64:
        raise HandoffError("Invalid journaled candidate tree")
    while raw:
        head, raw = raw.split(b"\0", 1)
        oid, raw = raw[:20].hex(), raw[20:]
        mode, name = head.decode().split(" ", 1)
        path = prefix + name
        if mode == "40000":
            _flatten(objects, oid, path + "/", out, depth + 1)
            continue
        if mode not in {"100644", "100755", "120000"}:
            raise HandoffError("Unsupported journaled candidate entry")
        _path(path)
        out[path] = (mode, oid)
        if len(out) > MAX_PATHS:
            raise HandoffError("Candidate exceeds path bound")
    return out


def _remove(root: Path, name: str):
    _path(name)
    opened = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened.append(fd)
        parts = name.split("/")
        for part in parts[:-1]:
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            opened.append(fd)
        os.unlink(parts[-1], dir_fd=fd)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise HandoffError("Unsafe candidate restoration path") from exc
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _write(root: Path, name: str, mode: str, content: bytes):
    _path(name)
    opened = []
    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened.append(fd)
        parts = name.split("/")
        for part in parts[:-1]:
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            opened.append(fd)
        try:
            os.unlink(parts[-1], dir_fd=fd)
        except FileNotFoundError:
            pass
        if mode == "120000":
            target = os.fsdecode(content)
            if not ((root / name).parent / target).resolve().is_relative_to(root):
                raise HandoffError("Escaping symlink")
            os.symlink(target, parts[-1], dir_fd=fd)
            return
        handle = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o755 if mode == "100755" else 0o644, dir_fd=fd)
        with os.fdopen(handle, "wb") as stream:
            os.fchmod(stream.fileno(), 0o755 if mode == "100755" else 0o644)
            stream.write(content)
    except OSError as exc:
        raise HandoffError("Unsafe candidate restoration path") from exc
    finally:
        for fd in reversed(opened):
            os.close(fd)


class Handoff:
    @classmethod
    def begin(cls, checkout: Path, trusted: Path, identity: CommitIdentity, *, expected_base: str):
        if sys.platform != "linux":
            raise HandoffError("Local handoff requires Linux")
        identity.headers()
        self = cls()
        self.checkout = checkout.resolve(strict=True)
        self.trusted = trusted.resolve()
        self.root_identity = _identity(self.checkout.stat())[:2]
        if self.trusted.is_relative_to(self.checkout):
            raise HandoffError("Trusted metadata must be outside checkout")
        self.identity = identity
        # Linked worktrees only. Ownership/claim policy remains with the caller.
        self.pointer = _pointer(self.checkout)
        if not self.pointer[1].startswith(b"gitdir: "):
            raise HandoffError("A linked worktree is required")
        self.base = _run(None, self.checkout, ["rev-parse", "--verify", "HEAD"]).decode().strip()
        if self.base != expected_base or not SHA.fullmatch(self.base):
            raise HandoffError("Unexpected base or unsupported object format")
        common = _run(None, self.checkout, ["rev-parse", "--git-common-dir"]).decode().strip()
        self.common = (self.checkout / common).resolve(strict=True)
        self.shared = self.common / "objects"
        if (self.shared / "info" / "alternates").exists() or self.shared.is_symlink():
            raise HandoffError("Shared alternate object stores unsupported")
        if self.trusted.is_relative_to(self.common) or self.common.is_relative_to(self.trusted):
            raise HandoffError("Private context must be separate from shared Git")
        self.admin = self.trusted / "git"
        _admin(self.admin, self.base)
        self.objects = self.admin / "objects"
        _run(self.admin, self.checkout, ["read-tree", self.base], index=self.admin / "base-index", alternates=self.shared)
        raw = self.git("ls-tree", "-rz", "--full-tree", self.base)
        self.base_entries = {}
        self.base_objects = {self.base}
        if len(raw) > MAX_TOTAL:
            raise HandoffError("Base tree exceeds inspection bound")
        for item in raw.split(b"\0"):
            if not item:
                continue
            descriptor, path = item.split(b"\t", 1)
            mode, kind, oid = descriptor.decode().split()
            name = os.fsdecode(path)
            _path(name)
            if mode not in {"100644", "100755", "120000"} or kind != "blob":
                raise HandoffError("Unsupported base tree entry")
            self.base_entries[name] = (mode, oid)
            self.base_objects.add(oid)
            if len(self.base_entries) > MAX_PATHS:
                raise HandoffError("Base tree exceeds path bound")
        for line in self.git("ls-tree", "-rt", self.base).splitlines():
            self.base_objects.add(line.split()[2].decode())
        self.base_tree = self.git("rev-parse", self.base + "^{tree}").decode().strip()
        self.base_objects.add(self.base_tree)
        self.links = registered_links(self.checkout)
        if any(os.fsdecode(p) in self.base_entries for p in self.links):
            raise HandoffError("Registered tool link is tracked")
        self.ignores = {p: _read(self.checkout, p) for p in self.base_entries if p.split("/")[-1] == ".gitignore"}
        # begin must precede edits to the trusted starting ignore policy.
        for p, value in self.ignores.items():
            mode, oid = self.base_entries[p]
            if value != (mode, self.git("cat-file", "blob", oid)):
                raise HandoffError("Starting ignore rules differ from base")
        self.shared_stamp = _stamp(self.common)
        self.seal = _seal(self.admin)
        self.tree = None
        self._context = None
        return self

    def git(self, *args, data=None):
        # Writes never see alternates: Git may freshen even an existing blob's
        # timestamp. Read-only base inspection may consult the pinned store.
        alternate = self.shared if args[0] in {"cat-file", "ls-tree", "rev-parse"} else None
        return _run(self.admin, self.checkout, args, data=data, alternates=alternate)

    def _unchanged(self):
        try:
            sealed = _seal(self.admin) == self.seal
            if self.tree is None:  # No host object write has happened yet.
                sealed = sealed and sorted(os.listdir(self.objects)) == ["info"]
        except OSError as exc:
            raise HandoffError("Trusted Git metadata is unreadable") from exc
        if not sealed:
            raise HandoffError("Trusted Git metadata was modified after it was sealed")
        if (_identity(self.checkout.stat())[:2] != self.root_identity
                or _pointer(self.checkout) != self.pointer
                or _stamp(self.common) != self.shared_stamp):
            raise HandoffError("Checkout identity or shared Git metadata changed")
        for path, target in self.links.items():
            if not link_intact(self.checkout, path, target):
                raise HandoffError("Registered tool link changed")

    def _snapshot(self, allowed_new):
        self._unchanged()
        for p, value in self.ignores.items():
            if _read(self.checkout, p) != value:
                raise HandoffError("Ignore-rule edits unsupported in this slice")
        # Enumerate WITHOUT excludes first: a newly written .gitignore can
        # hide itself as well as an undeclared file. Read no artifact contents.
        untracked = _run(self.admin, self.checkout, ["ls-files", "--others", "-z"],
                         index=self.admin / "base-index")
        if len(untracked) > MAX_TOTAL or untracked.count(b"\0") > MAX_PATHS:
            raise HandoffError("Untracked enumeration exceeds inspection bound")
        if any(p.split(b"/")[-1] == b".gitignore" for p in untracked.split(b"\0")):
            raise HandoffError("Ignore-rule additions unsupported in this slice")
        raw_eligible = _run(self.admin, self.checkout,
                            ["ls-files", "--others", "--exclude-standard", "-z"],
                            index=self.admin / "base-index")
        eligible = {os.fsdecode(p) for p in raw_eligible.split(b"\0") if p}
        eligible -= {os.fsdecode(p) for p in self.links}
        if eligible != set(allowed_new) or any(p.split("/")[-1] == ".gitignore" for p in eligible):
            raise HandoffError("New files do not match explicit eligible declaration")
        names = set(self.base_entries) | eligible
        if len(names) > MAX_PATHS:
            raise HandoffError("Candidate exceeds path bound")
        snapshot = {}
        total = 0
        for name in sorted(names):
            value = _read(self.checkout, name)
            total += len(value[1]) if value else 0
            if total > MAX_TOTAL:
                raise HandoffError("Candidate exceeds byte bound")
            if value is not None and (
                    value[1].startswith(b"version https://git-lfs.github.com/spec/v1\n")
                    or (name.split("/")[-1] == ".gitattributes"
                        and re.search(rb"filter\s*=\s*lfs(?:\s|$)", value[1]))):
                raise HandoffError("Git LFS is unsupported by the verified handoff")
            snapshot[name] = value
        return snapshot

    def restore(self, previous_objects: Path, tree: str):
        """Rebuild a journaled frozen candidate in this clean, base-pinned checkout.

        Reads the previous handoff's loose objects directly (hash-verified, no Git
        process), refuses unless the checkout currently equals the pinned base,
        then writes only the paths that differ. The caller re-captures and must
        require the identical tree before any commit is made.
        """
        if self.tree is not None:
            raise HandoffError("Candidate already frozen")
        for name, value in self._snapshot(frozenset()).items():
            mode_oid = None if value is None else (value[0], hashlib.sha1(
                b"blob " + str(len(value[1])).encode() + b"\0" + value[1]).hexdigest())
            if mode_oid != self.base_entries.get(name):
                raise HandoffError("Recovery requires a clean checkout at the pinned base")
        candidate = _flatten(previous_objects, tree)
        for name in sorted(set(self.base_entries) - set(candidate)):
            _remove(self.checkout, name)
        for name, (mode, oid) in sorted(candidate.items()):
            if self.base_entries.get(name) != (mode, oid):
                kind, content = _loose(previous_objects, oid)
                if kind != b"blob":
                    raise HandoffError("Invalid journaled candidate blob")
                _write(self.checkout, name, mode, content)

    def capture(self, *, allowed_new: set[str], allow_no_change: bool = False):
        if self.tree is not None:
            raise HandoffError("Candidate already frozen")
        self.allowed_new = frozenset(allowed_new)
        self.snapshot = self._snapshot(self.allowed_new)
        # Refuse before writing even private Git blobs/index: the operator's
        # namespace masks these paths and cannot commit them faithfully.
        self.changed_paths = tuple(p for p, value in self.snapshot.items() if (
            value is None or p not in self.base_entries or
            (value[0], hashlib.sha1(b"blob " + str(len(value[1])).encode() + b"\0" + value[1]).hexdigest()) != self.base_entries[p]))
        if any(name.split("/", 1)[0] in OPERATOR_MASKED_PATHS for name in self.changed_paths):
            self.changed_paths = ()
            raise HandoffError("Changes to operator-masked credential or settings paths are unsupported")
        self.git("read-tree", "--empty")
        entries = []
        for name, value in self.snapshot.items():
            if value is None:
                continue
            mode, content = value
            oid = self.git("hash-object", "--no-filters", "-w", "--stdin", data=content).strip()
            entries.append(mode.encode() + b" " + oid + b"\t" + os.fsencode(name) + b"\0")
        self.git("update-index", "-z", "--index-info", data=b"".join(entries))
        self.tree = self.git("write-tree").decode().strip()
        if self.tree == self.base_tree and not allow_no_change:
            self.tree = None
            raise HandoffError("No candidate changes")
        return self.tree

    def operator_context(self):
        if self.tree is None:
            raise HandoffError("Candidate not frozen")
        if self._context is None:
            private = self.trusted.parent / (self.trusted.name + "-operator")
            private.mkdir(mode=0o700)
            objects = private / "objects"
            objects.mkdir()
            # A missing private index makes `git status` falsely report the
            # tracked checkout as staged deletions. Initialize from the pinned
            # base before giving the operator any inspection tools. Only the
            # disposable index is written; base objects are read via alternates.
            _run(self.admin, self.checkout, ["read-tree", self.base],
                 index=private / "index", objects=objects, alternates=self.shared)
            self._context = OperatorContext(self.admin, private, objects, self.shared)
        return self._context

    def _object(self, oid):
        if not SHA.fullmatch(oid):
            raise HandoffError("Invalid raw object identity")
        ctx = self.operator_context()
        try:
            value = _read(ctx.objects, oid[:2] + "/" + oid[2:])
            if value is None:
                if oid not in self.base_objects:
                    raise HandoffError("Missing operator loose object")
                kind = self.git("cat-file", "-t", oid).strip()
                content = self.git("cat-file", kind.decode(), oid)
                return kind, content
            compressed = value[1]
            decoder = zlib.decompressobj()
            raw = decoder.decompress(compressed, MAX_FILE + 4097)
            if not decoder.eof or decoder.unused_data or len(raw) > MAX_FILE + 4096:
                raise HandoffError("Unsupported raw object encoding")
            if hashlib.sha1(raw).hexdigest() != oid:
                raise HandoffError("Raw object hash mismatch")
            header, content = raw.split(b"\0", 1)
            kind, size = header.split(b" ")
            if size != str(len(content)).encode():
                raise HandoffError("Invalid raw object size")
            return kind, content
        except (ValueError, zlib.error, OSError) as exc:
            raise HandoffError("Invalid operator object") from exc

    def verify(self, sha: str):
        if self.tree is None:
            raise HandoffError("Candidate not frozen")
        if self._snapshot(self.allowed_new) != self.snapshot:
            raise HandoffError("Source changed after candidate freeze")
        kind, content = self._object(sha)
        expected = (f"tree {self.tree}\nparent {self.base}\n".encode()
                    + self.identity.headers() + b"\n" + self.identity.message.encode())
        if kind != b"commit" or content != expected:
            raise HandoffError("Raw commit violates candidate or metadata contract")
        pending = [(self.tree, b"tree")]
        visited = set()
        while pending:
            oid, kind = pending.pop()
            if oid in visited:
                continue
            visited.add(oid)
            actual_kind, raw = self._object(oid)
            expected_raw = self.git("cat-file", kind.decode(), oid)
            if actual_kind != kind or raw != expected_raw:
                raise HandoffError("Operator tree/blob mismatch")
            if kind == b"tree":
                while raw:
                    head, raw = raw.split(b"\0", 1)
                    child, raw = raw[:20].hex(), raw[20:]
                    pending.append((child, b"tree" if head.startswith(b"40000 ") else b"blob"))
        if self._snapshot(self.allowed_new) != self.snapshot:
            raise HandoffError("Source changed during verification")
        return sha

    def _mounts(self, ctx):
        """Fixed Linux mount layout shared with the synthetic denial probe."""
        bwrap = _trusted_executable("/usr/bin/bwrap")
        mounts = [bwrap, "--die-with-parent", "--unshare-net", "--ro-bind", "/", "/",
                  "--dev-bind", "/dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
        for path in dict.fromkeys([self.common, self.checkout, self.trusted, ctx.private]):
            mounts += ["--dir", str(path), "--ro-bind", str(path), str(path)]
        mounts += ["--bind", str(ctx.private), str(ctx.private), "--chdir", str(self.checkout), "--"]
        return mounts

    def run_local_commit(self):
        """Trusted fixed stage+commit wrapper; accepts NO model shell/argv.

        This only exercises local Git. A Claude CLI allowlist, credential masks
        and isolated settings remain unimplemented; do not launch a model here.
        """
        ctx = self.operator_context()
        mounts = self._mounts(ctx)
        git = _trusted_executable("/usr/bin/git")
        env = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent", "LC_ALL": "C",
               "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
               "GIT_CONFIG_SYSTEM": "/dev/null", **ctx.environment()}
        prefix = [git, "--literal-pathspecs", "--work-tree", str(self.checkout),
                  "-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false",
                  "-c", "protocol.allow=never", "-c", "gc.auto=0"]

        def run(args, data=None):
            try:
                result = subprocess.run([*mounts, *prefix, *args], env=env, input=data,
                                        capture_output=True, timeout=30)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise HandoffError("Linux mount confinement unavailable") from exc
            if result.returncode:
                if result.stderr.startswith(b"bwrap:"):
                    raise HandoffError("Linux mount confinement unavailable")
                raise HandoffError("Confined local Git refused operation")
            return result.stdout.strip().decode()

        run(["read-tree", self.base])
        run(["add", "--all", "--", *self.changed_paths])
        tree = run(["write-tree"])
        # commit-tree uses native Git without hooks, signing, refs or gc.
        env.update({f"GIT_{role}_{field}": value for role in ("AUTHOR", "COMMITTER")
                    for field, value in (("NAME", self.identity.name), ("EMAIL", self.identity.email),
                                         ("DATE", f"{self.identity.timestamp} +0000"))})
        sha = run(["commit-tree", tree, "-p", self.base], self.identity.message.encode())
        return self.verify(sha)
