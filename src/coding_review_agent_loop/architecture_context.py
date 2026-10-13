"""Bounded, immutable architecture context for agent prompts.

Architecture documentation is advisory repository input.  This module reads it
from committed Git objects only, so a worktree symlink, a dirty checkout, or a
document's prose cannot affect acquisition or protocol state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable, Sequence
from urllib.parse import unquote

from .errors import AgentLoopError
from .protocol_markers import sanitize_historical_text
from .runner import Runner

DEFAULT_ARCHITECTURE_PATH = "ARCHITECTURE.md"
DEFAULT_ARCHITECTURE_READ_SIZE = 64 * 1024
DEFAULT_ARCHITECTURE_SNAPSHOT_CHARS = 12_000
DEFAULT_ARCHITECTURE_AGGREGATE_CHARS = 24_000
DEFAULT_MANAGED_CONTEXT_CHARS = 80_000

# The architecture set (#1454): the root document plus every Markdown file
# under this directory that the root links to, one level only.
ARCHITECTURE_DETAIL_DIR = "docs/architecture/"
# An oversized root is still read up to this ceiling so its heading index and
# links are known; above it the root is never read.
DEFAULT_ARCHITECTURE_SCAN_CEILING = 2 * 1024 * 1024
# First-heading reads (and therefore prompt rendering detail) stop after this
# many members; membership and identities never do.
ARCHITECTURE_MEMBER_HEADING_CAP = 64
ARCHITECTURE_LS_TREE_BATCH = 100
_ARCHITECTURE_HEADING_CHARS = 200
_OVERSIZED_HEADING_INDEX_CAP = 500
_ARCHITECTURE_CHANGED_MEMBER_RENDER_CAP = 20

_OID_RE = re.compile(rb"^[0-9a-fA-F]{40,64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MARKER_LIKE_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_URI_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_MEMBERSHIP_UNKNOWN_LINE = "- Architecture set: membership unknown (root could not be scanned)"
_SPLIT_RECOMMENDATION = (
    "The document is too large to inject and should be split into a short root "
    "overview plus docs/architecture/* detail docs."
)
_PARSER_UNAVAILABLE_NOTE = (
    "The CommonMark parser (markdown-it-py) is unavailable, so architecture-set "
    "membership is unknown."
)
_MEMBERSHIP_UNKNOWN_NOTE = (
    "Architecture-set membership is unknown because the root could not be scanned."
)


def _sanitize_architecture_text(text: str) -> str:
    """Neutralize even marker names unknown to the current protocol registry."""
    text = sanitize_historical_text(text)
    return _MARKER_LIKE_RE.sub("[repository comment omitted]", text)


def normalize_architecture_path(value: str | None) -> str:
    """Validate and normalize the configured repository-relative path."""
    path = DEFAULT_ARCHITECTURE_PATH if value is None else value
    if not isinstance(path, str) or not path or "\x00" in path:
        raise AgentLoopError("--architecture-path must be a non-empty repository-relative POSIX path.")
    if "\\" in path or _CONTROL_RE.search(path):
        raise AgentLoopError("--architecture-path must use printable POSIX path characters only.")
    if path.startswith("/"):
        raise AgentLoopError("--architecture-path must be repository-relative, not absolute.")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise AgentLoopError("--architecture-path must not contain empty, '.' or '..' components.")
    normalized = str(PurePosixPath(*parts))
    if normalized != path:
        raise AgentLoopError("--architecture-path must be a normalized POSIX path.")
    return normalized


@dataclass(frozen=True)
class ArchitectureLocator:
    repository: str
    path: str = DEFAULT_ARCHITECTURE_PATH

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not self.repository.strip():
            raise AgentLoopError("Architecture repository identity must be non-empty.")
        object.__setattr__(self, "path", normalize_architecture_path(self.path))


@dataclass(frozen=True)
class ArchitectureMember:
    """One linked detail document of the architecture set (#1454)."""

    path: str
    blob_oid: str | None
    # available, missing, oversized, binary, or unavailable.
    availability: str
    size: int | None = None
    first_heading: str | None = None

    def identity(self) -> dict[str, object]:
        return {
            "path": self.path, "blob_oid": self.blob_oid,
            "availability": self.availability, "size": self.size,
        }


@dataclass(frozen=True)
class ArchitectureSnapshot:
    repository: str
    path: str
    revision: str | None
    blob_oid: str | None
    sha256: str | None
    availability: str
    size: int | None = None
    content: str | None = None
    truncated: bool = False
    heading_index: tuple[str, ...] = ()
    diagnostic: str | None = None
    members: tuple[ArchitectureMember, ...] = ()
    # Whether the root's links are known: ``scanned`` (root text decoded),
    # ``unscanned`` (root exists but its text could not be decoded), or
    # ``not-applicable`` (root missing or unavailable).  Never part of the
    # identity, so root-only identities stay byte-identical.
    links_status: str = "not-applicable"

    @property
    def is_available(self) -> bool:
        return self.availability == "available"

    @property
    def material(self) -> bool:
        return self.is_available and bool(self.content)

    def identity(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "repository": self.repository, "path": self.path, "revision": self.revision,
            "blob_oid": self.blob_oid, "sha256": self.sha256,
            "availability": self.availability, "size": self.size,
        }
        if self.members:
            identity["members"] = [
                member.identity() for member in sorted(self.members, key=lambda item: item.path)
            ]
        return identity


@dataclass(frozen=True)
class ArchitecturePair:
    repository: str
    path: str
    target_revision: str | None
    candidate_revision: str | None
    merge_base_revision: str | None
    base: ArchitectureSnapshot
    candidate: ArchitectureSnapshot
    change: str

    @property
    def material(self) -> bool:
        return self.base.material or self.candidate.material

    @property
    def has_members(self) -> bool:
        return bool(self.base.members or self.candidate.members)

    @property
    def member_changes(self) -> tuple[tuple[str, str], ...]:
        """Every member whose (availability, blob) differs between the sides."""
        base_map = {member.path: member for member in self.base.members}
        candidate_map = {member.path: member for member in self.candidate.members}
        changes: list[tuple[str, str]] = []
        for path in sorted(set(base_map) | set(candidate_map)):
            base_member = base_map.get(path)
            candidate_member = candidate_map.get(path)
            base_key = (
                (base_member.availability, base_member.blob_oid) if base_member else ("unlinked", None)
            )
            candidate_key = (
                (candidate_member.availability, candidate_member.blob_oid)
                if candidate_member else ("unlinked", None)
            )
            if base_key == candidate_key:
                continue
            if "unavailable" in (base_key[0], candidate_key[0]):
                changes.append((path, "unavailable"))
                continue
            base_present = base_key[0] not in {"missing", "unlinked"}
            candidate_present = candidate_key[0] not in {"missing", "unlinked"}
            if candidate_present and not base_present:
                changes.append((path, "added"))
            elif base_present and not candidate_present:
                changes.append((path, "deleted"))
            else:
                changes.append((path, "modified"))
        return tuple(changes)

    @property
    def set_change(self) -> str:
        """Root-plus-members comparison; incomplete knowledge is never unchanged."""
        if self.change == "unavailable":
            return "unavailable"
        sides = (self.base, self.candidate)
        if any(side.links_status == "unscanned" for side in sides):
            return "unavailable"
        if any(member.availability == "unavailable" for side in sides for member in side.members):
            return "unavailable"
        if not self.member_changes:
            return self.change
        if self.change in {"added", "deleted"}:
            return self.change
        return "modified"

    def identity(self) -> dict[str, object]:
        identity: dict[str, object] = {
            "repository": self.repository, "path": self.path,
            "target_revision": self.target_revision,
            "candidate_revision": self.candidate_revision,
            "merge_base_revision": self.merge_base_revision,
            "change": self.change, "base": self.base.identity(),
            "candidate": self.candidate.identity(),
        }
        if self.has_members:
            identity["set_change"] = self.set_change
            identity["member_changes"] = [list(change) for change in self.member_changes]
        return identity


def _snapshot_unavailable(locator: ArchitectureLocator, revision: str | None, diagnostic: str) -> ArchitectureSnapshot:
    return ArchitectureSnapshot(
        repository=locator.repository,
        path=locator.path,
        revision=revision,
        blob_oid=None,
        sha256=None,
        availability="unavailable",
        diagnostic=diagnostic,
    )


def _parse_ls_tree(raw: bytes, path: str) -> tuple[bytes, bytes, bytes] | None:
    entries = [entry for entry in raw.split(b"\x00") if entry]
    matches: list[tuple[bytes, bytes, bytes]] = []
    for entry in entries:
        try:
            header, entry_path = entry.split(b"\t", 1)
            mode, kind, oid = header.split(b" ", 2)
        except ValueError:
            continue
        if entry_path == path.encode("utf-8"):
            matches.append((mode, kind, oid))
    if len(matches) != 1:
        return None
    return matches[0]


def _headings(text: str) -> tuple[str, ...]:
    result: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            result.append(match.group(1))
    return tuple(result)


def _bounded_heading(heading: str) -> str:
    heading = " ".join(heading.split())
    if len(heading) > _ARCHITECTURE_HEADING_CHARS:
        heading = heading[: _ARCHITECTURE_HEADING_CHARS - 3] + "..."
    return heading


def _resolve_repo_relative(base_dir_parts: Iterable[str], value: object) -> str | None:
    """Resolve a link target or reported path to a normalized repository path.

    Shared by link discovery and ``canonical_document_path`` membership, so an
    equivalent spelling of a member resolves identically.  Absolute paths,
    URIs, and escapes above the repository root return ``None``.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.startswith("/") or _URI_SCHEME_RE.match(value):
        return None
    parts = list(base_dir_parts)
    for segment in value.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(segment)
    if not parts:
        return None
    try:
        return normalize_architecture_path("/".join(parts))
    except AgentLoopError:
        return None


# Markdown link discovery (#1454) delegates to a CommonMark parser
# (markdown-it-py, a declared runtime dependency).  Membership must follow
# real Markdown structure -- containers, code, raw HTML, escapes, character
# references and reference-link resolution -- which a bounded hand parser
# kept approximating incorrectly.  Without the parser, membership is unknown
# and the root is treated as unscanned rather than guessed.
_COMMONMARK_PARSER = None


def _commonmark_parser():
    global _COMMONMARK_PARSER
    if _COMMONMARK_PARSER is None:
        try:
            from markdown_it import MarkdownIt
        except ImportError:
            return None
        _COMMONMARK_PARSER = MarkdownIt("commonmark")
    return _COMMONMARK_PARSER


def _link_destination_path(href: str) -> str:
    """The repository path a parsed link destination names.

    The parser has already decoded escapes and character references and
    percent-encoded the result, so a literal ``#`` or ``?`` is a URL
    delimiter; percent-decoding happens only after they are removed.
    """
    return unquote(href.split("#", 1)[0].split("?", 1)[0])


def _linked_detail_paths(root_text: str, root_path: str) -> tuple[str, ...] | None:
    """Repository paths of docs/architecture/*.md files linked from the root.

    Every link the CommonMark parser produces counts: inline links and
    reference links whose definition exists.  Images, code, raw HTML and
    unused definitions produce no links.  Detail docs are never parsed, so
    links are followed one level only.  Returns None when no parser is
    available, so membership is reported as unknown.
    """
    parser = _commonmark_parser()
    if parser is None:
        return None
    base_dir_parts = root_path.split("/")[:-1]
    found: set[str] = set()
    pending = list(parser.parse(root_text))
    while pending:
        token = pending.pop()
        if token.children:
            pending.extend(token.children)
        if token.type != "link_open":
            continue
        href = token.attrGet("href")
        if not isinstance(href, str):
            continue
        resolved = _resolve_repo_relative(base_dir_parts, _link_destination_path(href))
        if (
            resolved is not None
            and resolved != root_path
            and resolved.startswith(ARCHITECTURE_DETAIL_DIR)
            and resolved.endswith(".md")
        ):
            found.add(resolved)
    return tuple(sorted(found))


def _blob_size(runner: Runner, checkout, oid: str) -> int | None:
    result = runner.run_binary(("git", "cat-file", "-s", oid), cwd=checkout, check=False)
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except (TypeError, ValueError):
        return None


def _read_blob(runner: Runner, checkout, oid: str, size: int) -> bytes | None:
    blob = runner.run_binary(
        ("git", "cat-file", "blob", oid), cwd=checkout, max_bytes=size, check=False,
    )
    if blob.returncode != 0 or len(blob.stdout) != size:
        return None
    return blob.stdout


def _decode_text(data: bytes) -> tuple[str | None, str | None]:
    """Apply the shared NUL and UTF-8 checks; return (text, failure diagnostic)."""
    if b"\x00" in data:
        return None, "Architecture blob contains NUL bytes."
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "Architecture blob is not valid UTF-8."


def _parse_ls_tree_batch(raw: bytes, paths: Sequence[str]) -> dict[str, tuple[bytes, bytes, bytes]] | None:
    """Parse one batched listing; ``None`` when any entry is ambiguous or unparsable."""
    wanted = {path.encode("utf-8"): path for path in paths}
    found: dict[str, tuple[bytes, bytes, bytes]] = {}
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        try:
            header, entry_path = entry.split(b"\t", 1)
            mode, kind, oid = header.split(b" ", 2)
        except ValueError:
            return None
        path = wanted.get(entry_path)
        if path is None:
            return None
        if path in found:
            return None
        found[path] = (mode, kind, oid)
    return found


def _acquire_members(
    runner: Runner, *, checkout, revision: str, paths: Sequence[str], max_bytes: int,
) -> tuple[ArchitectureMember, ...]:
    """Resolve every linked member from immutable Git objects, in bounded batches."""
    members: list[ArchitectureMember] = []
    for start in range(0, len(paths), ARCHITECTURE_LS_TREE_BATCH):
        batch = list(paths[start:start + ARCHITECTURE_LS_TREE_BATCH])
        # One entry is "<mode> <type> <oid>\t<path>\0"; allow generous slack.
        bound = sum(len(path.encode("utf-8")) + 128 for path in batch)
        parsed: dict[str, tuple[bytes, bytes, bytes]] | None
        try:
            listing = runner.run_binary(
                ("git", "--literal-pathspecs", "ls-tree", "-z", revision, "--", *batch),
                cwd=checkout, max_bytes=bound, check=False,
            )
        except AgentLoopError:
            listing = None
        if listing is None or listing.returncode != 0 or len(listing.stdout) > bound:
            parsed = None
        else:
            parsed = _parse_ls_tree_batch(listing.stdout, batch)
        for path in batch:
            if parsed is None:
                # A failed, truncated, or ambiguous listing is unknown, never missing.
                members.append(ArchitectureMember(path, None, "unavailable"))
                continue
            entry = parsed.get(path)
            if entry is None:
                members.append(ArchitectureMember(path, None, "missing"))
                continue
            mode, kind, oid = entry
            if kind != b"blob" or mode not in {b"100644", b"100755"} or not _OID_RE.fullmatch(oid):
                members.append(ArchitectureMember(path, None, "unavailable"))
                continue
            members.append(ArchitectureMember(path, oid.decode("ascii"), "available"))
    resolved: list[ArchitectureMember] = []
    headings_read = 0
    for member in members:
        if member.availability != "available" or member.blob_oid is None:
            resolved.append(member)
            continue
        try:
            size = _blob_size(runner, checkout, member.blob_oid)
        except AgentLoopError:
            size = None
        if size is None:
            resolved.append(ArchitectureMember(member.path, member.blob_oid, "unavailable"))
            continue
        if size > max_bytes:
            resolved.append(ArchitectureMember(member.path, member.blob_oid, "oversized", size))
            continue
        if headings_read >= ARCHITECTURE_MEMBER_HEADING_CAP:
            resolved.append(ArchitectureMember(member.path, member.blob_oid, "available", size))
            continue
        headings_read += 1
        try:
            data = _read_blob(runner, checkout, member.blob_oid, size)
        except AgentLoopError:
            data = None
        if data is None:
            resolved.append(ArchitectureMember(member.path, member.blob_oid, "unavailable", size))
            continue
        text, failure = _decode_text(data)
        if text is None:
            resolved.append(ArchitectureMember(member.path, member.blob_oid, "binary", size))
            continue
        headings = _headings(text)
        first = _bounded_heading(headings[0]) if headings else None
        resolved.append(ArchitectureMember(member.path, member.blob_oid, "available", size, first))
    return tuple(resolved)


def _members_for_root(
    runner: Runner, *, checkout, revision: str, root_text: str, root_path: str, max_bytes: int,
) -> tuple[ArchitectureMember, ...] | None:
    """Linked members, or None when the root's links cannot be parsed."""
    paths = _linked_detail_paths(root_text, root_path)
    if paths is None:
        return None
    if not paths:
        return ()
    return _acquire_members(
        runner, checkout=checkout, revision=revision, paths=paths, max_bytes=max_bytes,
    )


def acquire_architecture_snapshot(
    runner: Runner,
    *,
    checkout,
    repository: str,
    revision: str,
    path: str = DEFAULT_ARCHITECTURE_PATH,
    max_bytes: int = DEFAULT_ARCHITECTURE_READ_SIZE,
    scan_ceiling: int | None = None,
) -> ArchitectureSnapshot:
    """Read the architecture set's root (and its linked members) from one immutable revision."""
    locator = ArchitectureLocator(repository, path)
    if scan_ceiling is None:
        scan_ceiling = DEFAULT_ARCHITECTURE_SCAN_CEILING
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise AgentLoopError("Architecture read size must be a positive integer.")
    listing = runner.run_binary(
        ("git", "--literal-pathspecs", "ls-tree", "-z", revision, "--", locator.path),
        cwd=checkout,
        max_bytes=1024 * 1024,
        check=False,
    )
    if listing.returncode != 0:
        return _snapshot_unavailable(locator, revision, "Git revision is unavailable.")
    parsed = _parse_ls_tree(listing.stdout, locator.path)
    if parsed is None:
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=None, sha256=None, availability="missing",
            diagnostic="Architecture path is missing or ambiguous.",
        )
    mode, kind, oid = parsed
    if kind != b"blob" or mode != b"100644" and mode != b"100755":
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii", "replace"), sha256=None,
            availability="unavailable", diagnostic="Architecture path is not a regular blob.",
        )
    if not _OID_RE.fullmatch(oid):
        return _snapshot_unavailable(locator, revision, "Git returned an invalid blob identity.")
    size_result = runner.run_binary(("git", "cat-file", "-s", oid.decode("ascii")), cwd=checkout, check=False)
    if size_result.returncode != 0:
        return _snapshot_unavailable(locator, revision, "Architecture blob size is unavailable.")
    try:
        size = int(size_result.stdout.strip())
    except (TypeError, ValueError):
        return _snapshot_unavailable(locator, revision, "Git returned an invalid blob size.")
    if size > max_bytes:
        diagnostic = f"Architecture blob is {size} bytes; limit is {max_bytes}. {_SPLIT_RECOMMENDATION}"
        oversized = ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii"), sha256=None, availability="oversized", size=size,
            diagnostic=f"{diagnostic} {_MEMBERSHIP_UNKNOWN_NOTE}", links_status="unscanned",
        )
        if size > scan_ceiling:
            return oversized
        try:
            data = _read_blob(runner, checkout, oid.decode("ascii"), size)
        except AgentLoopError:
            data = None
        text = _decode_text(data)[0] if data is not None else None
        if text is None:
            return oversized
        # Links are discovered independently of headings, so a headingless
        # oversized root keeps its members.
        members = _members_for_root(
            runner, checkout=checkout, revision=revision, root_text=text,
            root_path=locator.path, max_bytes=max_bytes,
        )
        headings = tuple(_bounded_heading(item) for item in _headings(text)[:_OVERSIZED_HEADING_INDEX_CAP])
        if members is None:
            diagnostic = f"{diagnostic} {_PARSER_UNAVAILABLE_NOTE}"
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii"), sha256=None, availability="oversized", size=size,
            heading_index=headings, diagnostic=diagnostic, members=members or (),
            links_status="unscanned" if members is None else "scanned",
        )
    blob = runner.run_binary(
        ("git", "cat-file", "blob", oid.decode("ascii")),
        cwd=checkout,
        max_bytes=size,
        check=False,
    )
    if blob.returncode != 0 or len(blob.stdout) != size:
        return _snapshot_unavailable(locator, revision, "Architecture blob could not be read completely.")
    text, failure = _decode_text(blob.stdout)
    if text is None:
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii"), sha256=hashlib.sha256(blob.stdout).hexdigest(),
            availability="binary", size=size, diagnostic=failure, links_status="unscanned",
        )
    members = _members_for_root(
        runner, checkout=checkout, revision=revision, root_text=text,
        root_path=locator.path, max_bytes=max_bytes,
    )
    return ArchitectureSnapshot(
        repository=locator.repository, path=locator.path, revision=revision,
        blob_oid=oid.decode("ascii"), sha256=hashlib.sha256(blob.stdout).hexdigest(),
        availability="available", size=size, content=text, heading_index=_headings(text),
        diagnostic=_PARSER_UNAVAILABLE_NOTE if members is None else None,
        members=members or (), links_status="unscanned" if members is None else "scanned",
    )


def acquire_architecture_pair(
    runner: Runner,
    *,
    checkout,
    repository: str,
    target_revision: str,
    candidate_revision: str,
    path: str = DEFAULT_ARCHITECTURE_PATH,
    max_bytes: int = DEFAULT_ARCHITECTURE_READ_SIZE,
) -> ArchitecturePair:
    """Capture base/candidate architecture from one immutable PR acquisition."""
    locator = ArchitectureLocator(repository, path)
    # PR metadata supplies a logical base branch, while a PR checkout normally
    # has only the remote-tracking ref. Resolve that ref for the read, but keep
    # the logical value in the pair identity so a retarget is observable. A
    # caller may already have supplied ``origin/<branch>`` or an immutable OID.
    merge_target = target_revision
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", target_revision):
        remote_target = (
            target_revision
            if target_revision.startswith(("origin/", "refs/"))
            else f"origin/{target_revision}"
        )
        resolved = runner.run(
            ("git", "rev-parse", "--verify", remote_target),
            cwd=checkout,
            check=False,
        )
        if resolved.returncode == 0 and resolved.stdout.strip():
            merge_target = remote_target
    merge = runner.run(("git", "merge-base", merge_target, candidate_revision), cwd=checkout, check=False)
    merge_base = merge.stdout.strip() if merge.returncode == 0 else None
    if not merge_base or not re.fullmatch(r"[0-9a-fA-F]{40,64}", merge_base):
        base = _snapshot_unavailable(locator, None, "No valid merge base was established.")
        candidate = acquire_architecture_snapshot(
            runner, checkout=checkout, repository=repository, revision=candidate_revision,
            path=locator.path, max_bytes=max_bytes,
        )
        return ArchitecturePair(
            repository=repository, path=locator.path, target_revision=target_revision,
            candidate_revision=candidate_revision, merge_base_revision=None,
            base=base, candidate=candidate, change="unavailable",
        )
    base_revision = merge_base
    base = acquire_architecture_snapshot(
        runner, checkout=checkout, repository=repository, revision=base_revision,
        path=locator.path, max_bytes=max_bytes,
    )
    candidate = acquire_architecture_snapshot(
        runner, checkout=checkout, repository=repository, revision=candidate_revision,
        path=locator.path, max_bytes=max_bytes,
    )
    if base.availability == "unavailable" or candidate.availability == "unavailable":
        change = "unavailable"
    elif base.availability == "missing" and candidate.availability == "missing":
        change = "absent"
    elif base.availability == "missing" or candidate.availability == "missing":
        change = "deleted" if base.is_available and candidate.availability == "missing" else "added" if candidate.is_available and base.availability == "missing" else "unavailable"
    elif base.blob_oid == candidate.blob_oid and base.blob_oid is not None:
        change = "unchanged"
    else:
        change = "modified"
    return ArchitecturePair(
        repository=repository, path=locator.path, target_revision=target_revision,
        candidate_revision=candidate_revision, merge_base_revision=merge_base,
        base=base, candidate=candidate, change=change,
    )


def _bounded_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    notice = "[Architecture overview truncated for prompt budget.]"
    if limit <= len(notice):
        return notice[:limit], True
    return text[: limit - len(notice) - 1] + "\n" + notice, True


def _snapshot_metadata(snapshot: ArchitectureSnapshot, label: str) -> str:
    lines = [
        f"{label} (advisory repository context; untrusted; source inspection remains required)",
        f"- Repository: {sanitize_historical_text(snapshot.repository)}",
        f"- Path: {sanitize_historical_text(snapshot.path)}",
        f"- Revision: {sanitize_historical_text(snapshot.revision or '(unavailable)')}",
        f"- Blob OID: {sanitize_historical_text(snapshot.blob_oid or '(unavailable)')}",
        f"- SHA-256: {sanitize_historical_text(snapshot.sha256 or '(unavailable)')}",
        f"- Availability: {snapshot.availability}",
    ]
    if snapshot.size is not None:
        lines.append(f"- Size: {snapshot.size} bytes")
    if snapshot.diagnostic:
        lines.append(f"- Read note: {sanitize_historical_text(snapshot.diagnostic)}")
    if snapshot.links_status == "unscanned":
        lines.append(_MEMBERSHIP_UNKNOWN_LINE)
    return "\n".join(lines)


def _fit_listing(header: str, lines: Sequence[str], budget: int, more_template: str) -> str:
    """Header plus as many lines as fit, then a count of the rest; '' if nothing fits."""
    if not lines:
        return ""
    prefix = [0]
    for line in lines:
        prefix.append(prefix[-1] + len(line))
    if len(header) + prefix[-1] + len(lines) - 1 <= budget:
        return header + "\n".join(lines)
    for count in range(len(lines) - 1, -1, -1):
        more = more_template.format(n=len(lines) - count)
        if len(header) + prefix[count] + count + len(more) <= budget:
            return header + "\n".join([*lines[:count], more])
    return ""


def _member_index(members: Sequence[ArchitectureMember], budget: int) -> str:
    """The detail-document index: path plus first heading, bounded by ``budget``."""
    if not members or budget <= 0:
        return ""
    lines: list[str] = []
    for member in sorted(members, key=lambda item: item.path):
        path = sanitize_historical_text(member.path)
        if member.availability != "available":
            lines.append(f"- {path} — ({member.availability})")
        elif member.first_heading:
            lines.append(f"- {path} — {_sanitize_architecture_text(member.first_heading)}")
        else:
            lines.append(f"- {path}")
    text = _fit_listing(
        "\n\nDetail documents in the architecture set (open the file in the assigned checkout):\n",
        lines, budget, "[{n} more members]",
    )
    if text:
        return text
    notice = (
        f"\n\nArchitecture set: {len(members)} linked detail document(s); "
        "index omitted for prompt budget."
    )
    return notice if len(notice) <= budget else ""


def _render_oversized_snapshot(snapshot: ArchitectureSnapshot, metadata: str, max_chars: int) -> str:
    """Identity, size and split diagnostic always; then bounded indexes, never prose."""
    detail = _member_index(snapshot.members, max_chars - len(metadata) - 1)
    budget = max_chars - len(metadata) - len(detail) - 1
    index = ""
    if snapshot.heading_index:
        index = _fit_listing(
            "\n\nSections available for targeted checkout inspection:\n",
            [f"- {sanitize_historical_text(heading)}" for heading in snapshot.heading_index],
            budget,
            "[{n} more headings omitted for prompt budget]",
        )
        notice = "\n\nSection index omitted for prompt budget; inspect headings in the assigned checkout."
    else:
        notice = "\n\nNo section index is available; inspect the document in the assigned checkout directly."
    if not index and len(notice) <= budget:
        index = notice
    result = metadata + index + detail
    # The trailing newline is optional: at the identity-only boundary it must
    # not push the render past its allocation.
    return result + "\n" if len(result) < max_chars else result


def render_architecture_snapshot(snapshot: ArchitectureSnapshot, *, max_chars: int = DEFAULT_ARCHITECTURE_SNAPSHOT_CHARS, label: str = "Architecture context") -> str:
    """Render advisory text while structurally retaining the full identity."""
    if max_chars <= 0:
        raise AgentLoopError("Architecture snapshot render size must be positive.")
    metadata = _snapshot_metadata(snapshot, label)
    if len(metadata) > max_chars:
        # Preserve the long-standing standalone renderer contract for callers
        # that intentionally request a tiny display bound. Pair rendering
        # performs its own mandatory-identity admission check below.
        return metadata[:max_chars]
    if snapshot.availability == "oversized":
        return _render_oversized_snapshot(snapshot, metadata, max_chars)
    # The detail-document index is budgeted after the mandatory identity and
    # before the overview prose, so the overview truncates first (#1454).
    detail = _member_index(snapshot.members, max_chars - len(metadata) - 1)
    index = ""
    if snapshot.heading_index:
        index = "\n\nSections available for targeted checkout inspection:\n" + "\n".join(
            f"- {sanitize_historical_text(heading)}" for heading in snapshot.heading_index
        )
    overview_prefix = "\n\nBounded architecture overview:\n"
    omission = "[Architecture text omitted because it is unavailable, unsafe, or not allocated.]"
    # Identity is mandatory. Indexes and prose are the first material to omit
    # under a tight budget; neither may be allowed to displace revision, blob,
    # digest, availability, or the diagnostic classification.
    truncation_status = "\n- Overview status: truncated; this is not complete coverage."
    # With a detail index, reserve room for a truncated overview's status
    # line so the render never collapses to identity-only.  Root-only
    # rendering keeps its historical budget arithmetic unchanged.
    reserve = len(omission) + (len(truncation_status) if detail and snapshot.content else 0)
    remaining = max_chars - len(metadata) - len(index) - len(detail) - len(overview_prefix) - 1
    if remaining < reserve:
        index = "\n\nSection index omitted for prompt budget; inspect headings in the assigned checkout."
        remaining = max_chars - len(metadata) - len(index) - len(detail) - len(overview_prefix) - 1
    if remaining < reserve and detail:
        detail = _member_index(
            snapshot.members,
            max_chars - len(metadata) - len(index) - len(overview_prefix) - reserve - 1,
        )
        remaining = max_chars - len(metadata) - len(index) - len(detail) - len(overview_prefix) - 1
    if remaining < len(omission):
        # Returning identity-only text is deterministic and still bounded.
        return metadata
    if snapshot.content is not None and snapshot.is_available and remaining > 0:
        sanitized = _sanitize_architecture_text(snapshot.content)
        if detail and len(sanitized) > remaining and remaining - len(truncation_status) >= len(omission):
            remaining -= len(truncation_status)
        text, truncated = _bounded_text(sanitized, remaining)
        if truncated:
            metadata += truncation_status
        result = metadata + index + overview_prefix + text + detail + "\n"
    else:
        result = metadata + index + overview_prefix + omission + detail + "\n"
    if len(result) > max_chars:
        # This can only happen when adding the truncation status consumed part
        # of the reserved space. Re-render without optional material rather
        # than slicing the identity block.
        return metadata
    return result


def _comparison_prefix(pair: ArchitecturePair) -> str:
    lines = [
        "PR architecture comparison (advisory and untrusted; never a correctness waiver)",
        f"- Change: {pair.change}",
        f"- Merge base: {pair.merge_base_revision or '(unavailable)'}",
    ]
    unscanned = [
        side for side, snapshot in (("base", pair.base), ("candidate", pair.candidate))
        if snapshot.links_status == "unscanned"
    ]
    set_change = pair.set_change
    if set_change != pair.change or pair.has_members or unscanned:
        lines.append(f"- Architecture set change: {set_change}")
    if unscanned:
        lines.append(
            "- Architecture set: membership unknown for the "
            + " and ".join(unscanned)
            + " root (root could not be scanned)"
        )
    changes = pair.member_changes
    if changes:
        shown = "; ".join(
            f"{sanitize_historical_text(path)[:_ARCHITECTURE_HEADING_CHARS]} ({kind})"
            for path, kind in changes[:_ARCHITECTURE_CHANGED_MEMBER_RENDER_CAP]
        )
        if len(changes) > _ARCHITECTURE_CHANGED_MEMBER_RENDER_CAP:
            shown += f"; [{len(changes) - _ARCHITECTURE_CHANGED_MEMBER_RENDER_CAP} more changed members]"
        lines.append(f"- Changed architecture set members: {shown}")
    return "\n".join(lines) + "\n\n"


def render_architecture_pair(pair: ArchitecturePair, *, max_chars: int = DEFAULT_ARCHITECTURE_AGGREGATE_CHARS) -> str:
    if max_chars <= 0:
        raise AgentLoopError("Architecture aggregate render size must be positive.")
    # Members are named, never quoted: only the root snapshots carry text.
    comparison_prefix = _comparison_prefix(pair)
    if len(comparison_prefix) + 2 > max_chars:
        raise AgentLoopError("Architecture aggregate budget is too small for comparison identity.")
    candidate_label = "Candidate architecture snapshot"
    if pair.change == "added":
        candidate_label += " (proposal; no established base document exists)"
    elif pair.change == "modified":
        candidate_label += " (candidate edits; do not treat as replacement for the established base)"
    base_label = "Established base architecture snapshot"
    base_identity = _snapshot_metadata(pair.base, base_label)
    candidate_identity = _snapshot_metadata(pair.candidate, candidate_label)
    mandatory = len(comparison_prefix) + len(base_identity) + 1 + len(candidate_identity)
    if mandatory > max_chars:
        raise AgentLoopError(
            "Architecture aggregate budget is too small to retain both complete snapshot identities."
        )
    extra = max_chars - mandatory
    # Allocate optional indexes/overviews structurally. Each side receives its
    # identity budget first, so a large base document can never erase the
    # candidate label, proposal state, or immutable identity.
    base_budget = len(base_identity) + extra // 2
    candidate_budget = len(candidate_identity) + extra - extra // 2
    base = render_architecture_snapshot(pair.base, max_chars=base_budget, label=base_label)
    candidate = render_architecture_snapshot(pair.candidate, max_chars=candidate_budget, label=candidate_label)
    result = comparison_prefix + base + "\n" + candidate
    if len(result) > max_chars:
        raise AgentLoopError("Architecture aggregate renderer exceeded its structural budget.")
    return result


def architecture_material(context: object | None) -> bool:
    return bool(getattr(context, "material", False))


def _context_sides(context: object | None) -> tuple[ArchitectureSnapshot, ...]:
    if isinstance(context, ArchitecturePair):
        return (context.base, context.candidate)
    if isinstance(context, ArchitectureSnapshot):
        return (context,)
    return ()


def architecture_injectable(context: object | None) -> bool:
    """Prompt gate for the architecture block only (#1454).

    Material content, or an oversized root whose identity, size and split
    diagnostic must never be silently dropped.  Every other material-gated
    prompt mode keeps using :func:`architecture_material`.
    """
    if architecture_material(context):
        return True
    return any(side.availability == "oversized" for side in _context_sides(context))


def architecture_set_guidance_applicable(context: object | None) -> bool:
    """Whether set-specific architecture-impact guidance applies."""
    return any(
        side.members or side.availability == "oversized" for side in _context_sides(context)
    )


def architecture_set_paths(context: object | None) -> frozenset[str] | None:
    """The root plus every linked member path, or ``None`` when membership is unknowable.

    ``None`` (skip validation) covers a disabled or absent context, a missing
    or unavailable root, and any side whose root could not be scanned.
    """
    sides = _context_sides(context)
    if not sides or any(side.links_status == "unscanned" for side in sides):
        return None
    scanned = [side for side in sides if side.links_status == "scanned"]
    if not scanned:
        return None
    paths: set[str] = set()
    for side in scanned:
        paths.add(side.path)
        paths.update(member.path for member in side.members)
    return frozenset(paths)


def resolve_architecture_document_path(value: object) -> str | None:
    """Normalize a reported repository-relative path with the link resolver."""
    return _resolve_repo_relative((), value)


def freeze_architecture_context(
    runner: Runner,
    *,
    checkout,
    repository: str,
    path: str = DEFAULT_ARCHITECTURE_PATH,
    read_size: int = DEFAULT_ARCHITECTURE_READ_SIZE,
    target_revision: str | None = None,
    candidate_revision: str | None = None,
) -> ArchitectureSnapshot | ArchitecturePair | None:
    """Acquire one frozen prompt identity for an issue or PR turn."""
    if candidate_revision is None:
        result = runner.run(("git", "rev-parse", "HEAD"), cwd=checkout, check=False)
        candidate_revision = result.stdout.strip() if result.returncode == 0 else None
    if not candidate_revision:
        return None
    if target_revision is None:
        return acquire_architecture_snapshot(
            runner, checkout=checkout, repository=repository, revision=candidate_revision,
            path=path, max_bytes=read_size,
        )
    return acquire_architecture_pair(
        runner, checkout=checkout, repository=repository,
        target_revision=target_revision, candidate_revision=candidate_revision,
        path=path, max_bytes=read_size,
    )
