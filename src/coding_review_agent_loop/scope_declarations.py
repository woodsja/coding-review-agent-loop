"""Operator-declared out-of-scope sections (#1389).

An issue body, or a signed human comment, may carry an ``## Out of scope`` or
``## Non-goals`` heading section.  These helpers extract those sections and
render them as a binding-scope block for the planner and every plan and PR
reviewer.  The output depends only on the issue context, so it is identical
across rounds and safe inside compact stable prompt prefixes.

Scope declarations are never signed-requirement IDs; they only steer reviewers
to downgrade blocks that concern a declared area to future follow-ups.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence

from .github import IssueContext
from .protocol_markers import sanitize_untrusted_prose

_HEADING_RE = re.compile(r"^ {0,3}(?P<hashes>#{1,6})(?:[ \t]+(?P<text>.*?))?[ \t]*#*[ \t]*$")
_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})")
# A closing fence carries no info string: only whitespace may follow it.
_CLOSING_FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*$")
_SCOPE_HEADING_TEXT_RE = re.compile(
    r"^(?:out[ -]of[ -]scope|non[- ]?goals)(?:\s+for\s+this\s+(?:issue|pr|change|plan))?\s*:?$",
    re.IGNORECASE,
)

DEFAULT_SCOPE_BLOCK_MAX_CHARS = 6000


@dataclass(frozen=True)
class DeclaredOutOfScope:
    source_label: str
    text: str


def _normalized_heading_text(text: str | None) -> str:
    value = (text or "").strip()
    # Accept emphasis around the heading text, e.g. ``## **Non-goals**``.
    value = value.strip("*_ ").strip()
    return " ".join(value.split())


def _extract_sections(text: str | None) -> list[str]:
    if not text:
        return []
    sections: list[str] = []
    capture_level: int | None = None
    captured: list[str] = []
    fence: str | None = None

    def flush() -> None:
        nonlocal captured
        body = "\n".join(captured).strip()
        if body:
            sections.append(body)
        captured = []

    for line in text.splitlines():
        fence_match = _FENCE_RE.match(line)
        if fence is not None:
            closing = _CLOSING_FENCE_RE.match(line)
            if closing and closing.group("fence")[0] == fence[0] and len(
                closing.group("fence")
            ) >= len(fence):
                fence = None
            if capture_level is not None:
                captured.append(line)
            continue
        if fence_match:
            fence = fence_match.group("fence")
            if capture_level is not None:
                captured.append(line)
            continue
        heading = _HEADING_RE.match(line)
        if heading:
            level = len(heading.group("hashes"))
            if capture_level is not None and level <= capture_level:
                flush()
                capture_level = None
            if capture_level is None:
                if _SCOPE_HEADING_TEXT_RE.match(_normalized_heading_text(heading.group("text"))):
                    capture_level = level
                continue
        if capture_level is not None:
            captured.append(line)
    if capture_level is not None:
        flush()
    return sections


def _comment_label(author: str | None, url: str | None) -> str:
    label = f"signed human comment by {author or 'unknown author'}"
    if url:
        label += f" ({url})"
    return label


def extract_declared_out_of_scope(
    issue_context: IssueContext | None,
) -> tuple[DeclaredOutOfScope, ...]:
    """Return the out-of-scope sections from the issue body and signed comments."""
    if issue_context is None:
        return ()
    found: list[DeclaredOutOfScope] = []
    seen: set[str] = set()

    def add(label: str, text: str | None) -> None:
        for section in _extract_sections(text):
            key = " ".join(section.split()).casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(DeclaredOutOfScope(source_label=label, text=section))

    add("issue body", issue_context.body)
    for requirement in issue_context.human_requirements:
        add(_comment_label(requirement.author, requirement.url), requirement.body)
    return tuple(found)


def collect_declared_out_of_scope(
    *contexts: IssueContext | None,
) -> tuple[DeclaredOutOfScope, ...]:
    """Merge sections from several issue contexts, deduplicating by text."""
    merged: list[DeclaredOutOfScope] = []
    seen: set[str] = set()
    for context in contexts:
        for section in extract_declared_out_of_scope(context):
            key = " ".join(section.text.split()).casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(section)
    return tuple(merged)


def render_declared_out_of_scope_block(
    sections: Sequence[DeclaredOutOfScope],
    *,
    max_chars: int = DEFAULT_SCOPE_BLOCK_MAX_CHARS,
) -> str:
    """Render the binding-scope block, or ``''`` when nothing is declared."""
    if not sections:
        return ""
    lines = [
        "Operator-declared out of scope (binding):",
        "The areas below are outside this issue's scope. Plans must not add work for "
        "them. A review finding that only concerns a declared out-of-scope area must be "
        "a `future_followups` entry (or omitted), not blocking, unless it is also a "
        "regression or an introduced security/data-loss risk under blocking scope "
        "criteria (ii)/(iii). A later signed human instruction overrides this "
        "declaration.",
        "",
    ]
    for section in sections:
        lines.append(f"From {sanitize_untrusted_prose(section.source_label)}:")
        lines.append(sanitize_untrusted_prose(section.text))
        lines.append("")
    block = "\n".join(lines).rstrip() + "\n"
    if len(block) > max_chars:
        omitted = len(block) - max_chars
        block = block[:max_chars].rstrip() + f"\n[truncated: {omitted} chars omitted]\n"
    return block


def declared_out_of_scope_block(*contexts: IssueContext | None) -> str:
    """Extract and render the binding-scope block for prompt builders."""
    block = render_declared_out_of_scope_block(collect_declared_out_of_scope(*contexts))
    return f"{block}\n" if block else ""
