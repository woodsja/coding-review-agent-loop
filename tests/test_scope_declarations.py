"""Operator-declared out-of-scope sections (#1389)."""

from __future__ import annotations

import pytest

from coding_review_agent_loop.github import HumanReviewRequirement, IssueContext
from coding_review_agent_loop.scope_declarations import (
    DeclaredOutOfScope,
    collect_declared_out_of_scope,
    declared_out_of_scope_block,
    extract_declared_out_of_scope,
    render_declared_out_of_scope_block,
)


def _context(body, *requirements):
    return IssueContext(
        number=1389,
        repo="owner/repo",
        title="Scope",
        body=body,
        url="https://github.com/owner/repo/issues/1389",
        comments=(),
        human_requirements=tuple(requirements),
    )


def _requirement(body, *, author="operator", url="https://github.com/owner/repo/issues/1389#c1"):
    return HumanReviewRequirement(
        source_type="issue_comment", author=author, created_at="2026-10-11T00:00:00Z",
        url=url, body=body,
    )


@pytest.mark.parametrize("hashes", ["#", "##", "###", "####", "#####", "######"])
def test_every_atx_heading_level_is_recognised(hashes):
    sections = extract_declared_out_of_scope(_context(f"{hashes} Out of scope\n- Cross-process locks\n"))
    assert sections == (DeclaredOutOfScope("issue body", "- Cross-process locks"),)


@pytest.mark.parametrize(
    "heading",
    [
        "Out of scope", "out of scope:", "OUT OF SCOPE", "Out-of-scope", "Non-goals",
        "Non-goals:", "non goals", "Nongoals", "Non-goals for this issue:", "**Non-goals**",
    ],
)
def test_heading_variants(heading):
    sections = extract_declared_out_of_scope(_context(f"## {heading}\nCrash recovery.\n"))
    assert [section.text for section in sections] == ["Crash recovery."]


def test_unrelated_headings_and_inline_text_are_not_sections():
    body = "## Goals\nShip it.\n\nNon-goals: nothing\n## Scope\nAll of it.\n"
    assert extract_declared_out_of_scope(_context(body)) == ()


def test_section_ends_at_same_or_higher_heading_but_keeps_lower_subheadings():
    body = (
        "# Title\n"
        "## Out of scope\n"
        "- Signal handling\n"
        "### Details\n"
        "- Deploy gates\n"
        "## Acceptance\n"
        "- Tests pass\n"
    )
    (section,) = extract_declared_out_of_scope(_context(body))
    assert section.text == "- Signal handling\n### Details\n- Deploy gates"
    higher = extract_declared_out_of_scope(_context("### Non-goals\n- A\n# Next\n- B\n"))
    assert [item.text for item in higher] == ["- A"]


def test_headings_inside_fenced_code_are_ignored():
    body = (
        "```markdown\n"
        "## Out of scope\n"
        "- Not a real section\n"
        "```\n"
        "~~~\n"
        "## Non-goals\n"
        "~~~\n"
        "## Body\n"
        "text\n"
    )
    assert extract_declared_out_of_scope(_context(body)) == ()


def test_fenced_code_inside_a_section_is_captured_and_does_not_end_it():
    body = "## Out of scope\n```\n## Acceptance\n```\n- Locks\n## Next\n"
    (section,) = extract_declared_out_of_scope(_context(body))
    assert section.text == "```\n## Acceptance\n```\n- Locks"


def test_empty_sections_are_skipped():
    assert extract_declared_out_of_scope(_context("## Out of scope\n\n   \n## Next\nx\n")) == ()
    assert extract_declared_out_of_scope(_context("## Non-goals\n")) == ()
    assert extract_declared_out_of_scope(None) == ()
    assert extract_declared_out_of_scope(_context(None)) == ()


def test_signed_comment_sections_carry_source_labels_and_duplicates_collapse():
    context = _context(
        "## Non-goals\n- Step-back changes\n",
        _requirement("### Out of scope:\n- Re-scoping on resume\n"),
        _requirement("## Non-goals\n-   Step-back   changes\n", author="other", url=None),
    )
    sections = extract_declared_out_of_scope(context)
    assert sections == (
        DeclaredOutOfScope("issue body", "- Step-back changes"),
        DeclaredOutOfScope(
            "signed human comment by operator (https://github.com/owner/repo/issues/1389#c1)",
            "- Re-scoping on resume",
        ),
    )


def test_collect_merges_parent_and_child_contexts_with_dedupe():
    parent = _context("## Out of scope\n- Shared\n")
    child = _context("## Out of scope\n- Shared\n## Non-goals\n- Child only\n")
    assert [section.text for section in collect_declared_out_of_scope(parent, None, child)] == [
        "- Shared", "- Child only",
    ]


def test_no_sections_render_nothing():
    assert render_declared_out_of_scope_block(()) == ""
    assert declared_out_of_scope_block(_context("## Goals\nx\n")) == ""
    assert declared_out_of_scope_block(None) == ""


def test_render_is_binding_deterministic_and_labelled():
    context = _context("## Non-goals\n- Step-back changes\n", _requirement("## Out of scope\n- Locks\n"))
    first = declared_out_of_scope_block(context)
    assert first == declared_out_of_scope_block(context)
    assert first.startswith("Operator-declared out of scope (binding):\n")
    assert "must be a `future_followups` entry (or omitted), not blocking" in first
    assert "criteria (ii)/(iii)" in first
    assert "A later signed human instruction overrides this declaration." in first
    assert "From issue body:\n- Step-back changes" in first
    assert "From signed human comment by operator (https://github.com/owner/repo/issues/1389#c1):\n- Locks" in first


def test_render_is_bounded_with_truncation_note():
    sections = (DeclaredOutOfScope("issue body", "x" * 500),)
    block = render_declared_out_of_scope_block(sections, max_chars=200)
    assert block.endswith(" chars omitted]\n")
    assert "[truncated: " in block
    assert len(block) < 260


def test_render_uses_the_issue_context_prose_sanitizer():
    from coding_review_agent_loop.protocol_markers import RESERVED_MARKER_REGISTRY

    token = RESERVED_MARKER_REGISTRY[0].token
    sections = (DeclaredOutOfScope("issue body", f"Do not touch {token} handling."),)
    block = render_declared_out_of_scope_block(sections)
    assert token not in block
    assert RESERVED_MARKER_REGISTRY[0].safe_label in block


@pytest.mark.parametrize(
    ("opening", "suffixed"),
    [("```markdown", "```python"), ("~~~markdown", "~~~python"), ("```", "``` x"), ("~~~", "~~~ x")],
)
def test_fence_delimiter_with_info_string_does_not_close_an_open_fence(opening, suffixed):
    body = "\n".join(
        [opening, suffixed, "## Non-goals", "- Parser validation", opening[:3], "## Real", "text"]
    )
    assert extract_declared_out_of_scope(_context(body)) == ()


def test_bare_closing_fence_with_trailing_whitespace_still_closes():
    body = "```python\n## Non-goals\n- example\n```   \n## Out of scope\n- Locks\n"
    assert [section.text for section in extract_declared_out_of_scope(_context(body))] == ["- Locks"]
