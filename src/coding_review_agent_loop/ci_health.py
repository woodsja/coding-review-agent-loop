"""CI infrastructure-stall detection.

Owns the check data types (`PullRequestCheck`, `PullRequestChecks`) so this
module has no dependency on `github.py`'s live API calls; `github.py` imports
and re-exports these types instead. Everything here is pure classification
over already-fetched data, so it only depends on stdlib.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Sequence


@dataclass(frozen=True)
class PullRequestCheck:
    name: str
    kind: Literal["check_run", "status_context"]
    status: str
    url: str | None = None
    check_id: int | None = None
    run_id: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    creator_login: str | None = None
    creator_id: int | None = None
    description: str | None = None
    # The check run's own check_suite id and app slug, as reported by the API.
    # They are the only join keys for ad hoc workflow_dispatch exclusion; the
    # URL-derived ``run_id`` is display/consistency only.
    check_suite_id: int | None = None
    app_slug: str | None = None


@dataclass(frozen=True)
class StalledCheck:
    name: str
    kind: Literal["check_run", "status_context"]
    reason: Literal["queued_too_long", "runner_unavailable"]
    check_id: int | None
    run_id: str | None
    url: str | None
    age_seconds: float | None

    def describe(self) -> str:
        location_parts: list[str] = []
        if self.check_id is not None:
            location_parts.append(f"check run {self.check_id}")
        if self.run_id is not None:
            location_parts.append(f"workflow run {self.run_id}")
        if self.url:
            location_parts.append(self.url)
        location = f" ({', '.join(location_parts)})" if location_parts else ""
        if self.reason == "queued_too_long":
            detail = f"queued {_format_age(self.age_seconds)} without starting a job"
        else:
            detail = "cancelled before any job started (runner unavailable)"
        return f"{self.name}{location} — {detail}"

    def identifiers(self) -> tuple[str, ...]:
        raw_ids = [self.name, self.run_id, str(self.check_id) if self.check_id is not None else None, self.url]
        seen: list[str] = []
        for raw_id in raw_ids:
            if raw_id and raw_id not in seen:
                seen.append(raw_id)
        return tuple(seen)


@dataclass(frozen=True)
class PullRequestChecks:
    state: Literal["passing", "failing", "pending", "no_checks", "unavailable"]
    required_checks: tuple[str, ...]
    passing: tuple[PullRequestCheck, ...]
    pending: tuple[PullRequestCheck, ...]
    failing: tuple[PullRequestCheck, ...]
    missing_required: tuple[str, ...]
    branch_protection_status: Literal["configured", "not_found", "forbidden", "unavailable"]
    branch_protection_note: str | None = None
    check_query_status: Literal["ok", "partial", "unavailable"] = "ok"
    check_query_errors: tuple[str, ...] = ()
    infrastructure_stalls: tuple[StalledCheck, ...] = field(default=())
    # Same-name observations dropped by first-wins deduplication, in API order.
    shadowed: tuple[PullRequestCheck, ...] = field(default=())
    # True only when get_pr_checks saw every counted check-run and status entry
    # and parsed each into a check. Fails closed for every other constructor.
    listing_complete: bool = False
    # Check runs removed from the counted board because they belong to an ad
    # hoc (non-managed) workflow_dispatch run, plus one audit note per
    # exclusion, conflict, lookup problem or fail-closed fallback.
    excluded: tuple[PullRequestCheck, ...] = field(default=())
    exclusion_notes: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class CiInfrastructureStall:
    checks: tuple[StalledCheck, ...]

    @property
    def is_stalled(self) -> bool:
        return bool(self.checks)


# Run name prefix of the managed qualification dispatch; shared with
# managed_ci._v2_run_name so the PR-checks gate always counts that run.
V2_RUN_NAME_PREFIX = "managed-ci-v2 nonce="


# Every exclusion note ends with this suffix; other exclusion_notes entries
# (conflicts, dispatch-lookup problems, non-authoritative fallback) are diagnostics.
EXCLUSION_NOTE_SUFFIX = ": not a PR check"


def partition_exclusion_notes(checks: "PullRequestChecks") -> tuple[list[str], list[str]]:
    """Split ``exclusion_notes`` into (exclusion lines, diagnostic lines)."""
    exclusions: list[str] = []
    diagnostics: list[str] = []
    for note in checks.exclusion_notes:
        (exclusions if note.endswith(EXCLUSION_NOTE_SUFFIX) else diagnostics).append(note)
    if checks.excluded and not exclusions:
        exclusions = [f"{c.name} ({c.status.lower()})" for c in checks.excluded]
    return exclusions, diagnostics


def decode_concatenated_json_objects(raw: str) -> list[dict] | None:
    """Decode back-to-back JSON object pages (``gh api --paginate`` output).

    Returns None on any decode error, non-object page or empty stream.
    """
    decoder = json.JSONDecoder()
    pages: list[dict] = []
    offset = 0
    try:
        while offset < len(raw):
            while offset < len(raw) and raw[offset].isspace():
                offset += 1
            if offset == len(raw):
                break
            page, end = decoder.raw_decode(raw, offset)
            if not isinstance(page, dict):
                return None
            pages.append(page)
            offset = end
    except (json.JSONDecodeError, ValueError):
        return None
    return pages or None


_RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")


def is_canonical_pending_only_text(text: str, *, check_names: Sequence[str]) -> bool:
    """Recognize only whole, simple CI-status statements, never keyword hits.

    Check names are allowed only in the subject slot. Substituting them across
    arbitrary prose would erase meaningful words such as ``test`` in findings.
    Unrecognized wording is intentionally left blocking for normal review.
    """
    if not text or len(text) > 400:
        return False
    names = "|".join(re.escape(name) for name in check_names if name.strip())
    subject = r"(?:GitHub(?: PR)? (?:checks?|check status)|CI(?: checks?)?)"
    if names:
        name = rf"(?:`(?:{names})`|(?:{names}))"
        subject = rf"(?:{subject}(?: {name})?|check {name}|{name})"
    status = r"(?:pending(?:/in_progress)?|in[_ ]progress|queued|running|unavailable|not yet reporting)"
    return re.fullmatch(
        rf"(?:The )?{subject} (?:is|are|remains?) (?:still |currently )?{status}[.!]?",
        text.strip(),
        flags=re.IGNORECASE,
    ) is not None


def _extract_run_id(url: str | None) -> str | None:
    if not url:
        return None
    match = _RUN_ID_RE.search(url)
    return match.group(1) if match else None


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_seconds(created_at: str | None, *, now: datetime) -> float | None:
    parsed = _parse_timestamp(created_at)
    if parsed is None:
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = (now - parsed).total_seconds()
    return max(0.0, delta)


def _format_age(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "for an unknown duration"
    minutes = int(age_seconds // 60)
    return f"{minutes}m"


_QUEUED_LIKE_STATUSES = {"queued", "pending", "waiting", "requested"}
_UNSTARTED_CANCEL_CONCLUSIONS = {"cancelled", "startup_failure"}


def classify_ci_infrastructure_stall(
    checks: Sequence[PullRequestCheck],
    *,
    now: datetime,
    grace_seconds: int,
) -> CiInfrastructureStall:
    """Classify individual checks as external-infrastructure stalls.

    Only ``check_run`` entries are eligible: commit-status contexts
    (``status_context``) are set by arbitrary external systems, not GitHub
    Actions runners, so they are never classified as an infrastructure stall.
    """
    stalled: list[StalledCheck] = []
    for check in checks:
        if check.kind != "check_run":
            continue
        status = (check.status or "").strip().lower()

        if status in _UNSTARTED_CANCEL_CONCLUSIONS and (
            check.started_at is None or check.started_at == check.completed_at
        ):
            stalled.append(
                StalledCheck(
                    name=check.name,
                    kind=check.kind,
                    reason="runner_unavailable",
                    check_id=check.check_id,
                    run_id=check.run_id,
                    url=check.url,
                    age_seconds=_age_seconds(check.created_at, now=now),
                )
            )
            continue

        if status in _QUEUED_LIKE_STATUSES and check.started_at is None:
            age = _age_seconds(check.created_at, now=now)
            if age is not None and age > grace_seconds:
                stalled.append(
                    StalledCheck(
                        name=check.name,
                        kind=check.kind,
                        reason="queued_too_long",
                        check_id=check.check_id,
                        run_id=check.run_id,
                        url=check.url,
                        age_seconds=age,
                    )
                )

    return CiInfrastructureStall(checks=tuple(stalled))


def is_wholly_infrastructure_blocked(pr_checks: PullRequestChecks) -> bool:
    """True only when every blocking/pending signal is an infrastructure stall.

    This is the single gate for the orchestrator terminal stop, the reviewer
    downgrade, and the auto-merge infrastructure outcome. A never-reporting
    required check, unknown branch protection, a partial/unavailable check
    query, or any genuinely failing/pending check makes this False.
    """
    if not pr_checks.infrastructure_stalls:
        return False
    if pr_checks.missing_required:
        return False
    if pr_checks.state == "unavailable":
        return False
    if pr_checks.branch_protection_status == "unavailable":
        return False
    if pr_checks.check_query_status != "ok":
        return False
    stalled_keys = {(stall.kind, stall.name) for stall in pr_checks.infrastructure_stalls}
    for check in (*pr_checks.failing, *pr_checks.pending):
        if (check.kind, check.name) not in stalled_keys:
            return False
    return True


_IDENTIFIER_SENTINEL = "identsentinel"

# Substring phrases whose presence signals stall semantics. Checked against the
# normalized text after identifier substitution.
_STALL_SEMANTIC_PHRASES: tuple[str, ...] = (
    "runner unavailable",
    "hosted runner",
    "never started",
    "did not start",
    "no runner",
    "before execution",
    "runner acquisition",
    "queued",
    "cancelled",
    "canceled",
    "runner",
    "capacity",
)

# Closed word-level vocabulary: every remaining token in a canonical stall-only
# text must belong to this set (plus digits, the identifier sentinel, and
# URLs) or the match fails. Kept intentionally small so an out-of-vocabulary
# word (a filename, symbol, or code-defect noun) fails the match.
_STALL_VOCABULARY_WORDS: frozenset[str] = frozenset(
    {
        "queued", "cancelled", "canceled", "runner", "runners", "hosted",
        "never", "started", "starting", "start", "did", "no", "not",
        "unavailable", "before", "execution", "capacity", "acquire",
        "acquired", "acquisition", "recover", "recovers", "recovered",
    }
)

_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "is", "was", "were", "this",
        "that", "these", "those", "for", "of", "to", "with", "without",
        "than", "over", "past", "its", "it", "has", "have", "had", "been",
        "being", "be", "still", "remains", "remained", "remain", "so",
        "because", "due", "on", "in", "at", "as", "any", "job", "jobs",
        "more", "external", "infrastructure", "issue", "since", "once",
        "will", "should", "resume", "after", "work", "affected", "which",
        "there", "not", "all", "one", "run", "runs", "runner",
    }
)

_CI_NOUNS: frozenset[str] = frozenset(
    {
        "check", "checks", "workflow", "workflows", "run", "runs", "job",
        "jobs", "github", "actions", "ci", "infrastructure", "external",
    }
)

_TIME_UNITS: frozenset[str] = frozenset(
    {
        "m", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour",
        "hours", "s", "sec", "secs", "second", "seconds",
    }
)

_CLOSED_VOCABULARY: frozenset[str] = (
    _STALL_VOCABULARY_WORDS | _FUNCTION_WORDS | _CI_NOUNS | _TIME_UNITS
)

_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_DIGIT_LETTER_BOUNDARY_RE = re.compile(r"(\d)([a-z])")
_LETTER_DIGIT_BOUNDARY_RE = re.compile(r"([a-z])(\d)")
_LEADING_MARKDOWN_RE = re.compile(r"^[\s\-\*•`'\"]+")

_MAX_CANONICAL_TEXT_LENGTH = 400


def is_canonical_stall_only_text(text: str, *, stalls: Sequence[StalledCheck]) -> bool:
    """Whole-text, closed-vocabulary check for "this item is only a CI stall".

    Fails closed: any token outside the closed vocabulary (a filename,
    symbol, or code-defect noun) makes this return False, so a mixed item
    naming a stalled run and then describing an unrelated code defect is
    rejected in full.
    """
    if not text or not text.strip():
        return False
    if len(text) > _MAX_CANONICAL_TEXT_LENGTH:
        return False

    normalized = text.strip().lower()
    normalized = _LEADING_MARKDOWN_RE.sub("", normalized)
    normalized = normalized.strip("`'\" \t")
    if not normalized:
        return False

    identifiers = sorted(
        {identifier.lower() for stall in stalls for identifier in stall.identifiers() if identifier},
        key=len,
        reverse=True,
    )

    substituted = normalized
    sentinel_count = 0
    for identifier in identifiers:
        if identifier and identifier in substituted:
            sentinel_count += substituted.count(identifier)
            substituted = substituted.replace(identifier, f" {_IDENTIFIER_SENTINEL} ")

    if sentinel_count == 0:
        return False

    substituted = _URL_RE.sub(" url ", substituted)

    if not any(phrase in substituted for phrase in _STALL_SEMANTIC_PHRASES):
        return False

    tokenizable = _DIGIT_LETTER_BOUNDARY_RE.sub(r"\1 \2", substituted)
    tokenizable = _LETTER_DIGIT_BOUNDARY_RE.sub(r"\1 \2", tokenizable)

    for token in _WORD_RE.findall(tokenizable):
        if token == _IDENTIFIER_SENTINEL or token == "url":
            continue
        if token.isdigit():
            continue
        if token in _CLOSED_VOCABULARY:
            continue
        return False

    return True


# --- Post-approval plumbing classification (#1477) --------------------------

WatchFailureClass = Literal[
    "test-failure",
    "plumbing-only",
    "plumbing-awaiting-tests",
    "stuck-infrastructure",
    "not-classifiable",
]

_PASSED_STATUSES = frozenset({"success", "neutral"})


def check_name_matches(name: str, names: frozenset[str] | set[str] | Sequence[str]) -> bool:
    """Exact whole-name or exact final `` / ``-segment equality, never substring.

    GitHub names a reusable-workflow job ``caller / job``; only that final
    segment, compared exactly, may stand in for the catalogued job name.
    """
    candidates = {candidate for candidate in names if candidate}
    if not name or not candidates:
        return False
    if name in candidates:
        return True
    segment = name.rsplit(" / ", 1)[-1] if " / " in name else None
    return segment is not None and segment in candidates


def board_tests_passed(
    board: PullRequestChecks, plumbing_names: frozenset[str] | set[str] | Sequence[str]
) -> bool:
    """True when every non-plumbing check in the board's passing bucket passed.

    The passing bucket also holds ``skipped`` checks; a skipped test did not
    pass, so it never supports a plumbing-only decision on either route.
    """
    return all(
        (check.status or "").strip().lower() in _PASSED_STATUSES
        for check in board.passing
        if not check_name_matches(check.name, plumbing_names)
    )


def classify_watch_failure(
    board: PullRequestChecks,
    plumbing_names: frozenset[str] | set[str] | Sequence[str],
    stuck_contexts: Sequence[PullRequestCheck] = (),
    *,
    stuck_without_failed_jobs: Sequence[str] = (),
    stuck_with_test_failures: Sequence[str] = (),
) -> WatchFailureClass:
    """Classify a post-approval CI failure before any coder routing.

    ``stuck_contexts`` are confirmed stuck-pending required statuses; they are
    plumbing by cause whatever their name.  ``stuck_without_failed_jobs``
    names stuck contexts whose completed producing run has no failed job, so
    a ``--failed`` rerun would be ineffective; ``stuck_with_test_failures``
    names stuck contexts whose producing run has a failed non-plumbing job.
    """
    stuck_keys = {(check.kind, check.name) for check in stuck_contexts}
    # (a) Test precedence: any failing non-plumbing check, or a stuck status
    # whose producing run failed a test job, reaches the coder.
    if any(not check_name_matches(check.name, plumbing_names) for check in board.failing):
        return "test-failure"
    if any(name in stuck_with_test_failures for _kind, name in stuck_keys):
        return "test-failure"
    # (b) Only an authoritative, complete board can be classified plumbing.
    if not (
        board.check_query_status == "ok"
        and board.listing_complete
        and board.branch_protection_status in {"configured", "not_found"}
        and not board.missing_required
    ):
        return "not-classifiable"
    if not board.failing and not stuck_keys:
        return "not-classifiable"
    if stuck_keys and any(name in stuck_without_failed_jobs for _kind, name in stuck_keys):
        return "stuck-infrastructure"
    unsettled = [
        check for check in board.pending if (check.kind, check.name) not in stuck_keys
    ]
    tests = [check for check in board.passing if not check_name_matches(check.name, plumbing_names)]
    if not board_tests_passed(board, plumbing_names):
        # A test skipped as a dependent of a failed plumbing job did not pass.
        return "not-classifiable"
    if unsettled:
        # (c2) Hold while tests (or plumbing) are still running.
        return "plumbing-awaiting-tests"
    if not tests:
        return "not-classifiable"
    return "plumbing-only"


RunJobsClass = Literal["plumbing-only", "test-failure", "no-failed-jobs", "not-classifiable"]

FAILED_JOB_CONCLUSIONS = frozenset(
    {"failure", "cancelled", "timed_out", "action_required", "startup_failure"}
)


def classify_run_jobs(
    jobs: Sequence[tuple[str, str, str]] | None,
    plumbing_names: frozenset[str] | set[str] | Sequence[str],
    *,
    require_passing_tests: bool = False,
) -> RunJobsClass:
    """Classify one workflow run from its complete latest-attempt job listing.

    Shared by every rerun decision (#1477): a failed non-plumbing job is a
    test failure; a run with no failed job has nothing for a rerun to retry;
    an unreadable listing is never plumbing-only.  ``require_passing_tests``
    also demands that every non-plumbing job succeeded (the managed route,
    where the run's jobs are the only test evidence).
    """
    if jobs is None:
        return "not-classifiable"
    failed = [job for job in jobs if job[1] in FAILED_JOB_CONCLUSIONS]
    if any(not check_name_matches(name, plumbing_names) for name, _c, _u in failed):
        return "test-failure"
    if not failed:
        return "no-failed-jobs"
    if require_passing_tests:
        tests = [job for job in jobs if not check_name_matches(job[0], plumbing_names)]
        if not tests or any(conclusion != "success" for _n, conclusion, _u in tests):
            return "not-classifiable"
    return "plumbing-only"
