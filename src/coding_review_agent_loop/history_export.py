"""Read-only outcome history export (#1433, stage 1 of #1430).

``agent-loop history export`` builds one row per issue from the round records
and plain stop diagnostics the authenticated actor posted on the issue and on
the PRs GitHub links to it as closing PRs.  Derivation is a pure function of
already-read authenticated comment views (:func:`derive_history_row`); the CLI
driver (:func:`export_history`) only reads GitHub, never writes, and starts no
agent.

Every count column is a ``{"value": ..., "evidence": ...}`` pair.  ``verified``
means a recognized terminal state (plan approval, a merge or close, a closed
issue, or a recognized stop) bounds the history; ``unavailable`` carries a
``None`` value and is never a zero.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .agent_failure import _QUOTA_RATE_LIMIT_RE
from .agents.registry import BACKENDS
from .checks import CI_INFRASTRUCTURE_COMMENT_HEADLINE, CI_INFRASTRUCTURE_STOP_HEADLINE
from .errors import AgentLoopError
from .github import (
    AuthenticatedComment,
    AuthenticatedCommentView,
    IssueOutcomeHistory,
    LinkedPullRequest,
    parse_comment_timestamp,
)
from .panel_evidence import (
    PLAN_PRIMARY_STALL_PREFIX,
    normalize_plan_issue_text,
    plan_issue_text_digest_for,
)
from .plan_first_loop import PLAN_SCHEDULING_DIAGNOSTIC_TEMPLATE
from .pr_loop import PR_SCHEDULING_DIAGNOSTIC_TEMPLATE
from .protocol import parse_agent_unavailable
from .review_step_back import PLAN_STEP_BACK_HUMAN_DECISION_PREFIX
from .round_state import PostedRoundRecord, _extract_round_metadata_records
from .round_transport import ROUND_RESUME_MARKER_RE
from .transient import NON_RETRYABLE_AGENT_OUTPUT_RE, is_transient_agent_output

if TYPE_CHECKING:
    from .config import AgentLoopConfig

SCHEMA_VERSION = 1

EVIDENCE_VERIFIED = "verified"
EVIDENCE_UNAVAILABLE = "unavailable"

ROUTE_DISCUSS = "discuss"
ROUTE_PLAN_FIRST = "plan-first"
ROUTE_DIRECT = "direct"

SNAPSHOT_DIGEST_VERIFIED = "digest-verified"
SNAPSHOT_DIGEST_NORMALIZED = "digest-normalized"
SNAPSHOT_NEVER_EDITED = "never-edited"
SNAPSHOT_TIME_INFERRED = "time-inferred"
SNAPSHOT_UNAVAILABLE = "unavailable"

COVERAGE_COMPLETE = "complete"
COVERAGE_PARTIAL = "partial"

KIND_HUMAN_DECISION_STOP = "human-decision-stop"
KIND_PLANNING_STOP = "planning-stop"
KIND_PR_STOP = "pr-stop"
KIND_INFRASTRUCTURE = "infrastructure"
STOP_KINDS = frozenset({KIND_HUMAN_DECISION_STOP, KIND_PLANNING_STOP, KIND_PR_STOP})

CAUSE_UNKNOWN = "unknown"

# Records carrying any of these fields were written after plan step-back
# entries (#1251) existed, so a plan history without entries is a verified
# zero rather than a legacy unknown.
_PLAN_STEP_BACK_ERA_FIELDS = ("plan_execution_mode", "scheduler_execution_mode", "plan_issue_digest")
# PR board lineage (#1376) postdates PR step-back entries.
_PR_STEP_BACK_ERA_FIELDS = ("pr_board_root", "pr_board_identity")

_REVIEW_ROLES = frozenset({"coder", "reviewer"})


# ---------------------------------------------------------------------------
# Diagnostic recognition (render-side templates, never duplicated regexes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecognizedDiagnostic:
    """One authenticated plain diagnostic that carries no round metadata."""

    kind: str
    reason: str
    round_number: int | None = None
    seat_signature: str | None = None


def _template_regex(template: str, groups: Mapping[str, str]) -> re.Pattern[str]:
    """Compile a ``str.format`` template into an anchored regex.

    Literal text is escaped; each ``{name}`` becomes the named group given in
    ``groups``, so the recognizer accepts exactly the bytes the poster renders.
    """
    pattern = ""
    position = 0
    for match in re.finditer(r"\{(\w+)\}", template):
        pattern += re.escape(template[position:match.start()])
        pattern += f"(?P<{match.group(1)}>{groups[match.group(1)]})"
        position = match.end()
    pattern += re.escape(template[position:])
    return re.compile(r"\A" + pattern + r"\Z", re.DOTALL)


_LABEL_GROUP = r"startup|round [1-9][0-9]*"
_PLAN_SCHEDULING_RE = _template_regex(
    PLAN_SCHEDULING_DIAGNOSTIC_TEMPLATE, {"label": _LABEL_GROUP, "message": r".*"}
)
_PR_SCHEDULING_RE = _template_regex(
    PR_SCHEDULING_DIAGNOSTIC_TEMPLATE, {"label": _LABEL_GROUP, "message": r".*"}
)
_CI_HEADLINE_RES = tuple(
    _template_regex(template, {"pr_number": r"[1-9][0-9]*"})
    for template in (CI_INFRASTRUCTURE_COMMENT_HEADLINE, CI_INFRASTRUCTURE_STOP_HEADLINE)
)
_SETTINGS_LOCK_RE = re.compile(r"settings[ _-]lock", re.I)
_AUTH_TOKEN_RE = re.compile(r"auth|unauthorized|forbidden|invalid api key", re.I)


def _label_round(label: str) -> int | None:
    return int(label.split(" ", 1)[1]) if label.startswith("round ") else None


def classify_infrastructure_failure(text: str, *, category: str | None = None) -> str:
    """Infrastructure category of an agent-unavailable text.

    Reuses the shared agent-failure and transient classifiers: settings-lock
    contention, then auth/billing (non-retryable), quota/rate limits, and
    transient backend outages; anything else keeps the envelope category.
    """
    if _SETTINGS_LOCK_RE.search(text):
        return "settings-lock"
    non_retryable = NON_RETRYABLE_AGENT_OUTPUT_RE.search(text)
    if non_retryable is not None:
        token = non_retryable.group(0)
        if _AUTH_TOKEN_RE.match(token):
            return "auth-expiry"
        if token.lower() in {"credit", "billing"}:
            return "billing"
        return "environment"
    if _QUOTA_RATE_LIMIT_RE.search(text):
        return "quota"
    if is_transient_agent_output(text):
        return "backend-outage"
    return f"agent-unavailable-{category or 'unknown'}"


def _signature_line(text: str) -> str | None:
    for line in reversed(text.strip().splitlines()):
        if line.startswith("-- "):
            return line[3:].strip()
    return None


def recognize_diagnostic(body: str) -> RecognizedDiagnostic | None:
    """Recognize a plain stop or infrastructure diagnostic, else ``None``.

    Bodies carrying round metadata are records, never plain diagnostics.
    """
    if not isinstance(body, str) or ROUND_RESUME_MARKER_RE.search(body):
        return None
    for regex, kind in ((_PLAN_SCHEDULING_RE, KIND_PLANNING_STOP), (_PR_SCHEDULING_RE, KIND_PR_STOP)):
        match = regex.match(body)
        if match is None:
            continue
        message = match.group("message")
        round_number = _label_round(match.group("label"))
        if message.startswith(PLAN_STEP_BACK_HUMAN_DECISION_PREFIX):
            return RecognizedDiagnostic(KIND_HUMAN_DECISION_STOP, "plan-step-back", round_number)
        if message.startswith(PLAN_PRIMARY_STALL_PREFIX):
            return RecognizedDiagnostic(KIND_HUMAN_DECISION_STOP, "plan-primary-stall", round_number)
        return RecognizedDiagnostic(kind, "scheduling-stop", round_number)
    first_line = body.split("\n", 1)[0]
    if any(regex.match(first_line) for regex in _CI_HEADLINE_RES):
        return RecognizedDiagnostic(KIND_INFRASTRUCTURE, "ci-infrastructure-stall")
    try:
        unavailable = parse_agent_unavailable(body)
    except AgentLoopError:
        return None
    if unavailable is None:
        return None
    return RecognizedDiagnostic(
        KIND_INFRASTRUCTURE,
        classify_infrastructure_failure(unavailable.summary, category=unavailable.category),
        seat_signature=_signature_line(body),
    )


def _signature_names_seat(signature: str | None, seat: str) -> bool:
    if not signature:
        return False
    if signature == seat or signature.startswith((f"{seat} (", f"{seat}:")):
        return True
    for backend in BACKENDS.values():
        if backend.display_name == seat and (
            signature == backend.signature or signature.startswith(f"{backend.signature}:")
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Run-start snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TextVersion:
    effective_at: datetime.datetime
    title: str
    body: str


def _same_text(left: str, right: str) -> bool:
    return left.replace("\r\n", "\n") == right.replace("\r\n", "\n")


def reconstruct_text_versions(history: IssueOutcomeHistory) -> tuple[TextVersion, ...] | None:
    """Every title/body version in effect, oldest first; ``None`` when unprovable.

    The reconstruction must end at the current title and body; any gap
    (deleted revision, truncated history, inconsistent rename chain) yields
    ``None`` rather than a guessed version.
    """
    if history.body_revisions is None or history.title_renames is None:
        return None
    body = history.body_revisions[0][1] if history.body_revisions else history.body
    title = history.title_renames[0][1] if history.title_renames else history.title
    created = parse_comment_timestamp(history.created_at)
    events: list[tuple[datetime.datetime, int, str, str, str | None]] = []
    for order, (edited_at, text) in enumerate(history.body_revisions[1:]):
        events.append((parse_comment_timestamp(edited_at), order, "body", text, None))
    for order, (renamed_at, previous, current) in enumerate(history.title_renames):
        events.append((parse_comment_timestamp(renamed_at), order, "title", current, previous))
    events.sort(key=lambda item: (item[0], item[2], item[1]))
    versions = [TextVersion(created, title, body)]
    for at, _order, field, text, previous in events:
        if field == "title":
            if previous is not None and previous != title:
                return None
            title = text
        else:
            body = text
        versions.append(TextVersion(at, title, body))
    if title != history.title or not _same_text(body, history.body):
        return None
    return tuple(versions)


def _iso(value: datetime.datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def derive_snapshot(
    history: IssueOutcomeHistory,
    *,
    plan_issue_digest: str | None,
    run_started_at: datetime.datetime | None,
) -> dict[str, object]:
    """The title/body the run received, with an honest exactness status."""

    def snapshot(status: str, version: TextVersion | None = None, *, normalized: bool = False):
        if version is None:
            return {
                "status": status, "title": None, "body": None, "as_of": None,
                "plan_issue_digest": plan_issue_digest,
            }
        title, body = version.title, version.body
        if normalized:
            title, body = normalize_plan_issue_text(title), normalize_plan_issue_text(body)
        return {
            "status": status, "title": title, "body": body,
            "as_of": _iso(version.effective_at), "plan_issue_digest": plan_issue_digest,
        }

    versions = reconstruct_text_versions(history)
    if versions is None:
        return snapshot(SNAPSHOT_UNAVAILABLE)
    if plan_issue_digest is not None:
        first_seen: dict[tuple[str, str], TextVersion] = {}
        for version in versions:
            first_seen.setdefault((version.title, version.body), version)
        matches = [
            version
            for (title, body), version in first_seen.items()
            if plan_issue_text_digest_for(title, body) == plan_issue_digest
        ]
        if not matches:
            return snapshot(SNAPSHOT_UNAVAILABLE)
        earliest = min(matches, key=lambda version: version.effective_at)
        if len(matches) == 1:
            return snapshot(SNAPSHOT_DIGEST_VERIFIED, earliest)
        return snapshot(SNAPSHOT_DIGEST_NORMALIZED, earliest, normalized=True)
    if len(versions) == 1:
        return snapshot(SNAPSHOT_NEVER_EDITED, versions[0])
    if run_started_at is None:
        return snapshot(SNAPSHOT_UNAVAILABLE)
    in_effect = [version for version in versions if version.effective_at <= run_started_at]
    if not in_effect:
        return snapshot(SNAPSHOT_UNAVAILABLE)
    return snapshot(SNAPSHOT_TIME_INFERRED, in_effect[-1])


# ---------------------------------------------------------------------------
# Row derivation
# ---------------------------------------------------------------------------


def _pair(value: object, verified: bool) -> dict[str, object]:
    if not verified:
        return {"value": None, "evidence": EVIDENCE_UNAVAILABLE}
    return {"value": value, "evidence": EVIDENCE_VERIFIED}


@dataclass(frozen=True)
class _Thread:
    """One authenticated surface: its records per flow and its diagnostics."""

    surface: str
    comments: tuple[AuthenticatedComment, ...]
    records: Mapping[str, tuple[PostedRoundRecord, ...]]
    diagnostics: tuple[tuple[int, RecognizedDiagnostic], ...]
    decode_errors: tuple[str, ...]

    def created(self, index: int) -> datetime.datetime:
        return parse_comment_timestamp(self.comments[index].created_at)


def _read_thread(view: AuthenticatedCommentView, flows: Sequence[str]) -> _Thread:
    comments = tuple(view.authored)
    records: dict[str, tuple[PostedRoundRecord, ...]] = {}
    errors: list[str] = []
    for flow in flows:
        try:
            records[flow] = _extract_round_metadata_records(comments, flow=flow)
        except AgentLoopError as exc:
            records[flow] = ()
            errors.append(f"{view.surface} {flow} records undecodable: {exc}")
    diagnostics = tuple(
        (index, diagnostic)
        for index, comment in enumerate(comments)
        if (diagnostic := recognize_diagnostic(comment.body)) is not None
    )
    return _Thread(view.surface, comments, records, diagnostics, tuple(errors))


@dataclass(frozen=True)
class _FlowLedger:
    flow: str
    thread: _Thread
    rounds: tuple[int, ...]
    infra_rounds: frozenset[int]
    missing: tuple[dict[str, object], ...]


def _seats(records: Sequence[PostedRoundRecord]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        record.metadata.agent for record in records if record.metadata.role == "reviewer"
    ))


def _expected_seats(
    records: Sequence[PostedRoundRecord], round_number: int, seats: tuple[str, ...]
) -> tuple[str, ...]:
    """Scheduler-selected reviewers for the round, else every seen seat."""
    selected = tuple(dict.fromkeys(
        reviewer
        for record in records
        if record.metadata.round_number == round_number
        and record.metadata.scheduler_metadata_status == "valid"
        for reviewer in record.metadata.scheduler_selected_reviewers
    ))
    return selected or seats


def _present_seats(
    records: Sequence[PostedRoundRecord], round_number: int, *, before: int | None = None
) -> set[str]:
    return {
        record.metadata.agent
        for record in records
        if record.metadata.round_number == round_number
        and record.metadata.role == "reviewer"
        and (before is None or record.index < before)
    }


def _attach_infrastructure(
    thread: _Thread, flow: str
) -> dict[int, list[tuple[int, RecognizedDiagnostic]]]:
    """Attach each infrastructure diagnostic to the round it interrupted.

    A diagnostic belongs to the newest round started before it only while
    that round still lacks an expected reviewer record; one posted after a
    fully reviewed round (for example during implementation) ends no round.
    """
    records = thread.records.get(flow, ())
    seats = _seats(records)
    attached: dict[int, list[tuple[int, RecognizedDiagnostic]]] = {}
    for index, diagnostic in thread.diagnostics:
        if diagnostic.kind != KIND_INFRASTRUCTURE:
            continue
        prior = [
            record for record in records
            if record.index < index and record.metadata.role in _REVIEW_ROLES
        ]
        if not prior:
            continue
        round_number = max(record.metadata.round_number for record in prior)
        expected = set(_expected_seats(records, round_number, seats))
        if expected and expected <= _present_seats(records, round_number, before=index):
            continue
        attached.setdefault(round_number, []).append((index, diagnostic))
    return attached


def _flow_ledger(thread: _Thread, flow: str, *, bounded: bool) -> _FlowLedger:
    records = thread.records.get(flow, ())
    review_records = [record for record in records if record.metadata.role in _REVIEW_ROLES]
    rounds = tuple(sorted({record.metadata.round_number for record in review_records}))
    attached = _attach_infrastructure(thread, flow)
    infra_rounds = frozenset(
        round_number
        for round_number, diagnostics in attached.items()
        if not any(
            record.metadata.round_number == round_number
            and record.index > max(index for index, _ in diagnostics)
            for record in review_records
        )
    )
    stop_rounds = {
        diagnostic.round_number
        for _index, diagnostic in thread.diagnostics
        if diagnostic.kind in STOP_KINDS and diagnostic.round_number is not None
    }
    seats = _seats(records)
    missing: list[dict[str, object]] = []
    for round_number in rounds:
        present = _present_seats(records, round_number)
        if not present and round_number in stop_rounds:
            continue
        if round_number == rounds[-1] and not bounded:
            continue  # the newest round may still be in progress
        for seat in _expected_seats(records, round_number, seats):
            if seat in present:
                continue
            cause = next(
                (
                    diagnostic.reason
                    for _index, diagnostic in attached.get(round_number, ())
                    if _signature_names_seat(diagnostic.seat_signature, seat)
                ),
                CAUSE_UNKNOWN,
            )
            missing.append({
                "surface": thread.surface, "flow": flow, "round": round_number,
                "reviewer": seat, "cause": cause,
            })
    return _FlowLedger(flow, thread, rounds, infra_rounds, tuple(missing))


def _latest_round_approved(records: Sequence[PostedRoundRecord]) -> bool:
    reviews = [
        record for record in records
        if record.metadata.role == "reviewer" and record.metadata.phase == "authoritative"
    ]
    if not reviews:
        return False
    latest = max(record.metadata.round_number for record in records if record.metadata.role in _REVIEW_ROLES)
    current = [record for record in reviews if record.metadata.round_number == latest]
    if not current:
        return False
    newest: dict[str, str | None] = {}
    for record in current:
        newest[record.metadata.agent] = record.metadata.state
    selected = {
        reviewer
        for record in records
        if record.metadata.round_number == latest
        and record.metadata.scheduler_metadata_status == "valid"
        for reviewer in record.metadata.scheduler_selected_reviewers
    }
    if selected - set(newest):
        return False
    return all(state == "approved" for state in newest.values())


def _era_known(records: Iterable[PostedRoundRecord], fields: Sequence[str]) -> bool:
    return any(
        record.metadata.step_back_status != "absent"
        or any(getattr(record.metadata, name) is not None for name in fields)
        for record in records
    )


def _step_back_keys(records: Iterable[PostedRoundRecord], excluded_rounds: frozenset[int]) -> set:
    keys: set[tuple[object, ...]] = set()
    for record in records:
        for entry in record.metadata.step_back_entries:
            if entry.get("trigger_round") in excluded_rounds:
                continue
            keys.add((entry.get("phase"), entry.get("reviewer"), entry.get("trigger_round"), entry.get("trigger_head")))
    return keys


def _final_plan_size(records: Sequence[PostedRoundRecord]) -> int | None:
    for record in reversed(records):
        if record.metadata.role != "coder":
            continue
        plan = record.metadata.canonical_plan or record.metadata.raw_structured_coder_response
        return len(plan.strip()) if plan else None
    return None


def derive_history_row(
    *,
    repo: str,
    history: IssueOutcomeHistory,
    issue_view: AuthenticatedCommentView,
    pr_views: Mapping[int, AuthenticatedCommentView],
    tool_commits: Iterable[str] | None = None,
) -> dict[str, object]:
    """Derive one schema-version-1 outcome row from authenticated views only.

    ``pr_views`` maps each linked PR number to its authenticated view; only
    ``view.authored`` (the authenticated actor's comments) contributes, so a
    byte-identical record posted by anyone else is ignored.
    """
    issue = _read_thread(issue_view, ("discuss", "plan"))
    prs = {
        pr.number: _read_thread(pr_views[pr.number], ("pr",))
        for pr in history.linked_prs
        if pr.number in pr_views
    }
    threads = (issue, *prs.values())
    decode_errors = [error for thread in threads for error in thread.decode_errors]
    missing_pr_views = [pr.number for pr in history.linked_prs if pr.number not in pr_views]

    plan_records = issue.records.get("plan", ())
    discuss_records = issue.records.get("discuss", ())
    pr_records = {number: thread.records.get("pr", ()) for number, thread in prs.items()}

    # Route actually taken, ordered by its first authenticated record.
    starts: list[tuple[datetime.datetime, str]] = []
    if discuss_records:
        starts.append((issue.created(discuss_records[0].index), ROUTE_DISCUSS))
    if plan_records:
        starts.append((issue.created(plan_records[0].index), ROUTE_PLAN_FIRST))
    elif any(pr_records.values()):
        first_pr = min(
            prs[number].created(records[0].index)
            for number, records in pr_records.items() if records
        )
        starts.append((first_pr, ROUTE_DIRECT))
    route_taken = [route for _at, route in sorted(starts)]

    # Terminal evidence.
    issue_closed = history.closed_at is not None or history.state.upper() == "CLOSED"
    # Workflow events are round records and recognized diagnostics only: the
    # operator often comments from the same account, and such a comment does
    # not end a workflow.
    events: list[tuple[datetime.datetime, int, str, int]] = [
        (thread.created(index), thread.comments[index].comment_id, thread.surface, index)
        for thread in threads
        for index in sorted(
            {record.index for records in thread.records.values() for record in records}
            | {index for index, _diagnostic in thread.diagnostics}
        )
    ]
    stop_indices = {
        (thread.surface, index)
        for thread in threads
        for index, diagnostic in thread.diagnostics
        if diagnostic.kind in STOP_KINDS
    }
    latest_is_stop = bool(events) and (max(events)[2], max(events)[3]) in stop_indices
    linked_open = [pr.number for pr in history.linked_prs if (pr.state or "").upper() == "OPEN"]
    plan_approved = bool(plan_records) and _latest_round_approved(plan_records)
    discuss_final = any(record.metadata.is_final for record in discuss_records)
    merged_any = any(pr.merged_at is not None or pr.state.upper() == "MERGED" for pr in history.linked_prs)
    prs_settled = bool(history.linked_prs) and not linked_open
    outcome_complete = (issue_closed or latest_is_stop) and not linked_open and not missing_pr_views
    bounded = bool(
        outcome_complete or issue_closed or latest_is_stop or plan_approved or prs_settled
        or (discuss_final and not plan_records and not history.linked_prs)
    ) and not decode_errors
    plan_terminal = not decode_errors and (plan_approved or latest_is_stop or issue_closed)
    pr_terminal = not decode_errors and not missing_pr_views and (
        prs_settled or (not history.linked_prs and bounded)
    )

    # Rounds, infrastructure attachment and missing reviewer records.
    ledgers = [_flow_ledger(issue, "plan", bounded=bounded)] + [
        _flow_ledger(thread, "pr", bounded=bounded) for thread in prs.values()
    ]
    plan_ledger = ledgers[0]
    pr_ledgers = ledgers[1:]
    infrastructure_failures = [
        {
            "surface": thread.surface,
            "comment_id": thread.comments[index].comment_id,
            "created_at": _iso(thread.created(index)),
            "category": diagnostic.reason,
            "seat": diagnostic.seat_signature,
        }
        for thread in threads
        for index, diagnostic in thread.diagnostics
        if diagnostic.kind == KIND_INFRASTRUCTURE
    ]
    infrastructure_rounds = [
        {"surface": ledger.thread.surface, "flow": ledger.flow, "round": round_number}
        for ledger in ledgers
        for round_number in sorted(ledger.infra_rounds)
    ]
    missing_reviewer_records = [entry for ledger in ledgers for entry in ledger.missing]
    coverage = (
        COVERAGE_PARTIAL
        if decode_errors
        or missing_pr_views
        or any(entry["cause"] == CAUSE_UNKNOWN for entry in missing_reviewer_records)
        else COVERAGE_COMPLETE
    )

    plan_round_count = len([r for r in plan_ledger.rounds if r not in plan_ledger.infra_rounds])
    pr_round_count = sum(
        len([r for r in ledger.rounds if r not in ledger.infra_rounds]) for ledger in pr_ledgers
    )

    # Step-backs: era-gated so legacy histories stay unavailable, not zero.
    all_pr_records = [record for records in pr_records.values() for record in records]
    step_back_era = (not plan_records or _era_known(plan_records, _PLAN_STEP_BACK_ERA_FIELDS)) and (
        not all_pr_records or _era_known(all_pr_records, _PR_STEP_BACK_ERA_FIELDS)
    )
    step_backs = len(_step_back_keys(plan_records, plan_ledger.infra_rounds)) + sum(
        len(_step_back_keys(ledger.thread.records.get("pr", ()), ledger.infra_rounds))
        for ledger in pr_ledgers
    )
    human_stops = sum(
        1
        for thread in threads
        for _index, diagnostic in thread.diagnostics
        if diagnostic.kind == KIND_HUMAN_DECISION_STOP
    )
    plan_size = _final_plan_size(plan_records) if plan_records else 0

    # Outcome availability: every contributing source, including edits.
    available: list[datetime.datetime] = []
    for thread in threads:
        for comment in thread.comments:
            available.append(parse_comment_timestamp(comment.created_at))
            if comment.updated_at is not None:
                available.append(parse_comment_timestamp(comment.updated_at))
    for pr in history.linked_prs:
        for stamp in (pr.merged_at, pr.closed_at):
            if stamp is not None:
                available.append(parse_comment_timestamp(stamp))
    if history.closed_at is not None:
        available.append(parse_comment_timestamp(history.closed_at))

    record_times = [
        thread.created(record.index)
        for thread in threads
        for records in thread.records.values()
        for record in records
    ]
    run_started_at = min(record_times) if record_times else None
    first_digest = next(
        (record.metadata.plan_issue_digest for record in plan_records if record.metadata.plan_issue_digest),
        None,
    )
    commits = sorted(set(tool_commits or ()))

    terminal_states = [
        name
        for name, present in (
            ("discuss-final", discuss_final),
            ("plan-approved", plan_approved),
            ("pr-merged", merged_any),
            ("prs-settled", prs_settled),
            ("issue-closed", issue_closed),
            ("recognized-stop", latest_is_stop),
        )
        if present
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "issue_number": history.number,
        "issue_state": history.state.lower(),
        "issue_created_at": _iso(parse_comment_timestamp(history.created_at)),
        "issue_closed_at": _iso(parse_comment_timestamp(history.closed_at)) if history.closed_at else None,
        "route_taken": route_taken,
        "route": " then ".join(route_taken) if route_taken else EVIDENCE_UNAVAILABLE,
        "plan_rounds": _pair(plan_round_count, plan_terminal if plan_records else bounded),
        "step_backs": _pair(step_backs, bounded and step_back_era),
        "human_decision_stops": _pair(human_stops, bounded),
        "final_plan_size": _pair(
            plan_size, plan_size is not None and (plan_terminal if plan_records else bounded)
        ),
        "pr_rounds": _pair(pr_round_count, pr_terminal),
        "merged": _pair(merged_any, merged_any or pr_terminal),
        "tool_commit": _pair(commits, bool(commits)),
        "linked_prs": [
            {
                "number": pr.number, "state": pr.state.lower(),
                "merged_at": pr.merged_at, "closed_at": pr.closed_at,
            }
            for pr in history.linked_prs
        ],
        "terminal_states": terminal_states,
        "infrastructure_failures": infrastructure_failures,
        "infrastructure_rounds": infrastructure_rounds,
        "missing_reviewer_records": missing_reviewer_records,
        "infrastructure_coverage": coverage,
        "decode_errors": decode_errors + [
            f"linked PR #{number} comments were not read" for number in missing_pr_views
        ],
        "ignored_foreign_records": sum(
            len(view.ignored_foreign) for view in (issue_view, *pr_views.values())
        ),
        "outcome_complete": outcome_complete,
        "outcome_available_at": _iso(max(available)) if available else None,
        "run_started_at": _iso(run_started_at),
        "snapshot": derive_snapshot(
            history, plan_issue_digest=first_digest, run_started_at=run_started_at
        ),
    }


def has_agent_loop_history(view: AuthenticatedCommentView) -> bool:
    """Whether the actor posted any round record or recognized diagnostic."""
    return any(
        ROUND_RESUME_MARKER_RE.search(comment.body) or recognize_diagnostic(comment.body)
        for comment in view.authored
    )


# ---------------------------------------------------------------------------
# Local tool-commit join (usage summaries + run attribution)
# ---------------------------------------------------------------------------


def load_tool_commits(
    summary_dir: Path, attribution_lines: Iterable[str], *, repo: str
) -> dict[tuple[str, int], set[str]]:
    """Map ``("issue"|"pr", number)`` to the tool commits of attributed runs.

    A usage summary carries its run ID and tool commit but no issue; only an
    explicit ``run-start`` attribution record for the same repository joins
    the two, so an unattributed summary contributes nothing (never a guess).
    """
    commits_by_run: dict[str, str] = {}
    for path in sorted(summary_dir.glob("*-usage-summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        run_id = payload.get("run_id")
        provenance = payload.get("tool_provenance")
        commit = provenance.get("commit") if isinstance(provenance, dict) else None
        if isinstance(run_id, str) and isinstance(commit, str) and commit:
            commits_by_run[run_id] = commit
    joined: dict[tuple[str, int], set[str]] = {}
    for line in attribution_lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("record") != "run-start":
            continue
        if record.get("repo") != repo or record.get("run_id") not in commits_by_run:
            continue
        commit = commits_by_run[record["run_id"]]
        for kind, key in (("issue", "issue_number"), ("pr", "pr_number")):
            number = record.get(key)
            if isinstance(number, int) and not isinstance(number, bool):
                joined.setdefault((kind, number), set()).add(commit)
    return joined


def row_tool_commits(
    joined: Mapping[tuple[str, int], set[str]], history: IssueOutcomeHistory
) -> set[str]:
    commits = set(joined.get(("issue", history.number), set()))
    for pr in history.linked_prs:
        commits |= joined.get(("pr", pr.number), set())
    return commits


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_PAIR_COLUMNS = (
    "plan_rounds", "step_backs", "human_decision_stops", "final_plan_size",
    "pr_rounds", "merged", "tool_commit",
)


def render_json(rows: Sequence[Mapping[str, object]], *, repo: str) -> str:
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "repo": repo, "rows": list(rows)},
        indent=2, sort_keys=True, ensure_ascii=False,
    ) + "\n"


def render_csv(rows: Sequence[Mapping[str, object]]) -> str:
    """Flatten rows: pairs become ``<name>``/``<name>_evidence``; lists are JSON."""
    header = [
        "schema_version", "repo", "issue_number", "issue_state", "route",
        *(name for column in _PAIR_COLUMNS for name in (column, f"{column}_evidence")),
        "outcome_complete", "outcome_available_at", "run_started_at",
        "infrastructure_coverage", "infrastructure_failures", "missing_reviewer_records",
        "snapshot_status", "snapshot_as_of", "snapshot_title", "snapshot_body",
    ]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        snapshot = row["snapshot"]
        assert isinstance(snapshot, Mapping)
        values: list[object] = [row["schema_version"], row["repo"], row["issue_number"], row["issue_state"], row["route"]]
        for column in _PAIR_COLUMNS:
            pair = row[column]
            assert isinstance(pair, Mapping)
            value = pair["value"]
            values.append(json.dumps(value) if isinstance(value, (list, bool)) else ("" if value is None else value))
            values.append(pair["evidence"])
        values.extend([
            json.dumps(row["outcome_complete"]), row["outcome_available_at"] or "",
            row["run_started_at"] or "", row["infrastructure_coverage"],
            json.dumps(row["infrastructure_failures"], sort_keys=True),
            json.dumps(row["missing_reviewer_records"], sort_keys=True),
            snapshot["status"], snapshot["as_of"] or "",
            snapshot["title"] if snapshot["title"] is not None else "",
            snapshot["body"] if snapshot["body"] is not None else "",
        ])
        writer.writerow(values)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# GitHub driver (reads only)
# ---------------------------------------------------------------------------


def read_only_history_config(repo: str, *, gh_cmd: str = "gh") -> "AgentLoopConfig":
    """A configuration that can drive only repo-explicit GitHub reads.

    Every agent directory is the GitHub API scratch directory, so nothing here
    can address a checkout, and no agent command is ever resolved or run.
    """
    from .config import AgentLoopConfig
    from .workdirs import github_api_cwd

    scratch = github_api_cwd()
    return AgentLoopConfig(
        repo=repo, claude_dir=scratch, codex_dir=scratch, gemini_dir=scratch,
        antigravity_dir=scratch, coder="claude", reviewer=("codex",), base=None,
        max_rounds=1, auto_merge=False, dry_run=False, allow_shared_dir=True,
        claude_cmd="claude", codex_cmd="codex", gemini_cmd="gemini", gh_cmd=gh_cmd,
        claude_args=(), codex_args=(), gemini_args=(), test_command=None,
        pre_review_tests=False, ci_timeout_seconds=0, ci_poll_interval_seconds=1,
        quiet=True, log_dir=scratch, progress_interval_seconds=30, agent_max_retries=0,
        agent_retry_backoff_seconds=(), agent_memory=False, refresh_agent_memory=False,
        agent_memory_dir=scratch, refresh_test_profile=False,
        execution_strategy_contract_required=False,
    )


def export_history(
    runner: object,
    *,
    config: object,
    issue_numbers: Sequence[int] = (),
    since: datetime.datetime | None = None,
    tool_commit_index: Mapping[tuple[str, int], set[str]] | None = None,
) -> list[dict[str, object]]:
    """Read authenticated history from GitHub and derive rows; never writes.

    Without ``issue_numbers`` every repository issue created at or after
    ``since`` is considered, and issues without any agent-loop history are
    skipped.  Explicit issues always yield a row.
    """
    from .github import (
        list_repository_issues,
        read_authenticated_protocol_comments,
        read_issue_outcome_history,
    )

    explicit = bool(issue_numbers)
    if explicit:
        candidates = list(dict.fromkeys(issue_numbers))
    else:
        candidates = [
            issue.number
            for issue in list_repository_issues(runner, config=config, since=since)  # type: ignore[arg-type]
            if since is None or parse_comment_timestamp(issue.created_at) >= since
        ]
    rows: list[dict[str, object]] = []
    for number in candidates:
        issue_view = read_authenticated_protocol_comments(
            runner, config=config, surface_kind="issue", number=number  # type: ignore[arg-type]
        )
        if not explicit and not has_agent_loop_history(issue_view):
            history = read_issue_outcome_history(runner, config=config, issue_number=number)  # type: ignore[arg-type]
            if not history.linked_prs:
                continue
        else:
            history = read_issue_outcome_history(runner, config=config, issue_number=number)  # type: ignore[arg-type]
        pr_views = {
            pr.number: read_authenticated_protocol_comments(
                runner, config=config, surface_kind="pr", number=pr.number  # type: ignore[arg-type]
            )
            for pr in history.linked_prs
        }
        if not explicit and not has_agent_loop_history(issue_view) and not any(
            has_agent_loop_history(view) for view in pr_views.values()
        ):
            continue
        rows.append(
            derive_history_row(
                repo=config.repo,  # type: ignore[attr-defined]
                history=history,
                issue_view=issue_view,
                pr_views=pr_views,
                tool_commits=row_tool_commits(tool_commit_index or {}, history),
            )
        )
    return rows


__all__ = [
    "LinkedPullRequest",
    "RecognizedDiagnostic",
    "SCHEMA_VERSION",
    "classify_infrastructure_failure",
    "derive_history_row",
    "derive_snapshot",
    "export_history",
    "load_tool_commits",
    "recognize_diagnostic",
    "reconstruct_text_versions",
    "render_csv",
    "render_json",
]
