"""Plan verification context and the issue-to-PR board-change audit (#1373).

An approved plan is verified against the board that approved it, never the PR
invocation's board.  ``derive_plan_verification_context`` reads that board
from the plan's owning issue: its persisted plan contract and signed plan
amendment lineage (C0 -> Cn), its seat binding, and its planning policy and
primary.  Every plan-approval consumer (managed-CI fresh authorization,
ordinary resume without an issue-side handoff, the handoff seam, and strict
managed-CI qualification) uses it.

The PR board is a per-invocation choice.  When it differs from the plan's
effective board, the orchestrator posts one plain-text board-change audit
record on the PR, identified by a digest so a rerun posts nothing new.  The
record carries no protocol marker and no signature grammar: it is neither a
signed human requirement nor a signed reviewer-board amendment.

Recorded plan board changes (#1377): once the plan history carries a plan
board root anchor or an orchestrator-authored plan board-change link, the
context is resolved through the plan board lineage (links and signed plan
amendments in one ordered chain).  The approving board is the board of the
latest lineage element that took effect, and approvals that the lineage
retired (a removed seat, an added or rebound seat before its effective round,
a pre-epoch approval after a primary change) never count.

:func:`classify_plan_approval` decides read-only, from history alone, whether
the current canonical plan is provably completely approved.  An approved plan
is never reopened by a later invocation's board or planning flags.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Callable, Sequence

from .errors import AgentLoopError
from .reviewer_floor import BoardSeat, SignedBoardAuthorization, distinct_providers

BOARD_CHANGE_AUDIT_HEADING = "Reviewer board change recorded at issue-to-PR handoff."
BOARD_CHANGE_KIND = "reviewer-board-change"
BOARD_CHANGE_REASON = "operator-reconfigured"
BOARD_CHANGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlanVerificationContext:
    """The approving plan board, derived from the plan's owning issue only."""

    issue_number: int | None
    original_reviewers: tuple[str, ...]
    effective_reviewers: tuple[str, ...]
    original_seats: tuple[BoardSeat, ...]
    effective_seats: tuple[BoardSeat, ...]
    policy: str
    primary_reviewer: str | None
    seat_binding: dict | None
    signed_amendment_count: int
    # Comment positions of approvals bound to a superseded seat model; they
    # never count towards the approving board (#1373 review item-2).
    stale_approval_indices: frozenset[int] = frozenset()
    # How the approving board was derived: ``lineage`` (plan board root or
    # link, #1377), ``contract``, ``binding`` or ``posters`` (plain legacy
    # history, which proves membership but never completion by itself).
    derivation: str = "contract"

    def verification_comments(self, comments: Sequence[object]) -> tuple[object, ...]:
        """``comments`` with superseded-model approvals neutralized in place.

        Positions are kept so sidecar hydration and ordering are unchanged.
        """
        from types import SimpleNamespace

        return tuple(
            SimpleNamespace(body="") if index in self.stale_approval_indices else comment
            for index, comment in enumerate(comments)
        )

    @property
    def signed_authorization(self) -> SignedBoardAuthorization | None:
        """The signed board, only when a validated signed link exists.

        An unamended plan (C0 equal to Cn) never supplies a floor exception.
        Each measure is excepted only where the chain lowered it.
        """
        if self.signed_amendment_count < 1:
            return None
        return SignedBoardAuthorization(
            seats=self.effective_seats,
            lowered_reviewers=len(self.effective_seats) < len(self.original_seats),
            lowered_providers=(
                distinct_providers(self.effective_seats)
                < distinct_providers(self.original_seats)
            ),
        )

    def reviewer_agents(self) -> tuple[object, ...]:
        from .agents.registry import agent_display_name
        from .reviewer_seats import ReviewerSeat, SeatAgent

        agents: list[object] = []
        for seat in self.effective_seats:
            if seat.seat_id == agent_display_name(seat.backend):
                agents.append(seat.backend)
            else:
                agents.append(SeatAgent(
                    ReviewerSeat(
                        seat.seat_id, seat.backend,
                        tuple(seat.model_chain or ()), seat.effort,
                    ),
                    Path("."),
                ))
        return tuple(agents)

    def config_for(self, config: object) -> object:
        """``config`` re-pointed at the approving plan board for verification.

        Only the plan-approval readers consume the result; the PR invocation
        board stays the PR review board.
        """
        from .agents.registry import agent_display_name
        from .config import (
            DEFAULT_PLAN_PRIMARY_STALL_ROUNDS,
            DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS,
            DEFAULT_PLAN_STEP_BACK_ROUNDS,
        )
        from .reviewer_seats import SeatAgent

        agents = self.reviewer_agents()
        primary = None
        if self.primary_reviewer is not None:
            primary = next(
                (agent for agent in agents if agent_display_name(agent) == self.primary_reviewer),
                None,
            )
            if primary is None:
                raise AgentLoopError(
                    "Approved plan primary reviewer is not on the approving plan board."
                )
        staged = self.policy == "primary-then-panel"
        return replace(
            config,
            reviewer=agents,
            reviewer_seats=tuple(agent for agent in agents if isinstance(agent, SeatAgent)),
            active_reviewer_seat_id=None,
            pr_seat_binding_override=self.seat_binding,
            plan_review_policy=self.policy,
            primary_plan_reviewer=primary,
            plan_review_force_full=False,
            plan_reset_stall_streak=False,
            plan_primary_stall_rounds=(
                config.plan_primary_stall_rounds if staged else DEFAULT_PLAN_PRIMARY_STALL_ROUNDS
            ),
            plan_step_back_rounds=(
                config.plan_step_back_rounds if staged else DEFAULT_PLAN_STEP_BACK_ROUNDS
            ),
            plan_step_back_escalation_rounds=(
                config.plan_step_back_escalation_rounds
                if staged else DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS
            ),
            pr_review_policy="all-reviewers",
            primary_reviewer=None,
            pr_review_force_full=False,
        )


def _plain_backends() -> dict[str, str]:
    from .agents.registry import agent_display_name
    from .reviewer_seats import _BACKENDS

    return {agent_display_name(backend): backend for backend in sorted(_BACKENDS)}


def _board_seats(
    names: Sequence[str], binding_entries: dict[str, dict],
) -> tuple[BoardSeat, ...]:
    plain = _plain_backends()
    seats: list[BoardSeat] = []
    for name in names:
        entry = binding_entries.get(name)
        if entry is not None:
            seats.append(BoardSeat(
                seat_id=name,
                backend=entry["backend"],
                model_chain=tuple(entry["model_chain"]),
                effort=entry.get("effort"),
            ))
        elif name in plain and not binding_entries:
            # A plain plan board records no models; only membership and the
            # implied backend are verifiable.
            seats.append(BoardSeat(seat_id=name, backend=plain[name]))
        else:
            raise AgentLoopError(
                f"Approved plan reviewer {name!r} has no verifiable backend binding."
            )
    return tuple(seats)


def derive_plan_verification_context(
    comments: Sequence[object],
    *,
    issue_number: int | None,
    trusted_actor: tuple[str, int] | None = None,
) -> PlanVerificationContext | None:
    """Derive the approving plan board from the plan's owning issue history.

    Returns ``None`` when the issue carries no plan round records (or no plan
    reviewer at all), in which case no plan board can be verified.  Malformed
    binding or amendment lineage fails closed.  History carrying a plan board
    root anchor or a plan board-change link is resolved through the plan
    board lineage; its links are admitted only from ``trusted_actor``.
    """
    from .board_amendment import collect_reviewer_board_amendments, resolve_contract_lineage
    from .board_change import PLAN_SURFACE, comments_bear_board_change_links
    from .reviewer_seats import (
        _persisted_plan_contract,
        validate_backend_outage_amendments_for_binding,
        validated_binding_entries,
    )
    from .round_state import _extract_round_metadata_records

    records = _extract_round_metadata_records(comments, flow="plan")
    if not records:
        return None
    if comments_bear_board_change_links(comments, surface=PLAN_SURFACE) or any(
        record.metadata.plan_board_root is not None for record in records
    ):
        return _lineage_verification_context(
            comments, records, issue_number=issue_number, trusted_actor=trusted_actor,
        )
    binding, binding_entries, stale = _validated_plan_binding_history(records)
    contract = next(
        (contract for record in records
         if (contract := _persisted_plan_contract(record.metadata)) is not None),
        None,
    )
    if contract is None and any(
        record.metadata.scheduler_metadata_status == "invalid" for record in records
    ):
        # A malformed planning checkpoint hides the persisted policy; never
        # downgrade it to all-reviewers verification.
        raise AgentLoopError(
            "Approved plan scheduler history is malformed, so its planning policy "
            "cannot be verified."
        )
    amendment_count = 0
    derivation = "contract"
    if contract is not None:
        amendments = collect_reviewer_board_amendments(
            comments, flow="plan", issue_number=issue_number,
        )
        validate_backend_outage_amendments_for_binding(amendments, binding)
        lineage = resolve_contract_lineage(
            records, amendments, contract, accept_base_configured=True,
            contract_from_metadata=_persisted_plan_contract,
            drift_error=lambda persisted, detail: AgentLoopError(
                f"Approved plan board history is inconsistent with its signed amendments: {detail}"
            ),
        )
        effective = lineage.contracts[-1]
        original_names = tuple(contract.required_reviewers)
        effective_names = tuple(effective.required_reviewers)
        policy = effective.policy
        primary = effective.primary_reviewer
        amendment_count = len(lineage.amendments)
    elif binding is not None:
        original_names = effective_names = tuple(binding_entries)
        policy, primary = "all-reviewers", None
        derivation = "binding"
    else:
        # Plain all-reviewers planning persists no contract or binding: every
        # reviewer that reviewed the plan is on the approving board.
        names: list[str] = []
        for record in records:
            if record.metadata.role == "reviewer" and record.metadata.agent not in names:
                names.append(record.metadata.agent)
        if not names:
            return None
        original_names = effective_names = tuple(names)
        policy, primary = "all-reviewers", None
        derivation = "posters"
    return PlanVerificationContext(
        issue_number=issue_number,
        original_reviewers=original_names,
        effective_reviewers=effective_names,
        original_seats=_board_seats(original_names, binding_entries),
        effective_seats=_board_seats(effective_names, binding_entries),
        policy=policy,
        primary_reviewer=primary,
        seat_binding=binding,
        signed_amendment_count=amendment_count,
        stale_approval_indices=stale,
        derivation=derivation,
    )


def _lineage_verification_context(
    comments: Sequence[object],
    records: Sequence[object],
    *,
    issue_number: int | None,
    trusted_actor: tuple[str, int] | None,
) -> PlanVerificationContext:
    """The approving plan board of history with a plan board lineage (#1377).

    Resolved through the plan surface resolver, never from the invocation's
    planning configuration.  The approving identity is the board of the
    latest lineage element any plan round record took effect under (pending
    elements never governed an approval).  Any lineage defect raises.
    """
    import dataclasses

    from .board_change import (
        PLAN_SURFACE,
        BoardIdentity,
        apply_inits,
        collect_board_change_links,
    )
    from .plan_board_state import resolve_plan_lineage

    if issue_number is None:
        raise AgentLoopError(
            "Approved plan board lineage cannot be verified without its owning issue."
        )
    anchored = next(
        (record.metadata.plan_board_root for record in records
         if record.metadata.plan_board_root is not None),
        None,
    )
    if anchored is not None:
        root = BoardIdentity.from_payload(anchored)
    else:
        links = collect_board_change_links(
            comments, pr_number=issue_number, trusted_actor=trusted_actor, surface=PLAN_SURFACE,
        )
        root = links[0].root
    lineage = resolve_plan_lineage(
        comments,
        issue_number=issue_number,
        configured=root,
        trusted_actor=trusted_actor,
        drift_error=lambda _persisted, detail: AgentLoopError(
            f"Approved plan board history is inconsistent with its lineage: {detail}"
        ),
    )
    # Only elements some plan round record took effect under can have
    # governed an approval.
    taken = [element for element in lineage.elements if element.digest in lineage.carried]
    last = lineage.elements.index(taken[-1]) + 1 if taken else 0
    view = dataclasses.replace(lineage, elements=lineage.elements[:last])
    effective = view.current_identity
    base = apply_inits(view.segment_base, view.inits)
    segment_amendments = 0
    for element in reversed(view.elements):
        if element.kind != "amendment":
            break
        segment_amendments += 1
    fresh = view.fresh_from_rounds
    epoch = view.primary_epoch_round
    effective_names = set(effective.names)
    stale: set[int] = set()
    for record in records:
        metadata = record.metadata
        if epoch is not None and metadata.round_number < epoch and metadata.role != "coder":
            # A primary change restarts the primary phase: pre-epoch reviews,
            # panel openings and approvals never count.
            stale.add(record.index)
            continue
        if metadata.role != "reviewer" or metadata.state != "approved":
            continue
        if metadata.agent not in effective_names or metadata.round_number < fresh.get(
            metadata.agent, 0
        ):
            stale.add(record.index)
    binding = base.seat_binding() if base.named else effective.seat_binding()
    return PlanVerificationContext(
        issue_number=issue_number,
        original_reviewers=base.names,
        effective_reviewers=effective.names,
        original_seats=base.seats,
        effective_seats=effective.seats,
        policy=effective.policy,
        primary_reviewer=effective.primary,
        seat_binding=binding,
        signed_amendment_count=segment_amendments,
        stale_approval_indices=frozenset(stale),
        derivation="lineage",
    )


def _validated_plan_binding_history(
    records: Sequence[object],
) -> tuple[dict | None, dict[str, dict], frozenset[int]]:
    """Validate every plan seat binding against the plan's own history.

    Returns the latest binding, its entries, and the positions of approvals
    that a later model chain or effort reconfiguration superseded: any
    approval recorded before the latest binding transition, including one
    that returns to an earlier model, plus any approval bound differently
    from the latest binding.  Mixed
    bound/unbound rounds, changed backends, a changed recorded board, and a
    reviewer outside its own binding fail closed.  The PR invocation board is
    never consulted.
    """
    from .reviewer_seats import validated_binding_entries

    bound = [record for record in records if record.metadata.seat_binding is not None]
    if not bound:
        return None, {}, frozenset()
    latest = bound[-1].metadata.seat_binding
    latest_entries = validated_binding_entries(latest)
    board = set(latest_entries)
    backends: dict[str, str] = {}
    stale: set[int] = set()
    approvals: list[tuple[int, int]] = []  # (record index, comment index)
    previous_models: dict[str, tuple[object, object]] | None = None
    last_transition = -1
    for position, record in enumerate(records):
        metadata = record.metadata
        recorded = metadata.seat_binding
        if recorded is None:
            if metadata.role in {"reviewer", "coder", "summary"}:
                raise AgentLoopError(
                    "Approved plan history mixes bound and unbound round records "
                    f"({metadata.role}, round {metadata.round_number}); its approving "
                    "board cannot be verified."
                )
            continue
        entries = validated_binding_entries(recorded)
        if set(entries) != board:
            raise AgentLoopError(
                "Approved plan seat binding changed its recorded board; its approving "
                "board cannot be verified."
            )
        for seat_id, entry in entries.items():
            if backends.setdefault(seat_id, entry["backend"]) != entry["backend"]:
                raise AgentLoopError(
                    f"Approved plan seat {seat_id!r} changed backend in plan history."
                )
        models = {
            seat_id: (tuple(entry["model_chain"]), entry.get("effort"))
            for seat_id, entry in entries.items()
        }
        if previous_models is not None and models != previous_models:
            # A model chain or effort reconfiguration: the live plan loop
            # clears approval and panel state here, so every approval recorded
            # before it is superseded, even when a later record restores the
            # earlier model (A -> B -> A).
            last_transition = position
        previous_models = models
        if metadata.role != "reviewer":
            continue
        own = entries.get(metadata.agent)
        if own is None:
            raise AgentLoopError(
                "Approved plan reviewer record is not represented in its seat binding."
            )
        current = latest_entries[metadata.agent]
        if metadata.state == "approved":
            approvals.append((position, record.index))
            if (
                own["model_chain"] != current["model_chain"]
                or own.get("effort") != current.get("effort")
            ):
                stale.add(record.index)
    stale.update(index for position, index in approvals if position < last_transition)
    return latest, latest_entries, frozenset(stale)


def plan_verification_inputs(
    config: object,
    comments: Sequence[object],
    *,
    issue_number: int | None,
    trusted_actor: tuple[str, int] | None = None,
) -> tuple[object, tuple[object, ...]]:
    """The config and comments every plan-approval reader uses.

    The config is re-pointed at the approving plan board; the comments have
    approvals bound to a superseded seat model neutralized.  Without a plan
    candidate there is nothing to verify, and every consumer already fails
    closed on a missing canonical plan.
    """
    from .round_state import _extract_round_metadata_records

    comments = tuple(comments)
    if not any(
        record.metadata.role == "coder"
        for record in _extract_round_metadata_records(comments, flow="plan")
    ):
        return config, comments
    context = derive_plan_verification_context(
        comments, issue_number=issue_number, trusted_actor=trusted_actor,
    )
    if context is None:
        return config, comments
    return context.config_for(config), context.verification_comments(comments)


# --- Board-change audit record ------------------------------------------------


def _seat_payload(seat: BoardSeat) -> dict[str, object]:
    return {
        "id": seat.seat_id,
        "backend": seat.backend,
        "model_chain": list(seat.model_chain) if seat.model_chain is not None else None,
        "effort": seat.effort,
    }


def board_change_payload(
    handoff: object, *, repo: str, pr_number: int, plan_hash: str | None,
) -> dict[str, object]:
    plan = handoff.plan
    return {
        "kind": BOARD_CHANGE_KIND,
        "schema_version": BOARD_CHANGE_SCHEMA_VERSION,
        "repo": repo,
        "pr_number": pr_number,
        "plan_issue": plan.issue_number,
        "plan_hash": plan_hash,
        "old_board": [_seat_payload(seat) for seat in plan.effective_seats],
        "new_board": [_seat_payload(seat) for seat in handoff.pr_seats],
        "effective_round": 1,
        "reason": BOARD_CHANGE_REASON,
    }


def board_change_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _describe_seat(seat: BoardSeat) -> str:
    if seat.model_chain is None:
        return f"{seat.seat_id} ({seat.backend})"
    models = " -> ".join(model or "default model" for model in seat.model_chain)
    effort = f", effort {seat.effort}" if seat.effort else ""
    return f"{seat.seat_id} ({seat.backend}: {models}{effort})"


def render_board_change_audit(handoff: object, payload: dict[str, object], digest: str) -> str:
    plan = handoff.plan
    changes = [
        f"{_describe_seat(old)} -> {_describe_seat(new)}"
        for old, new in handoff.binding_changes
    ]
    lines = [
        BOARD_CHANGE_AUDIT_HEADING,
        "",
        f"- Record digest: `{digest}`",
        f"- Reason: {BOARD_CHANGE_REASON}",
        "- Effective round: 1",
        (
            f"- Approved plan `{payload['plan_hash']}`"
            + (f" from issue #{plan.issue_number}" if plan.issue_number is not None else "")
            + " stays verified against the plan board that approved it."
        ),
        "- Plan board: " + "; ".join(_describe_seat(seat) for seat in plan.effective_seats),
        "- PR board: " + "; ".join(_describe_seat(seat) for seat in handoff.pr_seats),
        f"- Added reviewer(s): {', '.join(handoff.added) or 'none'}",
        f"- Removed reviewer(s): {', '.join(handoff.removed) or 'none'}",
        f"- Binding change(s): {'; '.join(changes) or 'none'}",
        "- Every PR board reviewer must approve the PR's exact head; plan approvals do "
        "not count as PR approvals.",
        "",
        "-- Orchestrator",
    ]
    return "\n".join(lines)


def board_change_audit_already_posted(comments: Sequence[object], digest: str) -> bool:
    def body(comment: object) -> object:
        if isinstance(comment, dict):
            return comment.get("body")
        return getattr(comment, "body", None)

    return any(
        isinstance(text := body(comment), str)
        and BOARD_CHANGE_AUDIT_HEADING in text
        and digest in text
        for comment in comments
    )


def signed_lineage_authorization(
    original_names: Sequence[str],
    *,
    effective_config: object,
    known_configs: Sequence[object] = (),
    binding: dict | None = None,
) -> SignedBoardAuthorization:
    """The signed authorization of a validated, non-empty PR amendment chain.

    The effective seats keep their configured bindings (PR history refuses a
    backend change on a recorded seat).  A provider exception applies only
    when every original seat's backend is known and the chain lowered it.
    """
    from .reviewer_floor import board_seats_from_config
    from .reviewer_seats import validated_binding_entries

    effective_seats = board_seats_from_config(effective_config)
    backends: dict[str, str] = {}
    for known in known_configs:
        for seat in board_seats_from_config(known):
            backends.setdefault(seat.seat_id, seat.backend)
    if binding is not None:
        for seat_id, entry in validated_binding_entries(binding).items():
            backends.setdefault(seat_id, entry["backend"])
    plain = _plain_backends()
    original_backends = [backends.get(name) or plain.get(name) for name in original_names]
    return SignedBoardAuthorization(
        seats=effective_seats,
        lowered_reviewers=len(effective_seats) < len(original_names),
        lowered_providers=(
            None not in original_backends
            and distinct_providers(effective_seats) < len(set(original_backends))
        ),
    )


# --- Plan approval classification (#1377) --------------------------------------

PLAN_APPROVAL_STATES = ("no-plan", "unfinished", "undetermined", "approved")


@dataclass(frozen=True)
class PlanApprovalClassification:
    """Whether the current canonical plan is provably completely approved.

    ``approved`` exempts a run from plan board-change candidacy, planning
    policy drift and the planning candidate floor; it never produces an
    approved outcome by itself.  The live approval boundary still runs, and
    any input it cannot confirm downgrades the run to normal resolution.
    """

    state: str
    reason: str
    context: PlanVerificationContext | None = None
    plan_hash: str | None = None

    @property
    def approved(self) -> bool:
        return self.state == "approved"


def plan_hash_has_issue_handoff(
    comments: Sequence[object], *, issue_number: int, repo: str | None, plan_hash: str,
) -> bool:
    """Whether an issue-side implementation handoff names exactly ``plan_hash``.

    Completion evidence for unanchored plain history, whose posters prove
    membership only (#1377 step 12a).  Malformed records never count.
    """
    from .decomposition import (
        ONE_SHOT_IMPL_HANDOFF_MARKER_RE,
        PHASE_IMPLEMENTATION_MARKER_RE,
        _decode_one_shot_impl_handoff_metadata,
        _decode_phase_implementation_handoff_metadata,
    )
    from .issue_pr_handoff import find_latest_issue_pr_handoff

    if repo is not None:
        try:
            handoff = find_latest_issue_pr_handoff(comments, issue_number=issue_number, repo=repo)
        except AgentLoopError:
            handoff = None
        if handoff is not None and handoff.plan_hash == plan_hash:
            return True
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for pattern, decode in (
            (ONE_SHOT_IMPL_HANDOFF_MARKER_RE, _decode_one_shot_impl_handoff_metadata),
            (PHASE_IMPLEMENTATION_MARKER_RE, _decode_phase_implementation_handoff_metadata),
        ):
            for match in pattern.finditer(body):
                try:
                    metadata = decode(match.group("payload"))
                except (AgentLoopError, ValueError, KeyError, TypeError):
                    continue
                if (
                    getattr(metadata, "parent_issue", None) == issue_number
                    and getattr(metadata, "plan_hash", None) == plan_hash
                ):
                    return True
    return False


def plan_open_obligations(plan_round: object, *, board: Sequence[str]) -> tuple[str, ...]:
    """Active blocking or same-plan items left after the resumed round reconciles.

    The ledger view reassigns removed reviewers' items to the remaining
    ``board``, so a removed reviewer's unresolved finding stays open.
    """
    from .board_amendment import apply_board_amendment_to_ledger
    from .round_state import _aggregate_record_dispositions
    from .unresolved_items import _apply_unresolved_item_dispositions

    prior = tuple(getattr(plan_round, "prior_items", ()))
    new = tuple(getattr(plan_round, "current_round_new_items", ()))
    if not new:
        new = tuple(
            item
            for record in getattr(plan_round, "completed_reviews", ())
            for item in record.metadata.new_items
        )
    owners = {
        owner
        for item in (*prior, *new)
        for owner in (tuple(item.resolution_owners or ()) or (item.reviewer,))
    }
    removed = tuple(sorted(name for name in owners if name and name not in set(board)))
    view, _ = apply_board_amendment_to_ledger(
        prior,
        removed_reviewers=removed,
        remaining_reviewers=tuple(board),
        primary_reviewer=None,
    )
    items, _future = _apply_unresolved_item_dispositions(
        view,
        _aggregate_record_dispositions(tuple(getattr(plan_round, "completed_reviews", ()))),
        same_status="same-plan",
        retain_future=False,
    )
    open_ids = [item.item_id for item in items if item.status in {"blocking", "same-plan"}]
    prior_ids = {item.item_id for item in prior}
    open_ids += [
        item.item_id for item in new
        if item.status in {"blocking", "same-plan"} and item.item_id not in prior_ids
    ]
    return tuple(dict.fromkeys(open_ids))


def _approval_requirement_binding(
    plan_round: object,
    *,
    board: Sequence[str],
    plan_text: str,
    surfaced_ids: Sequence[str],
) -> str:
    """``bound``, ``stale`` or ``unproven`` for an all-reviewers approval set (step 12c).

    A recorded exact-plan candidate key binds only when it equals the key
    recomputed against the current requirement set.  A keyless legacy
    approval binds only through the plan's own dispositions naming exactly
    the current set, plus the acknowledgement in every approval text when
    that set is non-empty.
    """
    from .panel_evidence import _plan_candidate_key_for, _plan_key_from_payload
    from .protocol import human_requirements_resolved
    from .response_validation import _current_plan_has_complete_human_requirement_dispositions
    from .plan_assembly import decode_assembled_plan_sidecar
    from .round_state import _plan_subject

    coder_metadata = getattr(plan_round, "coder_metadata", None)
    sidecar = None
    if coder_metadata is not None and coder_metadata.assembled_plan_sidecar is not None:
        try:
            sidecar = decode_assembled_plan_sidecar(coder_metadata.assembled_plan_sidecar)
        except (AgentLoopError, ValueError, KeyError, TypeError):
            return "unproven"
    current_key = _plan_candidate_key_for(
        plan_subject=_plan_subject(plan_text),
        sidecar=sidecar,
        surfaced_requirement_ids=surfaced_ids,
    )
    surfaced = {str(item) for item in surfaced_ids}
    approvals = [
        record for record in getattr(plan_round, "completed_reviews", ())
        if record.metadata.agent in set(board) and record.metadata.state == "approved"
    ]
    legacy = []
    for record in approvals:
        key = _plan_key_from_payload(record.metadata.plan_candidate_key)
        if key is None:
            legacy.append(record)
            continue
        if not key.matches(current_key) or (
            surfaced and set(record.metadata.surfaced_reviewer_requirement_ids) != surfaced
        ):
            return "stale"
    if not legacy:
        return "bound"
    if not _current_plan_has_complete_human_requirement_dispositions(
        getattr(plan_round, "coder_output", None), surfaced_requirement_ids=tuple(surfaced_ids),
    ):
        return "unproven"
    if surfaced and not all(human_requirements_resolved(record.body or "") for record in legacy):
        return "unproven"
    return "bound"


def classify_plan_approval(
    comments: Sequence[object],
    *,
    config: object,
    issue_number: int,
    human_requirements: Sequence[object],
    trusted_actor: tuple[str, int] | None = None,
    inherited_check: Callable[[str], str | None] | None = None,
    inherited_derivable: bool = True,
    supersession_pending: bool = False,
) -> PlanApprovalClassification:
    """Read-only plan approval classification from history alone (#1377 step 12).

    Shared by the issue-entry floor preflight and plan startup.  Every input
    comes from durable history: the approving plan identity (never the
    invocation's reviewers or planning flags), the reconstructed latest round
    and its ledger, the current signed requirement set and each counted
    approval's requirement binding, the inherited-matrix admissibility
    result, a pending signed supersession, and the plan-growth gate.  Any
    underivable input means not approved; a malformed lineage raises.
    """
    from .config import reviewers
    from .panel_evidence import _require_complete_canonical_plan_approval
    from .plan_review_scheduling import plan_policy_capabilities
    from .response_validation import _surfaced_reviewer_requirement_ids
    from .round_state import _extract_round_metadata_records, _resume_plan_round
    from .decomposition import approved_plan_hash

    from .board_change import PLAN_SURFACE, comments_bear_board_change_links
    from .config import phased_delivery_guard_active
    from .execution_policy import _current_execution_recommendation

    comments = tuple(comments)
    records = _extract_round_metadata_records(comments, flow="plan")
    if not records:
        return PlanApprovalClassification("no-plan", "no plan round records")
    lineage_bearing = comments_bear_board_change_links(comments, surface=PLAN_SURFACE) or any(
        record.metadata.plan_board_root is not None for record in records
    )
    try:
        context = derive_plan_verification_context(
            comments, issue_number=issue_number, trusted_actor=trusted_actor,
        )
    except AgentLoopError as exc:
        if lineage_bearing:
            # A malformed plan board lineage fails closed.
            raise
        # Degraded legacy history keeps today's handling in the live loop.
        return PlanApprovalClassification("undetermined", f"legacy plan history: {exc}")
    if context is None:
        return PlanApprovalClassification("unfinished", "no plan reviewer has reviewed the plan")
    plan_config = context.config_for(config)
    plan_comments = context.verification_comments(comments)
    try:
        resumed = _resume_plan_round(plan_comments, configured_reviewers=reviewers(plan_config))
    except (AgentLoopError, ValueError, KeyError, TypeError) as exc:
        return PlanApprovalClassification(
            "undetermined", f"the latest plan round cannot be reconstructed: {exc}", context,
        )
    if resumed is None:
        return PlanApprovalClassification("unfinished", "no reconstructible plan round", context)
    plan_text, plan_round = resumed
    plan_hash = approved_plan_hash(plan_text)

    def result(state: str, reason: str) -> PlanApprovalClassification:
        return PlanApprovalClassification(state, reason, context, plan_hash)

    if phased_delivery_guard_active(getattr(config, "plan_execution_mode", "")):
        # The execution-mode guard (#1268): a staged plan under a guard-active
        # mode is narrowed or stopped by the live loop, never approved here.
        try:
            recommendation = _current_execution_recommendation(plan_text, comments)
        except AgentLoopError:
            return result("undetermined", "the execution recommendation cannot be read")
        if recommendation is not None and recommendation.strategy == "staged":
            return result("unfinished", "a staged plan under a phased-delivery execution mode")

    if supersession_pending:
        return result("unfinished", "a validated signed plan supersession is pending")
    if not inherited_derivable:
        return result("undetermined", "the inherited-matrix admissibility is not derivable")
    if inherited_check is not None and inherited_check(plan_text) is not None:
        return result("unfinished", "the plan fails its inherited parent matrix rows")
    if context.derivation == "posters" and not plan_hash_has_issue_handoff(
        comments, issue_number=issue_number, repo=getattr(config, "repo", None),
        plan_hash=plan_hash,
    ):
        # Historical posters prove membership, never completion (step 12a).
        return result(
            "undetermined",
            "unanchored plain plan history has no issue-side handoff bound to the plan hash",
        )
    try:
        _require_complete_canonical_plan_approval(
            plan_comments,
            config=plan_config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=human_requirements,
            error_message="incomplete",
        )
    except AgentLoopError:
        return result("unfinished", "not every approving reviewer approved the exact plan")
    except (ValueError, KeyError, TypeError) as exc:
        return result("undetermined", f"the approval state cannot be reconstructed: {exc}")
    board = context.effective_reviewers
    open_items = plan_open_obligations(plan_round, board=board)
    if open_items:
        return result("unfinished", "open plan obligation(s) " + ", ".join(open_items))
    if not plan_policy_capabilities(context.policy).scheduler_enabled:
        binding = _approval_requirement_binding(
            plan_round,
            board=board,
            plan_text=plan_text,
            surfaced_ids=_surfaced_reviewer_requirement_ids(
                human_requirements, requirement_scope="planning requirements",
            ),
        )
        if binding == "stale":
            return result("unfinished", "an approval is bound to a stale requirement set")
        if binding != "bound":
            return result("undetermined", "an approval's requirement binding is unproven")
    return result("approved", "every approving reviewer approved the exact plan")


def enforce_issue_board_floor(
    config: object,
    comments: Sequence[object],
    *,
    issue_number: int,
    runner: object | None = None,
    human_requirements: Sequence[object] = (),
    planning: bool = False,
    inherited_derivable: bool = True,
) -> None:
    """Issue-mode floor preflight, before any planning or implementation turn.

    Two branches (#1377 step 9).  When the issue's current plan is provably
    approved, or there is no plan to resume, the effective board is the
    invocation's board after the stage 1 handoff board rule against the
    approving plan context, so a signed plan amendment excuses only the
    measures its chain lowered; the invocation's planning flags are never
    compared.  Otherwise a plan-first run resolves its plan board read-only,
    including any board-change candidate, and refuses a below-floor plan
    board before any agent invocation or comment post.
    """
    from .reviewer_floor import board_floor_enabled, enforce_configured_board_floor
    from .reviewer_seats import resolve_plan_handoff_board

    if not board_floor_enabled(config):
        return
    trusted_actor = None
    if runner is not None:
        from .plan_board_state import admitted_issue_comments

        comments, trusted_actor = admitted_issue_comments(
            runner, config=config, issue_number=issue_number, comments=comments,
        )
    if planning:
        classification = classify_plan_approval(
            comments,
            config=config,
            issue_number=issue_number,
            human_requirements=human_requirements,
            trusted_actor=trusted_actor,
            inherited_derivable=inherited_derivable,
        )
        if classification.state not in {"approved", "no-plan"}:
            from .plan_board_state import (
                plan_drift_error,
                plan_floor_refusal_hint,
                plan_resume_round,
                resolve_plan_board_state,
            )

            resolve_plan_board_state(
                runner,
                config=config,
                issue_number=issue_number,
                comments=comments,
                admitted=(comments, trusted_actor),
                drift_error=plan_drift_error(issue_number),
                floor_context=f"Issue #{issue_number} plan reviewer board",
                floor_refusal_hint=plan_floor_refusal_hint(
                    issue_number, lambda: plan_resume_round(comments, config),
                ),
            )
            return
    plan = derive_plan_verification_context(
        comments, issue_number=issue_number, trusted_actor=trusted_actor,
    )
    authorizations: tuple[SignedBoardAuthorization, ...] = ()
    effective = config
    if plan is not None:
        handoff = resolve_plan_handoff_board(config, plan)
        effective = handoff.config
        authorizations = handoff.signed_authorizations
    enforce_configured_board_floor(
        effective,
        authorizations=authorizations,
        context=f"Issue #{issue_number} reviewer board",
    )
