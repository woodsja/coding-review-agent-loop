"""Read-only PR board resolution shared by the entry preflight and startup (#1376).

:func:`resolve_pr_board_state` resolves the unified PR board lineage from PR
comments, applies the #1133 original-flags rebind for a signed segment,
derives the board-change candidate when the configured board differs from the
effective board, and checks the reviewer floor against the lineage that would
exist after posting that candidate.  It never posts and never invokes an
agent, so every caller can refuse before any write.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace

from .agents.registry import agent_display_name
from .board_amendment import collect_reviewer_board_amendments, require_amendment_activation
from .board_change import (
    BoardChangeCandidate,
    BoardIdentity,
    BoardLineage,
    LineageElement,
    binding_compatible,
    build_board_change_candidate,
    comments_bear_board_change_links,
    comments_with_candidate,
    configured_differs,
    identity_from_config,
    lineage_floor_authorizations,
    policy_differs,
    resolve_board_lineage,
)
from .config import reviewers
from .errors import AgentLoopError
from .reviewer_floor import board_floor_enabled, check_reviewer_floor
from .round_state import _extract_round_metadata_records

# Placeholder actor used only to validate an unposted candidate when the PR
# has no link records yet, so no authenticated actor is needed.
_CANDIDATE_ACTOR = ("orchestrator-candidate", 0)


@dataclass(frozen=True)
class PrBoardState:
    """The resolved PR board for one invocation."""

    lineage: BoardLineage
    config: object
    configured: BoardIdentity
    candidate: BoardChangeCandidate | None
    comments: tuple[object, ...]
    trusted_actor: tuple[str, int] | None
    rebound: bool

    @property
    def activation(self) -> LineageElement | None:
        """The pending link this run activates (a recovered run's last, or the candidate)."""
        return self.lineage.pending_links[-1] if self.lineage.pending_links else None

    @property
    def predecessor_board(self) -> BoardIdentity | None:
        """The board that governed the anchor-round records of a pending activation."""
        if not self.lineage.pending_links:
            return None
        first = self.lineage.pending_links[0].link
        assert first is not None
        return first.old_board


def admitted_pr_comments(
    runner: object,
    *,
    config: object,
    pr_number: int,
    comments: Sequence[object],
) -> tuple[tuple[object, ...], tuple[str, int] | None]:
    """Comments with REST author identities, plus the actor, when links exist.

    Board-change links are admitted only from the authenticated orchestrator
    actor, so a PR carrying any link record needs both; an incomplete read
    fails closed rather than reading as an empty history.
    """
    from .github import merge_pr_comment_transport_identity, resolve_authenticated_github_actor
    from .pr_loop_support import _cached_trusted_actor

    comments = tuple(comments)
    if not comments_bear_board_change_links(comments):
        return comments, None
    if getattr(config, "dry_run", False):
        actor = _cached_trusted_actor(runner)
        if actor is None:
            raise AgentLoopError(
                f"PR #{pr_number} carries reviewer-board change records, but the authenticated "
                "GitHub actor is unavailable in dry-run mode, so they cannot be admitted."
            )
        return comments, actor
    merged = merge_pr_comment_transport_identity(
        runner, config=config, pr_number=pr_number, comments=comments  # type: ignore[arg-type]
    )
    return tuple(merged), resolve_authenticated_github_actor(runner, config=config)  # type: ignore[arg-type]


def _rebind_signed_segment(config: object, lineage: BoardLineage, configured: BoardIdentity) -> object | None:
    """The #1133 rebind: original flags of a signed segment select its amended board.

    Applies when the latest element is a signed amendment and the configured
    board is the segment's base board (the root, or the latest link's new
    board) with compatible bindings.  Returns ``None`` when it does not apply.
    """
    from .reviewer_seats import reviewer_seat_binding

    latest = lineage.latest
    if latest is None or latest.kind != "amendment":
        return None
    base = lineage.segment_base
    from .board_change import apply_inits

    if not binding_compatible(apply_inits(base, lineage.inits), configured):
        return None
    names = set(lineage.current_board)
    return dataclasses_replace(
        config,
        reviewer=tuple(r for r in reviewers(config) if agent_display_name(r) in names),
        pr_seat_binding_override=reviewer_seat_binding(config),
    )


def resolve_pr_board_state(
    runner: object,
    *,
    config: object,
    pr_number: int,
    comments: Sequence[object],
    drift_error: Callable[[object, str], AgentLoopError],
    handoff_authorizations: Sequence[object] = (),
    floor_context: str | None = None,
    resume_round: Callable[[object, BoardLineage], int] | None = None,
    floor_refusal_hint: Callable[[object, object], str] | None = None,
    diagnostics: list[str] | None = None,
) -> PrBoardState:
    """Resolve the PR board read-only and refuse a below-floor board (D4).

    ``resume_round(effective_config, lineage)`` returns the round a resume
    re-enters; it is consulted only to validate a pending signed amendment.
    """
    from .pr_loop_support import _scheduler_contract_from_metadata
    from .panel_evidence import _board_amendment_template

    comments, trusted_actor = admitted_pr_comments(
        runner, config=config, pr_number=pr_number, comments=comments
    )
    records = _extract_round_metadata_records(comments, flow="pr")
    amendments = collect_reviewer_board_amendments(
        comments, flow="pr", pr_number=pr_number,
        ignored_sink=diagnostics if diagnostics is not None else [],
    )
    configured = identity_from_config(config)

    def resolve(source: Sequence[object], actor: tuple[str, int] | None) -> BoardLineage:
        return resolve_board_lineage(
            records,
            source,
            amendments=amendments,
            configured=configured,
            pr_number=pr_number,
            trusted_actor=actor,
            contract_from_metadata=_scheduler_contract_from_metadata,
            drift_error=drift_error,
        )

    lineage = resolve(comments, trusted_actor)
    if policy_differs(configured, lineage.root):
        # Changing the review policy or broad-path rules stays contract drift.
        raise drift_error(lineage.current_contract, "the configured contract does not match it")
    effective_config = config
    rebound = False
    rebind = _rebind_signed_segment(config, lineage, configured)
    if rebind is not None:
        effective_config, rebound = rebind, True
    effective_identity = identity_from_config(effective_config)
    candidate = None
    previous_contract = lineage.current_contract
    if configured_differs(effective_identity, lineage.current_identity):
        candidate = build_board_change_candidate(
            lineage, configured=effective_identity, records=records, pr_number=pr_number
        )
        actor = trusted_actor or _CANDIDATE_ACTOR
        lineage = resolve(
            comments_with_candidate(comments, candidate, actor_id=actor[1]), actor
        )
    elif lineage.pending_amendments and not lineage.pending_links and resume_round is not None:
        start_round = resume_round(effective_config, lineage)
        require_amendment_activation(
            lineage,  # type: ignore[arg-type]
            start_round_number=start_round,
            template=lambda amendment, round_number: _board_amendment_template(
                flow="pr",
                issue_number=None,
                pr_number=pr_number,
                persisted=lineage.contract_before(amendment),
                removed=amendment.removed_reviewers,
                restored=amendment.restored_reviewers,
                start_round_number=round_number,
            ),
        )
    try:
        check_lineage_floor(
            config,
            lineage,
            context=floor_context or f"PR #{pr_number} reviewer board",
            handoff_authorizations=handoff_authorizations,
        )
    except AgentLoopError as exc:
        if candidate is None or floor_refusal_hint is None:
            raise
        # A refused board change posts nothing; a removal can still be
        # authorized by a signed amendment, whose template is printed here.
        raise AgentLoopError(
            str(exc) + floor_refusal_hint(previous_contract, effective_identity.contract())
        ) from exc
    return PrBoardState(
        lineage=lineage,
        config=effective_config,
        configured=effective_identity,
        candidate=candidate,
        comments=comments,
        trusted_actor=trusted_actor,
        rebound=rebound,
    )


def check_lineage_floor(
    config: object, lineage: BoardLineage, *, context: str,
    handoff_authorizations: Sequence[object] = (),
) -> None:
    """The per-round and qualification floor check (pure, no I/O)."""
    if not board_floor_enabled(config):
        return
    check_reviewer_floor(
        lineage.current_identity.seats,
        min_reviewers=getattr(config, "min_reviewers", None),
        min_distinct_providers=getattr(config, "min_distinct_providers", None),
        authorizations=(*handoff_authorizations, *lineage_floor_authorizations(lineage)),
        context=context,
    )


def qualification_board_check(
    runner: object,
    *,
    config: object,
    pr_number: int,
    expected: BoardIdentity,
    handoff_authorizations: Sequence[object] = (),
) -> Callable[[Sequence[object]], None]:
    """The qualification gate's board check over the same fresh PR comment fetch.

    Re-resolves the unified lineage for every policy, requires the run's
    board to equal the effective board, and rechecks the floor.  Any failure
    permits no qualification or merge.
    """
    from .logging import log
    from .pr_loop_support import _pr_contract_drift_error

    def check(comments: Sequence[object]) -> None:
        diagnostics: list[str] = []
        try:
            state = resolve_pr_board_state(
                runner,
                config=config,
                pr_number=pr_number,
                comments=comments,
                drift_error=lambda persisted, detail: _pr_contract_drift_error(
                    persisted,
                    detail,
                    pr_number=pr_number,
                    configured=None,
                    start_round_number=lambda: None,
                    during="qualification",
                ),
                handoff_authorizations=handoff_authorizations,
                floor_context=f"PR #{pr_number} qualification reviewer board",
                diagnostics=diagnostics,
            )
            if state.candidate is not None or state.lineage.pending_links:
                raise AgentLoopError(
                    "PR reviewer board changed during qualification: the effective board "
                    f"is {state.lineage.describe()}, but this run reviewed "
                    + "; ".join(seat.seat_id for seat in expected.seats)
                )
            if state.configured.names != expected.names:
                raise AgentLoopError(
                    "PR reviewer board changed during qualification."
                )
        except AgentLoopError as exc:
            if "no qualification or merge is permitted" in str(exc):
                raise
            raise AgentLoopError(f"{exc} No qualification or merge is permitted.") from exc
        finally:
            for diagnostic in diagnostics:
                log(config, f"PR #{pr_number}: {diagnostic}")

    return check
