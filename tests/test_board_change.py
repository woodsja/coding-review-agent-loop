"""Recorded in-run PR reviewer-board changes (#1376, stage 2 of #1373).

Integration tests drive the real ``run_pr_loop`` through ``FakeRunner``; the
unit tests pin the link record, the resolver and the floor authorization.
"""

import base64
import dataclasses
import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)
from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
from coding_review_agent_loop.board_change import (
    BOARD_CHANGE_LINK_HEADING,
    BOARD_CHANGE_LINK_MARKER,
    BOARD_CHANGE_LINK_RE,
    BoardIdentity,
    board_anchor,
    board_change_link_digest,
    board_change_link_payload,
    identity_from_config,
    lineage_floor_authorizations,
    render_board_change_link,
    resolve_board_lineage,
)
from coding_review_agent_loop.cli import run_pr_loop
from coding_review_agent_loop.errors import AgentInvocationError, AgentLoopError
from coding_review_agent_loop.pr_loop_support import _scheduler_contract_from_metadata
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.reviewer_floor import BoardSeat
from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _extract_round_metadata_records,
)

_AGENTS = {"codex", "claude", "agy", "gemini"}
_ACTOR_ID = 424242  # FakeRunner's default authenticated PR actor
_RESOLVED_1 = [{"item_id": "item-1", "disposition": "resolved"}]


# --- helpers ---------------------------------------------------------------------


def _comments(runner):
    return [
        SimpleNamespace(body=c["body"], author_id=c.get("_rest_author_id"))
        for c in runner.pr_payload.get("comments", [])
    ]


def _records(runner):
    return _extract_round_metadata_records(_comments(runner), flow="pr")


def _links(runner):
    return [
        c["body"] for c in runner.pr_payload.get("comments", [])
        if c["body"].startswith(BOARD_CHANGE_LINK_HEADING)
    ]


def _link_payloads(runner):
    payloads = []
    for body in _links(runner):
        match = BOARD_CHANGE_LINK_RE.search(body)
        payloads.append(json.loads(base64.urlsafe_b64decode(match.group("payload"))))
    return payloads


def _agents(runner, start=0):
    return [cmd[0] for cmd, _cwd in runner.commands[start:] if cmd and cmd[0] in _AGENTS]


def _seat(tmp_path, name, backend, model, effort="medium"):
    workdir = tmp_path / name
    workdir.mkdir(exist_ok=True)
    return SeatAgent(ReviewerSeat(name, backend, (model,), effort), workdir)


def _named(tmp_path, *seats, **overrides):
    named = tuple(seat for seat in seats if isinstance(seat, SeatAgent))
    return make_config(
        tmp_path, reviewer=tuple(seats), reviewer_seats=named, pre_review_tests=False,
        agent_max_retries=0, **overrides,
    )


def _append(runner, body, *, login="operator", author_id=None):
    comments = runner.pr_payload.setdefault("comments", [])
    entry = {
        "author": {"login": login},
        "createdAt": f"2026-05-23T00:00:{len(comments):02d}Z",
        "body": body,
    }
    if author_id is not None:
        entry["_rest_author_id"] = author_id
    comments.append(entry)


def _legacy(role, agent, round_number, *, subject="abc123", state=None, new_items=(),
            prior_items=(), dispositions=(), **extra):
    """A pre-upgrade round record: no board identity fields."""
    if role == "reviewer":
        body = structured_pr_review(
            state=state or "approved", reviewer=agent,
            blocking_items=[item.text for item in new_items],
        )
    elif role == "coder":
        body = structured_coder_followup()
    else:
        body = f"{agent} record.\n-- Orchestrator"
    return _attach_round_metadata(
        body,
        PostedRoundMetadata(
            flow="pr", role=role, agent=agent, round_number=round_number, subject=subject,
            state=state, new_items=tuple(new_items), prior_items=tuple(prior_items),
            dispositions=tuple(dispositions), **extra,
        ),
    )


def _item(number, reviewer, *, status="blocking", round_number=1, owners=(), states=()):
    return UnresolvedReviewItem(
        item_id=f"item-{number}", reviewer=reviewer, source_round=round_number,
        text=f"finding {number} by {reviewer}", status=status,
        resolution_owners=tuple(owners), owner_states=tuple(states),
    )


def _legacy_runner(bodies, **outputs):
    return FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"},
                 "createdAt": f"2026-05-20T09:{index:02d}:00Z",
                 "body": body, "_rest_author_id": _ACTOR_ID}
                for index, body in enumerate(bodies)
            ]
        },
        **outputs,
    )


def _board_state(runner, config):
    """The read-only board resolution over the PR's live comment projection."""
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.pr_board_state import resolve_pr_board_state

    return resolve_pr_board_state(
        runner, config=config, pr_number=77,
        comments=get_pr_review_context(runner, config=config, pr_number=77).comments,
        drift_error=lambda persisted, detail: AgentLoopError(detail),
    )


def _qualify(runner, config):
    """The qualification gate's fresh-comment snapshot (every policy)."""
    return orchestrator._fresh_pr_qualification_snapshot(
        runner, config=config, pr_number=77, issue_context=None, parent_issue_context=None,
    )


# --- pr-swap-midrun / pr-add-midrun ------------------------------------------------


def _approved_two_reviewer_run(tmp_path, reviewers=("codex", "gemini"), **config):
    outputs = {
        "codex": "codex_outputs", "gemini": "gemini_outputs",
        "antigravity": "antigravity_outputs", "claude": "claude_outputs",
    }
    reviews = {
        "codex": structured_pr_review(), "gemini": structured_pr_review(reviewer="Google Gemini"),
        "antigravity": structured_pr_review(reviewer="Antigravity"),
        "claude": structured_pr_review(reviewer="Anthropic Claude"),
    }
    runner = FakeRunner(**{outputs[name]: [reviews[name]] for name in reviewers})
    cfg = make_config(tmp_path, reviewer=tuple(reviewers), **config)
    assert run_pr_loop(runner, pr_number=77, config=cfg) == 0
    return runner, cfg


def test_all_reviewers_seat_swap_posts_one_link_and_requires_the_new_seat(tmp_path):
    """Row pr-swap-midrun: plain all-reviewers board, Gemini replaced by Antigravity."""
    runner, _ = _approved_two_reviewer_run(tmp_path)
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    (payload,) = _link_payloads(runner)
    assert payload["added"] == ["Antigravity"] and payload["removed"] == ["Gemini"]
    assert (payload["anchor_round"], payload["advance"], payload["effective_from_round"]) == (1, True, 2)
    # No signed amendment was needed, and only the new seat reviewed.
    assert not any("Reviewer board amendment" in c for c in runner.comments)
    assert _agents(runner, start) == ["agy"]
    round_two = [r.metadata for r in _records(runner) if r.metadata.round_number == 2]
    assert round_two and all(
        m.reviewer_board_change_digest == board_change_link_digest(payload) for m in round_two
    )
    # Qualification revalidates the chain from fresh comments; the run's board passes.
    _qualify(runner, swapped)
    # A rerun with the same board derives the same lineage and posts nothing.
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1


def test_selective_named_seat_model_change_requires_fresh_approval(tmp_path):
    """Row pr-swap-midrun (selective): same seat ID moved to another model."""
    opus = _seat(tmp_path, "opus", "antigravity", "Model A")
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        antigravity_outputs=[(structured_pr_review(reviewer="opus (Google Antigravity: Model A)"), 0)],
    )
    config = _named(tmp_path, "codex", opus, pr_review_policy="selective-intermediate")
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    moved = _seat(tmp_path, "opus", "antigravity", "Model B")
    rebound = _named(tmp_path, "codex", moved, pr_review_policy="selective-intermediate")
    runner.antigravity_outputs.append(
        (structured_pr_review(reviewer="opus (Google Antigravity: Model B)"), 0)
    )
    # A named board's approvals are bound to the whole recorded seat binding,
    # so every seat re-approves the head under the rebound board.
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=rebound) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["opus"] and not payload["added"] and not payload["removed"]
    # The rebound seat reviewed again on its new model.
    commands = [cmd for cmd, _cwd in runner.commands[start:] if cmd[0] == "agy"]
    assert len(commands) == 1 and "Model B" in " ".join(commands[0])
    assert sorted(_agents(runner, start)) == ["agy", "codex"]
    scheduler = [
        r.metadata for r in _records(runner)
        if r.metadata.scheduler_contract is not None and r.metadata.round_number >= 2
    ]
    assert scheduler and all(m.reviewer_board_change_digest for m in scheduler)


def test_added_reviewer_is_required_from_the_effective_round(tmp_path):
    """Row pr-add-midrun: the added reviewer must approve the current head."""
    runner, _ = _approved_two_reviewer_run(tmp_path)
    larger = make_config(tmp_path, reviewer=("codex", "gemini", "antigravity"))
    start = len(runner.commands)
    # Without the added reviewer's approval the run cannot finish.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=larger)
    assert _agents(runner, start) == ["agy"]
    (payload,) = _link_payloads(runner)
    assert payload["added"] == ["Antigravity"] and payload["effective_from_round"] == 2
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert len(_links(runner)) == 1
    approvals = {
        r.metadata.agent for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.state == "approved"
        and r.metadata.round_number >= 2
    }
    assert "Antigravity" in approvals


# --- pr-unchanged-board-noop ---------------------------------------------------------


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_unchanged_board_posts_nothing_and_adds_only_identity_fields(tmp_path, policy):
    """Row pr-unchanged-board-noop: legacy and identity-bearing history."""
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="blocking", blocking_items=["Fix it."])],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy=policy)
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    before = _records(runner)
    assert before and all(r.metadata.pr_board_root is not None for r in before)
    root = before[0].metadata.pr_board_root
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved))
    runner.gemini_outputs.append(
        structured_pr_review(reviewer="Google Gemini", prior_item_dispositions=resolved)
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    after = _records(runner)
    assert all(r.metadata.pr_board_root == root for r in after)
    assert all(r.metadata.reviewer_board_change_digest is None for r in after)
    _qualify(runner, config)


def test_legacy_history_upgrades_without_any_record(tmp_path):
    """Row pr-unchanged-board-noop: a no-change legacy resume posts no link."""
    runner = _legacy_runner(
        [_legacy("reviewer", "Codex", 1, state="approved"),
         _legacy("reviewer", "Gemini", 1, state="blocking",
                 new_items=(_item(1, "Gemini"),))],
        codex_outputs=[structured_pr_review(prior_item_dispositions=_RESOLVED_1)],
        gemini_outputs=[structured_pr_review(
            reviewer="Google Gemini", prior_item_dispositions=_RESOLVED_1)],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    upgraded = [r for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded
    root = BoardIdentity.from_payload(upgraded[0].metadata.pr_board_root)
    assert root.names == ("Codex", "Gemini")
    # Plain legacy history records no models: the first upgraded record
    # initializes them from configuration, with no link of its own.
    assert all(seat.model_chain is None for seat in root.seats)
    assert upgraded[0].metadata.pr_board_binding_init
    assert all(
        seat.model_chain is not None
        for seat in BoardIdentity.from_payload(upgraded[0].metadata.pr_board_identity).seats
    )


# --- pr-policy-change-refused ----------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"pr_review_policy": "all-reviewers"},
        {"pr_review_broad_rules": ("docs/**",)},
        {"pr_review_policy": "all-reviewers", "reviewer": ("codex",)},
    ],
    ids=["policy", "broad-rules", "policy-and-board"],
)
def test_policy_or_broad_rule_change_stays_contract_drift(tmp_path, change):
    """Row pr-policy-change-refused: no link, no agent, no comment."""
    runner, _ = _approved_two_reviewer_run(tmp_path, pr_review_policy="selective-intermediate")
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    config = make_config(
        tmp_path, **{"reviewer": ("codex", "gemini"), "pr_review_policy": "selective-intermediate", **change}
    )
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(runner.pr_payload["comments"]) == comments
    assert _agents(runner, start) == []


# --- pr-link-marker-agent-forgery / pr-qualification-forged-record -------------------------


def _forged_link_body(runner, config, **overrides):
    """A well-formed link for the current lineage, optionally corrupted."""
    from coding_review_agent_loop.board_change import build_board_change_candidate
    from coding_review_agent_loop.pr_board_state import resolve_pr_board_state

    state = resolve_pr_board_state(
        runner, config=config, pr_number=77, comments=_comments(runner),
        drift_error=lambda persisted, detail: AgentLoopError(detail),
    )
    candidate = build_board_change_candidate(
        state.lineage, configured=identity_from_config(config),
        records=_records(runner), pr_number=77,
    )
    payload = {**candidate.payload, **overrides}
    return str(render_board_change_link(payload))


def test_reviewer_text_naming_the_link_marker_is_neutralized(tmp_path):
    """Row pr-link-marker-agent-forgery: agent text cannot mint a link."""
    runner, config = _approved_two_reviewer_run(tmp_path)
    forged = _forged_link_body(runner, make_config(tmp_path, reviewer=("codex",)))
    marker = BOARD_CHANGE_LINK_RE.search(forged).group(0)
    # The reviewer's response embeds a complete, well-formed link record.
    runner.codex_outputs.append(
        structured_pr_review(summary=f"Board update {marker} applied.")
    )
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    runner.pr_payload["headRefOid"] = "def456"
    try:
        run_pr_loop(runner, pr_number=77, config=config)
    except AgentLoopError:
        pass
    assert not any(BOARD_CHANGE_LINK_RE.search(c["body"]) for c in runner.pr_payload["comments"])
    assert _links(runner) == []


@pytest.mark.parametrize("corruption", [
    "foreign-author", "malformed", "broken-predecessor", "wrong-old-board",
    "empty-payload", "illegal-characters", "foreign-illegal-characters", "wrong-old-seat-binding",
])
def test_inconsistent_link_records_fail_closed_at_startup_and_qualification(tmp_path, corruption):
    """Rows pr-qualification-forged-record and pr-link-marker-agent-forgery."""
    runner, config = _approved_two_reviewer_run(tmp_path)
    smaller = make_config(tmp_path, reviewer=("codex",))
    if corruption == "foreign-author":
        _append(runner, _forged_link_body(runner, smaller), login="mallory", author_id=9)
    elif corruption == "malformed":
        body = _forged_link_body(runner, smaller)
        match = BOARD_CHANGE_LINK_RE.search(body)
        bad = base64.urlsafe_b64encode(b'{"kind": "reviewer-board-change"}').decode()
        _append(runner, body.replace(match.group("payload"), bad), author_id=_ACTOR_ID)
    elif corruption == "broken-predecessor":
        _append(runner, _forged_link_body(runner, smaller, predecessor="0" * 64), author_id=_ACTOR_ID)
    elif corruption == "wrong-old-board":
        wrong = identity_from_config(make_config(tmp_path, reviewer=("gemini",))).payload()
        _append(runner, _forged_link_body(runner, smaller, old_board=wrong), author_id=_ACTOR_ID)
    elif corruption == "wrong-old-seat-binding":
        forged_binding = {"version": 1, "seats": [
            {"id": "x", "backend": "codex", "model_chain": ["m"], "effort": None},
        ]}
        _append(runner, _forged_link_body(runner, smaller, old_seat_binding=forged_binding),
                author_id=_ACTOR_ID)
    else:
        # Record-shaped spans the strict grammar cannot match are still records.
        payload = {"empty-payload": "", "illegal-characters": "!!not*base64!!",
                   "foreign-illegal-characters": "!!not*base64!!"}[corruption]
        author = 9 if corruption.startswith("foreign") else _ACTOR_ID
        _append(runner, f"Board change.\n\n<!-- {BOARD_CHANGE_LINK_MARKER}: {payload} -->",
                login="mallory" if author == 9 else "operator", author_id=author)
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="lineage is inconsistent|not authored"):
        run_pr_loop(runner, pr_number=77, config=smaller)
    assert len(runner.pr_payload["comments"]) == comments
    assert _agents(runner, start) == []
    with pytest.raises(AgentLoopError, match="No qualification or merge is permitted"):
        _qualify(runner, config)


def test_unknown_change_digest_on_a_scheduler_record_fails_closed(tmp_path):
    """Row pr-qualification-forged-record: a record names an unknown link digest."""
    runner, config = _approved_two_reviewer_run(tmp_path, pr_review_policy="selective-intermediate")
    record = _records(runner)[-1]
    forged = dataclasses.replace(
        record.metadata, round_number=2, reviewer_board_change_digest="e" * 64,
    )
    _append(runner, _attach_round_metadata("Forged.", forged), author_id=_ACTOR_ID)
    with pytest.raises(AgentLoopError, match="No qualification or merge is permitted"):
        _qualify(runner, config)
    with pytest.raises(AgentLoopError, match="unknown board-change or amendment digest"):
        run_pr_loop(runner, pr_number=77, config=config)


# --- pr-remove-with-blocker ------------------------------------------------------------


def _dispositions(reviewer, *item_ids, disposition="resolved"):
    from coding_review_agent_loop.protocol import ReviewItemDisposition

    return tuple(
        ReviewItemDisposition(item_id=item_id, reviewer=reviewer, disposition=disposition)
        for item_id in item_ids
    )


def _gemini_blocker_history(*, evidence=()):
    """Round 2 on head abc123: item-1 resolved by both, Gemini alone raises item-2."""
    item1 = _item(1, "Gemini")
    item2 = _item(2, "Gemini", round_number=2)
    return [
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item1,)),
        _legacy("coder", "Claude", 2, prior_items=(item1,)),
        _legacy("reviewer", "Codex", 2, state="approved", prior_items=(item1,),
                dispositions=_dispositions("Codex", "item-1")),
        _legacy("reviewer", "Gemini", 2, state="blocking", prior_items=(item1,),
                dispositions=_dispositions("Gemini", "item-1"), new_items=(item2,),
                evidence_requests=tuple(evidence)),
    ]


def _crash_after_link(monkeypatch):
    real_post = orchestrator.post_pr_comment

    def post(*args, **kwargs):
        real_post(*args, **kwargs)
        if str(kwargs["body"]).startswith(BOARD_CHANGE_LINK_HEADING):
            raise KeyboardInterrupt

    monkeypatch.setattr(orchestrator, "post_pr_comment", post)


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
def test_removed_reviewer_blocker_is_carried_reassigned_and_settled_items_stay_settled(
    tmp_path, monkeypatch, activation
):
    """Row pr-remove-with-blocker: Gemini is the sole publisher of item-2."""
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
        assert len(_links(runner)) == 1 and _agents(runner) == []
    # Neither remaining reviewer resolves item-2 yet: it blocks completion.
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.antigravity_outputs.append(structured_pr_review(
        reviewer="Antigravity", state="blocking",
        prior_item_dispositions=[{
            "item_id": "item-2", "disposition": "blocking",
            "note": "The worker still retries forever in src/worker.py; bound it and add a test.",
        }],
    ))
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    assert (payload["anchor_round"], payload["effective_from_round"]) == (2, 3)
    round_three = [r.metadata for r in _records(runner) if r.metadata.round_number == 3]
    reviews = [m for m in round_three if m.role == "reviewer"]
    assert {m.agent for m in reviews} == {"Codex", "Antigravity"}
    # item-1 was resolved in the anchor round and stays settled; item-2 is
    # carried, and both remaining reviewers had to disposition it.
    assert all([item.item_id for item in m.prior_items] == ["item-2"] for m in reviews)
    assert all(
        {d.item_id for d in m.dispositions} == {"item-2"} for m in reviews
    )
    assert not any(m.agent == "Gemini" for m in round_three)


def test_removed_reviewer_evidence_request_is_rebuilt_and_blocks(tmp_path):
    """Row pr-remove-with-blocker: an outgoing reviewer's exact-head evidence request."""
    from coding_review_agent_loop.board_change import carry_over_anchor_round

    records = _extract_round_metadata_records(
        [SimpleNamespace(body=body) for body in _gemini_blocker_history(
            evidence=("Attach the production deploy log.",)
        )],
        flow="pr",
    )
    anchor = [r for r in records if r.metadata.round_number == 2]
    reviews = [r for r in anchor if r.metadata.role == "reviewer"]
    carried = carry_over_anchor_round(
        prior_items=anchor[0].metadata.prior_items,
        completed_reviews=reviews,
        settled_new_items=(),
        records=records,
        round_number=2,
        reconciliation_mode="aggregate",
        evidence_clearances=(),
        next_item_number=3,
    )
    ids = [item.item_id for item in carried.items]
    assert ids[:1] == ["item-2"] and "item-1" not in ids
    evidence = [item for item in carried.items if item.item_id not in {"item-2"}]
    assert len(evidence) == 1 and evidence[0].item_id == "item-3"
    assert "deploy log" in evidence[0].text
    # A request already cleared at this head is not recreated...
    cleared = carry_over_anchor_round(
        prior_items=anchor[0].metadata.prior_items,
        completed_reviews=reviews,
        settled_new_items=(),
        records=records,
        round_number=2,
        reconciliation_mode="aggregate",
        evidence_clearances=tuple(
            (item.obligation_identity, "abc123") for item in evidence if item.obligation_identity
        ),
        next_item_number=3,
    )
    assert [item.item_id for item in cleared.items] == ["item-2"]
    # ...but at a different head it is a new obligation.
    moved = carry_over_anchor_round(
        prior_items=anchor[0].metadata.prior_items,
        completed_reviews=reviews,
        settled_new_items=(),
        records=records,
        round_number=2,
        reconciliation_mode="aggregate",
        evidence_clearances=tuple(
            (item.obligation_identity, "f" * 40) for item in evidence if item.obligation_identity
        ),
        next_item_number=3,
    )
    assert len(moved.items) == 2


# --- pr-legacy-partial-round-noop --------------------------------------------------------


@pytest.mark.parametrize("missing", ["crash", "unavailable-summary", "coder-only"])
def test_unchanged_plain_board_with_missing_reviewer_outcome_posts_nothing(tmp_path, missing):
    """Row pr-legacy-partial-round-noop: membership is never inferred from completeness."""
    history = [
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="approved"),
        _legacy("coder", "Claude", 2),
    ]
    if missing != "coder-only":
        history.append(_legacy("reviewer", "Codex", 2, state="approved"))
    if missing == "unavailable-summary":
        history.append(_legacy("summary", "Orchestrator", 2, phase="reconciliation"))
    runner = _legacy_runner(
        history,
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    # The round is never advanced for an unchanged board.
    new = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert new and all(m.round_number == 2 for m in new)


def test_historical_poster_missing_from_the_board_is_one_removal_link(tmp_path):
    """Row pr-legacy-partial-round-noop: B posted in an earlier legacy round."""
    item = _item(1, "Gemini")
    runner = _legacy_runner(
        [
            _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
            _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item,)),
            _legacy("coder", "Claude", 2, prior_items=(item,)),
        ],
        codex_outputs=[structured_pr_review(prior_item_dispositions=_RESOLVED_1)],
    )
    config = make_config(tmp_path, reviewer=("codex",))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    (payload,) = _link_payloads(runner)
    assert payload["removed"] == ["Gemini"] and payload["added"] == []
    # Gemini's open item was carried and reassigned to the remaining reviewer.
    (review,) = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.pr_board_root is not None
    ]
    assert [item.item_id for item in review.prior_items] == ["item-1"]
    assert [d.item_id for d in review.dispositions] == ["item-1"]


# --- pr-floor-unsigned-vs-signed ----------------------------------------------------------


def _amendment_from(message):
    start = message.index("Reviewer board amendment:")
    return message[start:].replace(
        "<why the removed reviewer cannot be reached>", "Gemini quota exhausted."
    )


def _two_reviewer_history(policy):
    """Round 1: Codex approved, Gemini raised item-1; the coder answered in round 2."""
    if policy == "legacy-plain":
        item = _item(1, "Gemini")
        return _legacy_runner([
            _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
            _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item,)),
            _legacy("coder", "Claude", 2, prior_items=(item,)),
        ])
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(
            reviewer="Google Gemini", state="blocking", blocking_items=["Bound the retry."]
        )],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=make_config(
            tmp_path_placeholder(), reviewer=("codex", "gemini"), pr_review_policy=policy,
        ))
    return runner


_TMP = {}


def tmp_path_placeholder():
    return _TMP["path"]


@pytest.mark.parametrize("policy", ["legacy-plain", "selective-intermediate"])
@pytest.mark.parametrize(
    ("floor", "refusal"),
    [
        ({"min_reviewers": 2}, "1 reviewer\\(s\\) is below the floor of 2"),
        ({"min_distinct_providers": 2}, "1 distinct provider\\(s\\) is below the floor of 2"),
    ],
    ids=["reviewer-floor", "provider-floor"],
)
@pytest.mark.parametrize("flags", ["reduced", "original"])
def test_below_floor_shrink_needs_a_signed_amendment(tmp_path, policy, floor, refusal, flags):
    """Row pr-floor-unsigned-vs-signed, both flag forms and both floor measures."""
    _TMP["path"] = tmp_path
    runner = _two_reviewer_history(policy)
    config_policy = "all-reviewers" if policy == "legacy-plain" else policy
    reduced = make_config(tmp_path, reviewer=("codex",), pr_review_policy=config_policy, **floor)
    original = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy=config_policy, **floor
    )
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match=refusal) as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    # Refused after read-only lineage validation: nothing posted, no agent.
    assert len(runner.pr_payload["comments"]) == comments
    assert _agents(runner, start) == [] and _links(runner) == []
    _append(runner, _amendment_from(str(refused.value)))
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    config = reduced if flags == "reduced" else original
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    assert any("Reviewer board amendment applied." in c for c in runner.comments)
    assert "gemini" not in _agents(runner, start)
    # The qualification gate accepts the signed board from fresh comments; the
    # run's own config is the effective (rebound) board whichever flags it got.
    _qualify(runner, reduced)


# --- pr-floor-candidate-link --------------------------------------------------------------


def _signed_three_to_two(tmp_path, **floor):
    """A selective PR approved by three reviewers, then a signed 3 -> 2 reduction."""
    runner, _ = _approved_two_reviewer_run(
        tmp_path, reviewers=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
    )
    reduced = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate",
        min_reviewers=3, **floor,
    )
    with pytest.raises(AgentLoopError, match="below the floor of 3") as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _append(runner, _amendment_from(str(refused.value)).replace("Gemini quota", "Antigravity quota"))
    runner.pr_payload["headRefOid"] = "def456"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert any("Reviewer board amendment applied." in c for c in runner.comments)
    return runner, reduced


def test_signed_reduction_stays_authorized_through_later_links_but_never_further(tmp_path):
    """Row pr-floor-candidate-link: frozen signed authorization, links never enlarge it."""
    runner, reduced = _signed_three_to_two(tmp_path)
    # A binding-only change of a remaining seat posts one link and stays admissible.
    rebound = dataclasses.replace(reduced, reviewer_codex_model="gpt-alternate")
    runner.pr_payload["headRefOid"] = "fed789"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=rebound) == 0
    (first,) = _link_payloads(runner)
    assert first["rebound"] == ["Codex"] and not first["added"] and not first["removed"]
    _qualify(runner, rebound)
    # An above-floor membership change (re-adding the signed-removed seat) is a link too.
    larger = make_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate", min_reviewers=3,
        reviewer_codex_model="gpt-alternate",
    )
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert [p["added"] for p in _link_payloads(runner)] == [[], ["Antigravity"]]
    # A further unsigned reduction below the signed board is refused before any post.
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    smaller = make_config(
        tmp_path, reviewer=("codex",), pr_review_policy="selective-intermediate",
        min_reviewers=3, reviewer_codex_model="gpt-alternate",
    )
    with pytest.raises(AgentLoopError, match="1 reviewer\\(s\\) is below the floor of 2"):
        run_pr_loop(runner, pr_number=77, config=smaller)
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner, start) == []


def test_link_never_creates_a_provider_exception(tmp_path):
    """Row pr-floor-candidate-link: a provider-reducing swap after a signed 3 -> 2."""
    runner, reduced = _signed_three_to_two(tmp_path, min_distinct_providers=2)
    comments = len(runner.pr_payload["comments"])
    # Gemini -> a second Codex-backed seat: one provider, and the signed chain
    # never lowered the provider count, so no authorization applies.
    seat = _seat(tmp_path, "second", "codex", "gpt-second")
    swapped = make_config(
        tmp_path, reviewer=("codex", seat), reviewer_seats=(seat,),
        pr_review_policy="selective-intermediate", min_reviewers=3, min_distinct_providers=2,
        reviewer_codex_model="gpt-main",
    )
    with pytest.raises(AgentLoopError, match="1 distinct provider\\(s\\) is below the floor of 2"):
        run_pr_loop(runner, pr_number=77, config=swapped)
    assert len(runner.pr_payload["comments"]) == comments


# --- resolver-level helpers -----------------------------------------------------------


def _identity(*seats, policy="all-reviewers", primary=None, models=None):
    from coding_review_agent_loop.review_scheduling import normalize_broad_rules

    models = models or {}
    return BoardIdentity(
        policy=policy, primary=primary, broad_rules=normalize_broad_rules(None),
        seats=tuple(
            BoardSeat(seat_id, backend, (models.get(seat_id, f"{seat_id}-model"),), None)
            for seat_id, backend in seats
        ),
    )


def _record_body(round_number, root, identity, *, change=None, amendment=None,
                 agent=None, binding="identity", **extra):
    seat_binding = identity.seat_binding() if binding == "identity" else binding
    return _attach_round_metadata(
        structured_pr_review(reviewer=agent or identity.names[0]),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent=agent or identity.names[0],
            round_number=round_number, subject="abc123", state="approved",
            pr_board_root=root.payload(), pr_board_identity=identity.payload(),
            reviewer_board_change_digest=change, reviewer_board_amendment_digest=amendment,
            seat_binding=seat_binding, **extra,
        ),
    )


def _link_body(root, predecessor, old, new, *, anchor, advance, inits=None, old_base=None):
    """``old_base`` is the predecessor segment's base board (its seat binding)."""
    payload = board_change_link_payload(
        pr_number=77, root=root, predecessor=predecessor, old_board=old, new_board=new,
        old_seat_binding=(old_base or old).seat_binding(), binding_init=inits or {},
        anchor_round=anchor, advance=advance,
    )
    return str(render_board_change_link(payload)), board_change_link_digest(payload)


def _signed_body(original, removed=(), round_number=2, *, restored=(), policy="all-reviewers"):
    return format_reviewer_board_amendment_comment(
        flow="pr", issue=None, pr_number=77, original_required_reviewers=original,
        policy=policy, primary_reviewer=None, removed_reviewers=removed,
        restored_reviewers=restored, effective_from_round=round_number,
        reason=None if restored and not removed else "seat-unavailable",
        rationale="Seat quota changed.",
    )


def _amendment_digest(body):
    from coding_review_agent_loop.board_amendment import parse_reviewer_board_amendment_records

    (record,), _ = parse_reviewer_board_amendment_records(body, comment_locator="c")
    return record.digest


def _resolve(bodies, *, configured=None, authors=None):
    from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments

    comments = [
        SimpleNamespace(body=body, author_id=(authors or {}).get(index, _ACTOR_ID))
        for index, body in enumerate(bodies)
    ]
    records = _extract_round_metadata_records(comments, flow="pr")
    amendments = collect_reviewer_board_amendments(comments, flow="pr", pr_number=77)
    return resolve_board_lineage(
        records, comments, amendments=amendments,
        configured=configured or _identity(("a", "codex")), pr_number=77,
        trusted_actor=("orchestrator", _ACTOR_ID),
        contract_from_metadata=_scheduler_contract_from_metadata,
        drift_error=lambda persisted, detail: AgentLoopError(f"drift: {detail}"),
    )


def test_frozen_signed_bindings_come_from_the_predecessor_link():
    """Row pr-floor-candidate-link: a signed removal after a backend-rebinding link.

    Root a/codex, b/claude, c/antigravity; a link rebinds b to codex; a signed
    removal of c then leaves a/codex, b/codex, which lowered the provider count
    from the link's board, so it authorizes one provider -- identically while
    pending and after activation.
    """
    root = _identity(("a", "codex"), ("b", "claude"), ("c", "antigravity"))
    rebound = _identity(("a", "codex"), ("b", "codex"), ("c", "antigravity"))
    link, link_digest = _link_body(root, root.digest, root, rebound, anchor=1, advance=True)
    signed = _signed_body(("a", "b", "c"), removed=("c",), round_number=3)
    history = [
        _record_body(1, root, root),
        link,
        _record_body(2, root, rebound, change=link_digest),
        signed,
    ]
    pending = _resolve(history)
    activated = _resolve([
        *history,
        _record_body(3, root, rebound.with_seats(rebound.seats[:2]),
                     amendment=_amendment_digest(signed), binding=rebound.seat_binding()),
    ])
    assert pending.pending_amendments and not activated.pending_amendments
    for lineage in (pending, activated):
        (authorization,) = lineage_floor_authorizations(lineage)
        assert [seat.backend for seat in authorization.seats] == ["codex", "codex"]
        assert authorization.lowered_providers and authorization.lowered_reviewers


# --- pr-link-crash-recovery ------------------------------------------------------------------


def test_pending_link_recovery_re_enters_the_advanced_round_exactly_once(tmp_path, monkeypatch):
    """Row pr-link-crash-recovery: a partially published reviewer round."""
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=swapped)
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=resolved_2)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1
    new_rounds = {r.metadata.round_number for r in _records(runner) if r.metadata.pr_board_root}
    assert new_rounds == {3}
    # Another rerun neither reposts nor advances again.
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1
    assert {r.metadata.round_number for r in _records(runner) if r.metadata.pr_board_root} == {3}


def test_coder_only_anchor_round_advances_once(tmp_path, monkeypatch):
    """Row pr-link-crash-recovery: the anchor round holds only a coder record."""
    item = _item(1, "Gemini")
    runner = _legacy_runner([
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item,)),
        _legacy("coder", "Claude", 2, prior_items=(item,)),
    ])
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    assert (payload["anchor_round"], payload["advance"], payload["effective_from_round"]) == (2, True, 3)
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=_RESOLVED_1)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    reviews = [r.metadata for r in _records(runner) if r.metadata.pr_board_root]
    assert {m.round_number for m in reviews} == {3}
    # The anchor round's coder output is the artifact the new board reviews.
    assert all(m.subject == "abc123" for m in reviews)


def test_prelaunch_only_anchor_round_keeps_its_round_and_ledger(tmp_path, monkeypatch):
    """Row pr-link-crash-recovery: only a scheduler-prelaunch record exists at R > 1."""
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="blocking", blocking_items=["Bound the retry."])],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate",
    )
    real_post = orchestrator.post_pr_comment
    prelaunches = []

    def interrupt_round_two_prelaunch(*args, **kwargs):
        real_post(*args, **kwargs)
        if "PR review scheduling audit:" in str(kwargs["body"]):
            prelaunches.append(True)
            if len(prelaunches) == 2:
                raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "post_pr_comment", interrupt_round_two_prelaunch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    highest = board_anchor(_records(runner))
    assert highest.round_number == 2
    assert [r.metadata.phase for r in _records(runner) if r.metadata.round_number == 2][-1] == (
        "scheduler-prelaunch"
    )
    assert [item.item_id for item in highest.ledger] == ["item-1"]
    swapped = make_config(
        tmp_path, reviewer=("codex", "antigravity"), pr_review_policy="selective-intermediate",
    )
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=_RESOLVED_1)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    (payload,) = _link_payloads(runner)
    assert (payload["anchor_round"], payload["effective_from_round"]) == (2, 3)
    reviews = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.round_number == 3
    ]
    assert {m.agent for m in reviews} >= {"Antigravity"}
    assert all([item.item_id for item in m.prior_items] == ["item-1"] for m in reviews)


def test_superseding_a_pending_link_copies_its_anchor_and_activates_only_the_last(
    tmp_path, monkeypatch
):
    """Row pr-link-crash-recovery: a second change while the first link is pending."""
    runner = _legacy_runner(_gemini_blocker_history())
    first = make_config(tmp_path, reviewer=("codex", "antigravity"))
    second = make_config(tmp_path, reviewer=("codex", "claude"), coder="codex")
    for config in (first, second):
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=config)
    one, two = _link_payloads(runner)
    assert two["predecessor"] == board_change_link_digest(one)
    for key in ("anchor_round", "advance", "effective_from_round"):
        assert two[key] == one[key]
    assert two["old_board"] == one["new_board"]
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.claude_outputs.append(
        structured_pr_review(reviewer="Anthropic Claude", prior_item_dispositions=resolved_2)
    )
    assert run_pr_loop(runner, pr_number=77, config=second) == 0
    assert len(_links(runner)) == 2
    governed = {
        r.metadata.reviewer_board_change_digest for r in _records(runner) if r.metadata.pr_board_root
    }
    assert governed == {board_change_link_digest(two)}
    assert "agy" not in _agents(runner)
    _qualify(runner, second)


def test_superseding_link_with_mismatched_rounds_fails_closed():
    """Row pr-link-crash-recovery: a superseding link must copy anchor and advance."""
    root = _identity(("a", "codex"), ("b", "claude"))
    first_board = _identity(("a", "codex"), ("c", "gemini"))
    second_board = _identity(("a", "codex"), ("d", "antigravity"))
    link_one, digest_one = _link_body(root, root.digest, root, first_board, anchor=1, advance=True)
    link_two, _ = _link_body(root, digest_one, first_board, second_board, anchor=2, advance=False)
    with pytest.raises(AgentLoopError, match="does not copy its anchor round"):
        _resolve([_record_body(1, root, root), link_one, link_two])


@pytest.mark.parametrize("state", ["consistent-empty", "records-after-link", "wrong-anchor"])
def test_pending_link_activation_follows_durable_history(tmp_path, state):
    """Row pr-link-crash-recovery: consistent and inconsistent pending links."""
    root = _identity(("a", "codex"), ("b", "claude"))
    new = _identity(("a", "codex"), ("c", "gemini"))
    if state == "consistent-empty":
        link, _ = _link_body(root, root.digest, root, new, anchor=1, advance=False)
        lineage = _resolve([link])
        assert [e.effective_from_round for e in lineage.pending_links] == [1]
        return
    if state == "records-after-link":
        link, _ = _link_body(root, root.digest, root, new, anchor=1, advance=True)
        bodies = [_record_body(1, root, root), link, _record_body(1, root, root)]
    else:
        link, _ = _link_body(root, root.digest, root, new, anchor=3, advance=True)
        bodies = [_record_body(1, root, root), link]
    with pytest.raises(AgentLoopError, match="lineage is inconsistent"):
        _resolve(bodies)


# --- pr-primary-change -------------------------------------------------------------------


@pytest.mark.parametrize("change", ["new-primary", "primary-model", "primary-effort"])
def test_primary_change_restarts_the_primary_phase_and_carries_secondary_obligations(
    tmp_path, monkeypatch, change
):
    """Row pr-primary-change: a new primary, or a model or effort change on the
    same primary seat, restarts the primary phase; the panel's open item
    survives it and is dispositioned by its owner when the panel reopens."""
    from test_orchestrator_pr import _audit_phases, _staged_config, _staged_review

    runner = _staged_panel_history(tmp_path, monkeypatch)
    assert _agents(runner)[:3] == ["codex", "gemini", "agy"]
    overrides = {
        "new-primary": {"primary_reviewer": "antigravity"},
        "primary-model": {"reviewer_codex_model": "gpt-primary-next"},
        "primary-effort": {"reviewer_codex_reasoning_effort": "high"},
    }[change]
    changed = _staged_config(tmp_path, pre_review_tests=False, **overrides)
    primary, primary_agent = (
        ("Antigravity", "agy") if change == "new-primary" else ("Codex", "codex")
    )
    # A non-owner's disposition is a note under owner-scoped reconciliation:
    # item-1 stays pending on its owner, Gemini.
    carried = [{"item_id": "item-1", "disposition": "resolved",
                "note": "Looks addressed to me; Gemini owns the finding."}]
    # The restarted primary alone first, then the reopened panel at the same
    # head, where Gemini dispositions its own carried finding.
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=carried))
    runner.gemini_outputs.append(_staged_review(
        reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=carried))
    phases_before = len(_audit_phases(runner))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=changed) == 0
    (payload,) = _link_payloads(runner)
    assert payload["new_board"]["primary"] == primary
    if change == "new-primary":
        assert payload["old_board"]["primary"] == "Codex"
    else:
        assert payload["rebound"] == ["Codex"]
    # The restarted primary runs alone first -- no pre-panel stop for the
    # carried secondary obligation -- then the panel reopens.
    agents = _agents(runner, start)
    assert agents[0] == primary_agent and "gemini" in agents[1:] and "claude" not in agents
    phases = [phase for phase, _head in _audit_phases(runner)][phases_before:]
    assert phases[0] == "primary" and "secondary-audit" in phases
    restarted = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == primary
        and r.metadata.reviewer_board_change_digest == board_change_link_digest(payload)
    ]
    # item-1 was still pending on Gemini when the restarted primary reviewed.
    assert restarted and [item.item_id for item in restarted[0].prior_items] == ["item-1"]
    assert not any("pre-panel safety" in c for c in runner.comments)
    _qualify(runner, changed)


def test_only_carried_pre_epoch_obligations_are_exempt_from_the_pre_panel_stop():
    """Row pr-primary-change: a secondary-owned item created after the epoch still stops."""
    from coding_review_agent_loop.review_scheduling import (
        PrePanelSafetyError,
        ReviewObligation,
        SchedulerSnapshot,
        TransitionClassification,
        make_contract,
        select_reviewers,
    )

    contract = make_contract(("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Antigravity")
    obligation = ReviewObligation(
        item_id="item-1", status="blocking", scope=("src/worker.py",),
        resolution_owners=("Gemini",), pending_owners=("Gemini",),
    )
    narrow = TransitionClassification("narrow", "scoped fix")

    def decide(carried):
        return select_reviewers(
            SchedulerSnapshot(
                previous_sha="a", current_sha="b", contract=contract,
                obligations=(obligation,), carried_obligations=carried,
            ),
            narrow,
        )

    assert decide(("item-1",)).selected_reviewers == ("Antigravity",)
    with pytest.raises(PrePanelSafetyError, match="item-1"):
        decide(())


# --- pr-resume-chain ---------------------------------------------------------------------


def test_link_then_signed_amendment_then_link_resolves_one_ordered_chain(tmp_path):
    """Row pr-resume-chain: link -> signed amendment of the link's board -> link."""
    runner, _ = _approved_two_reviewer_run(tmp_path, pr_review_policy="selective-intermediate")
    larger = make_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate", min_reviewers=3,
    )
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    # The signed amendment after the link amends the link's new board.
    reduced = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate",
        min_reviewers=3,
    )
    with pytest.raises(AgentLoopError, match="below the floor of 3") as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    signed = _amendment_from(str(refused.value))
    assert '"Antigravity"' in signed.split('"original_required_reviewers"')[1].split("]")[0]
    _append(runner, signed)
    runner.pr_payload["headRefOid"] = "def456"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    # Original flags of the link's board take the #1133 rebind.
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert len(_links(runner)) == 1
    def lineage(config=larger):
        return _board_state(runner, config).lineage

    first = lineage()
    assert [element.kind for element in first.elements] == ["link", "amendment"]
    assert first.current_board == ("Codex", "Gemini")
    # A rerun derives the identical lineage and posts nothing.
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert lineage().elements == first.elements and len(_links(runner)) == 1
    # A further differing board adds exactly one link after the amendment.
    swapped = make_config(
        tmp_path, reviewer=("codex", "claude"), coder="codex",
        pr_review_policy="selective-intermediate", min_reviewers=3,
    )
    runner.claude_outputs.append(structured_pr_review(reviewer="Anthropic Claude"))
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 2
    after = lineage(swapped)
    assert [element.kind for element in after.elements] == ["link", "amendment", "link"]
    _qualify(runner, swapped)


def test_signed_restoration_after_a_link_only_restores_seats_signed_off_since_it():
    """Row pr-resume-chain: restorations are bounded by the latest link."""
    root = _identity(("a", "codex"), ("b", "claude"), ("c", "antigravity"))
    abd = _identity(("a", "codex"), ("b", "claude"), ("d", "gemini"))
    ab = abd.with_seats(abd.seats[:2])
    # c was removed by a signed amendment before the link: a restoration after
    # the link cannot bring it back.
    removal = _signed_body(("a", "b", "c"), removed=("c",), round_number=2)
    link, _ = _link_body(root, _amendment_digest(removal), root.with_seats(root.seats[:2]), abd,
                         anchor=2, advance=True, old_base=root)
    restore_c = _signed_body(("a", "b", "d"), restored=("c",), round_number=4)
    history = [
        _record_body(1, root, root),
        removal,
        _record_body(2, root, root.with_seats(root.seats[:2]), amendment=_amendment_digest(removal),
                     binding=root.seat_binding()),
        link,
    ]
    with pytest.raises(AgentLoopError, match="restored reviewer|not on the persisted board|already"):
        _resolve([*history, restore_c])
    # d was added by the link and signed off after it: its restoration is valid.
    remove_d = _signed_body(("a", "b", "d"), removed=("d",), round_number=4)
    restore_d = _signed_body(("a", "b"), restored=("d",), round_number=5)
    lineage = _resolve([*history, remove_d, restore_d])
    assert lineage.current_board == ("a", "b", "d")
    assert lineage.restoration_rounds == {"d": 5}
    del ab


# --- pr-legacy-root-anchor ----------------------------------------------------------------


def _strip_board_identity(runner):
    """Simulate pre-upgrade history: drop every board identity field."""
    from test_orchestrator_pr import _rewrite_pr_metadata

    keys = ("pr_board_root", "pr_board_identity", "pr_board_binding_init",
            "reviewer_board_change_digest")

    def strip(payload):
        if not any(key in payload for key in keys):
            return None
        return {key: value for key, value in payload.items() if key not in keys}

    _rewrite_pr_metadata(runner, strip)


def test_legacy_signed_chain_keeps_its_original_root_after_upgrade(tmp_path):
    """Row pr-legacy-root-anchor: legacy C0 -> signed C1, then upgraded records."""
    runner, reduced = _signed_three_to_two(tmp_path)
    _strip_board_identity(runner)
    assert all(r.metadata.pr_board_root is None for r in _records(runner))
    runner.pr_payload["headRefOid"] = "fed789"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded
    roots = {BoardIdentity.from_payload(m.pr_board_root).names for m in upgraded}
    assert roots == {("Codex", "Gemini", "Antigravity")}
    assert all(
        BoardIdentity.from_payload(m.pr_board_identity).names == ("Codex", "Gemini")
        for m in upgraded
    )
    assert all(m.reviewer_board_amendment_digest for m in upgraded)
    assert _links(runner) == []
    _qualify(runner, reduced)
    runner.pr_payload["headRefOid"] = "aaa111"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0


def test_legacy_link_root_and_unknown_bindings_initialize_once(tmp_path, monkeypatch):
    """Row pr-legacy-root-anchor: A -> link B keeps root A; later model change is one link."""
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    # The link carries the derived legacy root: plain history records no board,
    # so the root is the configured board plus every historical poster, with
    # membership-only bindings.  Antigravity is adopted; only Gemini is removed.
    root = BoardIdentity.from_payload(payload["root"])
    assert root.names == ("Codex", "Antigravity", "Gemini")
    assert all(s.model_chain is None for s in root.seats)
    assert (payload["added"], payload["removed"]) == ([], ["Gemini"])
    # The seats that stay are initialized by the link, with no link of their own.
    assert set(payload["binding_init"]) == {"Codex", "Antigravity"}
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=resolved_2)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert {BoardIdentity.from_payload(m.pr_board_root).names for m in upgraded} == {
        ("Codex", "Antigravity", "Gemini")
    }
    # A later model change of an initialized seat is exactly one link and
    # needs Codex's fresh approval.
    moved = dataclasses.replace(swapped, reviewer_codex_model="gpt-later")
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=moved) == 0
    assert len(_links(runner)) == 2
    assert _link_payloads(runner)[1]["rebound"] == ["Codex"]
    assert _agents(runner, start) == ["codex"]
    _qualify(runner, moved)


def test_unchanged_upgrade_then_model_change_posts_one_link(tmp_path):
    """Row pr-legacy-root-anchor: an unchanged plain upgrade, then a model change."""
    runner = _legacy_runner(
        [_legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
         _legacy("reviewer", "Gemini", 1, subject="abc000", state="approved"),
         _legacy("coder", "Claude", 2)],
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    first = next(r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None)
    assert set(first.pr_board_binding_init) == {"Codex", "Gemini"}
    # A resume after the upgrade reads the initialization: no link.
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    moved = dataclasses.replace(config, reviewer_codex_model="gpt-later")
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=moved) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["Codex"] and payload["binding_init"] == {}
    assert _agents(runner, start) == ["codex"]


# --- pr-all-reviewers-board-change ------------------------------------------------------------


def test_all_reviewers_plain_to_named_and_back(tmp_path):
    """Row pr-all-reviewers-board-change: plain -> named -> plain, one link each."""
    runner, plain = _approved_two_reviewer_run(tmp_path)
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    named = _named(tmp_path, "codex", flash)
    runner.codex_outputs.append(structured_pr_review())
    runner.antigravity_outputs.append(
        (structured_pr_review(reviewer="flash (Google Antigravity: Model A)"), 0)
    )
    assert run_pr_loop(runner, pr_number=77, config=named) == 0
    (to_named,) = _link_payloads(runner)
    assert (to_named["added"], to_named["removed"]) == (["flash"], ["Gemini"])
    assert to_named["old_seat_binding"] is None and to_named["new_seat_binding"]["version"] == 1
    # Records under the named board carry its seat binding.
    named_records = [
        r.metadata for r in _records(runner)
        if r.metadata.reviewer_board_change_digest == board_change_link_digest(to_named)
    ]
    assert named_records and all(m.seat_binding == to_named["new_seat_binding"] for m in named_records)
    _qualify(runner, named)
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=plain) == 0
    back = _link_payloads(runner)[1]
    assert (back["added"], back["removed"]) == (["Gemini"], ["flash"])
    assert back["new_seat_binding"] is None
    _qualify(runner, plain)
    assert run_pr_loop(runner, pr_number=77, config=plain) == 0
    assert len(_links(runner)) == 2


def test_all_reviewers_same_seat_backend_change(tmp_path):
    """Row pr-all-reviewers-board-change: a named seat moves to another backend."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        antigravity_outputs=[(structured_pr_review(reviewer="flash (Google Antigravity: Model A)"), 0)],
    )
    config = _named(tmp_path, "codex", flash)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    moved = _seat(tmp_path, "flash", "gemini", "gemini-pro")
    rebound = _named(tmp_path, "codex", moved)
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="flash (Google Gemini: gemini-pro)"))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=rebound) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["flash"]
    assert "agy" not in _agents(runner, start) and "gemini" in _agents(runner, start)
    _qualify(runner, rebound)


@pytest.mark.parametrize("defect", ["identity-without-root", "foreign-root", "wrong-identity"])
def test_malformed_board_history_fails_closed(defect):
    """Row pr-all-reviewers-board-change: malformed board history is never accepted."""
    root = _identity(("a", "codex"), ("b", "claude"))
    other = _identity(("a", "codex"), ("c", "gemini"))
    if defect == "identity-without-root":
        bad = _attach_round_metadata(
            "x", PostedRoundMetadata(
                flow="pr", role="summary", agent="Orchestrator", round_number=2, subject="abc123",
                reviewer_board_change_digest="a" * 64,
            ),
        )
    elif defect == "foreign-root":
        bad = _record_body(2, other, other)
    else:
        bad = _record_body(2, root, other)
    with pytest.raises(AgentLoopError, match="lineage is inconsistent"):
        _resolve([_record_body(1, root, root), bad])


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_removal_with_carried_approvals_qualifies_and_reruns_cleanly(tmp_path, policy):
    """Rows pr-all-reviewers-board-change and pr-link-crash-recovery.

    Removing a reviewer whose peers already approved the current head may need
    no new reviewer turn; the link is then the board the run reviewed with.
    """
    runner, _ = _approved_two_reviewer_run(
        tmp_path, reviewers=("codex", "gemini", "antigravity"), pr_review_policy=policy,
    )
    smaller = make_config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy=policy)
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=smaller) == 0
    assert _agents(runner, start) == []
    (payload,) = _link_payloads(runner)
    assert payload["removed"] == ["Antigravity"]
    _qualify(runner, smaller)
    assert run_pr_loop(runner, pr_number=77, config=smaller) == 0
    assert len(_links(runner)) == 1 and _agents(runner, start) == []


def test_signed_amendment_posted_during_the_run_blocks_qualification(tmp_path):
    """Row pr-qualification-forged-record: the gate re-reads the lineage from fresh comments."""
    runner, config = _approved_two_reviewer_run(
        tmp_path, reviewers=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
    )
    _append(runner, _signed_body(
        ("Codex", "Gemini", "Antigravity"), removed=("Antigravity",), round_number=2,
        policy="selective-intermediate",
    ).replace('"seat-unavailable"', '"backend-unavailable"'))
    with pytest.raises(AgentLoopError, match="No qualification or merge is permitted"):
        _qualify(runner, config)


# --- carried-finding provenance (D4b) ---------------------------------------------------


def _review_record(index, agent, new_items, *, round_number=2):
    return SimpleNamespace(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="reviewer", agent=agent, round_number=round_number,
            subject="abc123", state="blocking", new_items=tuple(new_items),
        ),
    )


@pytest.mark.parametrize("case", [
    "unknown-owner", "other-seat-owner", "record-collision", "ledger-collision",
])
def test_carry_over_refuses_unverifiable_or_conflicting_findings(case):
    """Row pr-remove-with-blocker: the publisher must own its findings, and a
    repeated item ID must carry identical content wherever it repeats."""
    from coding_review_agent_loop.board_change import carry_over_anchor_round

    prior = ()
    if case == "unknown-owner":
        reviews = [_review_record(1, "Gemini", [_item(2, "Somebody Else")])]
    elif case == "other-seat-owner":
        reviews = [_review_record(1, "Gemini", [_item(2, "Codex")])]
    elif case == "record-collision":
        first = _item(2, "Gemini")
        reviews = [
            _review_record(1, "Gemini", [first]),
            _review_record(2, "Gemini", [dataclasses.replace(first, text="another claim")]),
        ]
    else:
        prior = (_item(2, "Gemini"),)
        reviews = [_review_record(1, "Gemini", [dataclasses.replace(prior[0], text="changed")])]
    with pytest.raises(AgentLoopError, match="cannot verify the owner|conflicting content"):
        carry_over_anchor_round(
            prior_items=prior, completed_reviews=reviews, settled_new_items=(),
            records=reviews, round_number=2, reconciliation_mode="aggregate",
            evidence_clearances=(), next_item_number=3,
        )


# --- signed original board as the legacy root (D1a) ---------------------------------------


@pytest.mark.parametrize("shape", ["coder-only", "never-published", "non-tail"])
@pytest.mark.parametrize("floor", ["reviewers", "providers"])
@pytest.mark.parametrize("flags", ["reduced", "original"])
def test_signed_original_board_is_the_legacy_plain_root(tmp_path, shape, floor, flags):
    """Rows pr-floor-unsigned-vs-signed and pr-legacy-root-anchor.

    Plain legacy history with signed amendments takes the earliest signed
    original board as its root, so a reduction of a never-published seat or a
    non-tail seat resolves in both flag forms and for both floor measures.
    """
    if shape == "non-tail":
        board = ("Codex", "Gemini", "Antigravity")
        history = [_legacy("reviewer", name, 1, subject="abc000", state="approved") for name in board]
        removed, kept = ("Gemini",), ("codex", "antigravity")
    else:
        board = ("Codex", "Gemini")
        history = (
            [] if shape == "coder-only"
            else [_legacy("reviewer", "Codex", 1, subject="abc000", state="approved")]
        )
        removed, kept = ("Gemini",), ("codex",)
    history.append(_legacy("coder", "Claude", 2 if history else 1))
    reentry = 2 if len(history) > 1 else 1
    runner = _legacy_runner(history)
    minimum = len(board)
    floor_kwargs = (
        {"min_reviewers": minimum} if floor == "reviewers" else {"min_distinct_providers": minimum}
    )
    plain = {"Codex": "codex", "Gemini": "gemini", "Antigravity": "antigravity"}
    original = make_config(tmp_path, reviewer=tuple(plain[name] for name in board), **floor_kwargs)
    reduced = make_config(tmp_path, reviewer=kept, **floor_kwargs)
    comments = len(runner.pr_payload["comments"])
    with pytest.raises(AgentLoopError, match="below the floor"):
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner) == []
    _append(runner, _signed_body(board, removed=removed, round_number=reentry))
    runner.codex_outputs.append(structured_pr_review())
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    config = reduced if flags == "reduced" else original
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert "gemini" not in _agents(runner) and _links(runner) == []
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded and all(
        BoardIdentity.from_payload(m.pr_board_root).names == board for m in upgraded
    )
    _qualify(runner, reduced)


# --- evidence requests across a board change (item-1, row pr-remove-with-blocker) ----------


_DEPLOY_LOG = "Attach the production deploy log for this head."
_CANARY = "Attach the canary dashboard for this head."


def _evidence_history(release_head):
    """Gemini's deploy-log request was cleared at ``release_head`` by a durable
    release; in round 2 Gemini (outgoing) re-emits it and Codex (retained)
    emits a new request, at head abc123."""
    from coding_review_agent_loop.round_state import EvidenceReleaseRecord
    from coding_review_agent_loop.unresolved_items import evidence_obligation_identity

    release = _attach_round_metadata(
        "## Exact-head evidence accepted\n\n-- coding-review-agent-loop",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1,
            subject="abc123", state="blocking", phase="evidence-release",
            evidence_release=EvidenceReleaseRecord(
                released_head="abc123", reason="evidence-cleared",
                signed_requirement_ids_surfaced=(), allowed_rounds=5,
                watch_failure_extension_used=False, watch_head_extension_used=False,
            ),
            evidence_clearances=((evidence_obligation_identity(_DEPLOY_LOG, "Gemini"), release_head),),
        ),
    )
    return [
        _legacy("reviewer", "Codex", 1, state="approved"),
        _legacy("reviewer", "Gemini", 1, state="approved", evidence_requests=(_DEPLOY_LOG,)),
        release,
        _legacy("reviewer", "Codex", 2, state="approved", evidence_requests=(_CANARY,)),
        _legacy("reviewer", "Gemini", 2, state="approved", evidence_requests=(_DEPLOY_LOG,)),
    ]


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
@pytest.mark.parametrize("cleared_at", ["same-head", "other-head"])
def test_evidence_requests_survive_a_board_change_with_their_head_scoped_clearances(
    tmp_path, monkeypatch, activation, cleared_at
):
    """Item-1: durable clearances accumulated before the anchor round are kept.

    The outgoing reviewer's request cleared at this head stays cleared; at
    another head it is a new obligation.  The retained reviewer's request is
    rebuilt.  Open evidence obligations block qualification.
    """
    runner = _legacy_runner(_evidence_history("abc123" if cleared_at == "same-head" else "f" * 40))
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
    # Every reviewer dispositions the carried evidence obligations (item-1,
    # and item-2 at another head); on a human-only evidence obligation that is
    # only a note, so the obligation stays owed.
    owed = [
        {"item_id": item_id, "disposition": "resolved",
         "note": "The code looks right; the human evidence is still owed."}
        for item_id in (("item-1", "item-2") if cleared_at == "other-head" else ("item-1",))
    ]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=owed))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=owed)
    )
    try:
        outcome = run_pr_loop(runner, pr_number=77, config=swapped)
    except AgentLoopError:
        outcome = None
    (payload,) = _link_payloads(runner)
    assert payload["effective_from_round"] == 3
    round_three = [
        r.metadata for r in _records(runner)
        if r.metadata.round_number == 3 and r.metadata.role == "reviewer"
    ]
    assert round_three
    texts = {item.text for item in round_three[0].prior_items if item.obligation_kind}
    assert _CANARY in texts
    assert (_DEPLOY_LOG in texts) is (cleared_at == "other-head")
    # Human-only evidence is still owed: the run neither approves nor merges.
    assert outcome != 0
    assert not any(c.startswith("PR #77 approved") for c in runner.comments)


_LIVE_SUITE = "Run the live suite at this exact head and attach the output."
_LIVE_LOG = "Attach the live deploy log for this exact head."


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
def test_frozen_evidence_is_answered_by_the_new_board_after_a_change(
    tmp_path, monkeypatch, activation
):
    """Item-9.s1 (row pr-remove-with-blocker), driven by the live loop.

    Round 1 freezes on two evidence requests: one from Gemini (the outgoing
    reviewer) and one from Codex (retained).  A human signs the evidence.  The
    evidence pass then stops after Gemini resolved both and re-emitted its own
    request, before reconciliation.  Replay never clears evidence from an
    unreconciled pass, so carry-over keeps both obligations frozen, and the
    freeze stays in force across the change.  The new board answers the
    signed input in an evidence-response pass whose reconciliation clears
    both at abc123.  Gemini's re-emission reopens nothing, and the PR
    qualifies.
    """
    from test_orchestrator_pr import _evidence_review, _signed

    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_LIVE_LOG])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini", evidence=[_LIVE_SUITE])],
    )
    original = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(orchestrator.HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=original)
    runner.pr_payload["comments"].append(
        _signed("Live suite and deploy log at abc123: 11 passed, deploy healthy.", 1)
    )
    resolved = [
        {"item_id": item_id, "disposition": "resolved",
         "note": "The signed output covers this exact head."}
        for item_id in ("item-1", "item-2")
    ]
    runner.codex_outputs.append(_evidence_review(dispositions=resolved, hr=True))
    runner.gemini_outputs.append(_evidence_review(
        reviewer="Google Gemini", dispositions=resolved, evidence=[_LIVE_SUITE], hr=True,
    ))
    real_post = orchestrator.post_pr_comment

    def stop_after_gemini(*args, **kwargs):
        real_post(*args, **kwargs)
        (record,) = _extract_round_metadata_records(
            [SimpleNamespace(body=str(kwargs["body"]))], flow="pr"
        ) or (None,)
        if record is not None and record.metadata.role == "reviewer" and record.metadata.dispositions:
            if record.metadata.agent == "Gemini":
                raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "post_pr_comment", stop_after_gemini)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=original)
    anchor = board_anchor(_records(runner))
    in_anchor = [r.metadata for r in _records(runner) if r.metadata.round_number == anchor.round_number]
    # The anchor round's dispositions were never reconciled.
    assert not any(m.phase == "reconciliation" for m in in_anchor)
    assert any(m.agent == "Gemini" and m.evidence_requests == (_LIVE_SUITE,) for m in in_anchor)
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
    runner.codex_outputs.append(_evidence_review(dispositions=resolved, hr=True))
    runner.antigravity_outputs.append(
        _evidence_review(reviewer="Antigravity", dispositions=resolved, hr=True)
    )
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    (payload,) = _link_payloads(runner)
    assert payload["anchor_round"] == anchor.round_number and payload["advance"] is True
    assert "gemini" not in _agents(runner, start) and "claude" not in _agents(runner, start)
    after = [
        r.metadata for r in _records(runner)
        if r.metadata.round_number == payload["effective_from_round"] and r.metadata.role == "reviewer"
    ]
    assert after and {m.reviewer_board_change_digest for m in after} == {
        board_change_link_digest(payload)
    }
    # The freeze stayed in force across the change: the new board answered the
    # signed input in an evidence-response pass, and its reconciliation
    # cleared both requests at abc123 (Gemini's, whose requester left the
    # board, needed every configured reviewer).  Gemini's re-emission at that
    # head reopened nothing: one freeze, one release, nothing owed.
    assert {m.phase for m in after} == {"evidence-response"}
    releases = [r.metadata for r in _records(runner) if r.metadata.phase == "evidence-release"]
    assert [m.evidence_release.reason for m in releases] == ["evidence-cleared"]
    assert {identity for identity, head in releases[0].evidence_clearances if head == "abc123"} >= {
        item.obligation_identity for item in after[0].prior_items if item.obligation_kind
    }
    assert len([r for r in _records(runner) if r.metadata.phase == "evidence-freeze"]) == 1
    _qualify(runner, swapped)


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
@pytest.mark.parametrize("head", ["same-head", "new-head"])
def test_evidence_released_in_the_anchor_round_stays_cleared_at_that_head(
    tmp_path, monkeypatch, activation, head
):
    """Item-9.s1 (row pr-remove-with-blocker), driven by the live loop.

    Round 1 freezes on Codex's (retained) and Gemini's (outgoing) evidence
    requests.  A human signs the evidence; in the evidence pass each requester
    resolves its own request and Gemini re-emits its request at the same head.
    The anchor round's reconciliation clears both and the release (with its
    clearances) is recorded, then the run stops.  After the board change --
    immediate or recovered after a crash -- carry-over keeps that
    post-reconciliation ledger as it is (D4b step 2), so the re-emission at
    abc123 reopens nothing: no obligation, no second freeze, and the PR
    qualifies.  (Rebuilding an unreconciled anchor round's requests against
    the accumulated clearances is pinned by
    test_evidence_requests_survive_a_board_change_with_their_head_scoped_clearances.)  At a different head the retained requester's same request is a
    new obligation and freezes that head.
    """
    from test_orchestrator_pr import _evidence_review, _signed

    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_LIVE_LOG])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini", evidence=[_LIVE_SUITE])],
    )
    original = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(orchestrator.HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=original)
    runner.pr_payload["comments"].append(
        _signed("Live suite and deploy log at abc123: 11 passed, deploy healthy.", 1)
    )
    resolved = [
        {"item_id": item_id, "disposition": "resolved",
         "note": "The signed output covers this exact head."}
        for item_id in ("item-1", "item-2")
    ]
    runner.codex_outputs.append(_evidence_review(dispositions=resolved, hr=True))
    runner.gemini_outputs.append(_evidence_review(
        reviewer="Google Gemini", dispositions=resolved, evidence=[_LIVE_SUITE], hr=True,
    ))
    real_post = orchestrator.post_pr_comment

    def stop_after_release(*args, **kwargs):
        real_post(*args, **kwargs)
        (record,) = _extract_round_metadata_records(
            [SimpleNamespace(body=str(kwargs["body"]))], flow="pr"
        ) or (None,)
        if record is not None and record.metadata.phase == "evidence-release":
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "post_pr_comment", stop_after_release)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=original)
    (release,) = [r.metadata for r in _records(runner) if r.metadata.phase == "evidence-release"]
    assert release.evidence_release.reason == "evidence-cleared"
    cleared = {identity for identity, at in release.evidence_clearances if at == "abc123"}
    assert len(cleared) == 2
    anchor = board_anchor(_records(runner))
    assert release.round_number == anchor.round_number
    assert any(
        r.metadata.agent == "Gemini" and r.metadata.round_number == anchor.round_number
        and r.metadata.dispositions and r.metadata.evidence_requests == (_LIVE_SUITE,)
        for r in _records(runner)
    )
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if head == "new-head":
        runner.pr_payload["headRefOid"] = "def456"
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
    if head == "same-head":
        runner.codex_outputs.append(_evidence_review(hr=True))
        runner.antigravity_outputs.append(_evidence_review(reviewer="Antigravity", hr=True))
        start = len(runner.commands)
        assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
        assert "gemini" not in _agents(runner, start)
    else:
        # At def456 Codex, the retained requester, asks for its evidence again.
        runner.codex_outputs.append(_evidence_review(evidence=[_LIVE_LOG], hr=True))
        runner.antigravity_outputs.append(_evidence_review(reviewer="Antigravity", hr=True))
        with pytest.raises(orchestrator.HumanDecisionRequiredError, match="frozen at head def456"):
            run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    assert payload["anchor_round"] == anchor.round_number and payload["advance"] is True
    after = [r.metadata for r in _records(runner)
             if r.metadata.round_number >= payload["effective_from_round"]]
    assert after and all(
        m.reviewer_board_change_digest == board_change_link_digest(payload) for m in after
    )
    # Nothing the anchor round cleared is carried as an obligation.
    carried = [item for m in after if m.role == "reviewer" for item in m.prior_items
               if item.obligation_kind and item.obligation_identity in cleared]
    assert carried == []
    freezes = [r.metadata for r in _records(runner) if r.metadata.phase == "evidence-freeze"]
    if head == "same-head":
        assert [m.subject for m in freezes] == ["abc123"]
        assert not any(m.phase == "evidence-response" for m in after)
        _qualify(runner, swapped)
    else:
        assert [m.subject for m in freezes] == ["abc123", "def456"]
        owed = [item for item in freezes[-1].prior_items if item.obligation_kind]
        # The same obligation identity, cleared only at abc123, is owed at def456.
        assert [(item.text, item.reviewer, item.candidate_head_sha) for item in owed] == [
            (_LIVE_LOG, "Codex", "def456")
        ]
        assert owed[0].obligation_identity in cleared


def _staged_panel_history(tmp_path, monkeypatch):
    """Codex (primary) approves, the panel opens, Gemini raises item-1; then stop."""
    from test_orchestrator_pr import _staged_config, _staged_review
    from coding_review_agent_loop.review_scheduling import TransitionClassification

    monkeypatch.setattr(
        orchestrator, "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(
            reviewer="Google Gemini", state="blocking",
            blocking_items=[{"text": "panel gap", "fix_scope": ["src/worker.py"]}],
        )],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pre_review_tests=False))
    return runner


@pytest.mark.parametrize("opening", ["qualified", "never-opened"])
def test_carried_obligations_require_a_qualified_pre_epoch_opening(tmp_path, monkeypatch, opening):
    """Item-4 (row pr-primary-change): only items created under a qualified
    panel opening before the epoch are exempt from the pre-panel stop."""
    from test_orchestrator_pr import _staged_config
    from coding_review_agent_loop.board_change import (
        carried_secondary_obligations,
        pre_epoch_panel_items,
    )

    runner = _staged_panel_history(tmp_path, monkeypatch)
    changed = _staged_config(tmp_path, primary_reviewer="antigravity", pre_review_tests=False)
    lineage = _board_state(runner, changed).lineage
    assert lineage.primary_epoch_round == 4
    records = _records(runner)
    if opening == "never-opened":
        # The same secondary-owned item without the qualified opening record.
        records = [
            r for r in records
            if not (r.metadata.scheduler_phase == "secondary-audit" and r.metadata.role == "summary")
        ]
    eligible = pre_epoch_panel_items(records, lineage)
    ledger = [item for r in records for item in r.metadata.prior_items][-1:]
    carried = carried_secondary_obligations(
        ledger, epoch_round=4, primary="Antigravity", eligible=eligible,
    )
    assert carried == (("item-1",) if opening == "qualified" else ())


def test_unqualified_secondary_item_still_stops_after_a_primary_change(tmp_path, monkeypatch):
    """Item-4: a secondary-owned item that never had a qualified opening keeps
    the pre-panel safety stop, before any reviewer runs."""
    from test_orchestrator_pr import _rewrite_pr_metadata, _staged_config

    runner = _staged_panel_history(tmp_path, monkeypatch)

    def drop_opening(payload):
        # Demote the secondary-audit decision so history holds no qualified opening.
        if payload.get("scheduler_phase") == "secondary-audit":
            return {**payload, "scheduler_phase": "primary"}
        return None

    _rewrite_pr_metadata(runner, drop_opening)
    changed = _staged_config(tmp_path, primary_reviewer="antigravity", pre_review_tests=False)
    start = len(runner.commands)
    with pytest.raises(orchestrator.PrePanelSafetyError, match="item-1"):
        run_pr_loop(runner, pr_number=77, config=changed)
    assert _agents(runner, start) == []


@pytest.mark.parametrize("path", ["activated", "superseded"])
def test_carried_secondary_finding_survives_successive_primary_changes(
    tmp_path, monkeypatch, path
):
    """Item-11 (row pr-primary-change): a secondary finding published under the
    original qualified opening stays a carried obligation through a second
    primary change -- after the first change activated (later windows hold it
    only in their ledgers), and when the second change supersedes the first
    while it is still pending (both epochs share one round)."""
    from test_orchestrator_pr import _staged_config, _staged_review

    runner = _staged_panel_history(tmp_path, monkeypatch)
    first = _staged_config(tmp_path, primary_reviewer="antigravity", pre_review_tests=False)
    note = [{"item_id": "item-1", "disposition": "resolved",
             "note": "Looks addressed to me; Gemini owns the finding."}]
    if path == "activated":
        # The restarted primary approves; the run stops before the panel answers.
        runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=note))
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=first)
    else:
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=first)
    second = _staged_config(
        tmp_path, primary_reviewer="codex", reviewer_codex_model="gpt-primary-next",
        pre_review_tests=False, max_rounds=8,
    )
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=note))
    runner.gemini_outputs.append(_staged_review(
        reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=note))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=second) == 0
    first_link, second_link = _link_payloads(runner)
    assert second_link["new_board"]["primary"] == "Codex"
    assert second_link["old_board"]["primary"] == "Antigravity"
    if path == "superseded":
        assert second_link["effective_from_round"] == first_link["effective_from_round"]
    else:
        assert second_link["effective_from_round"] > first_link["effective_from_round"]
    # No pre-panel stop: the restarted primary runs first, then Gemini
    # dispositions its own carried finding when the panel reopens.
    agents = _agents(runner, start)
    assert agents[0] == "codex" and "gemini" in agents[1:]
    assert not any("pre-panel safety" in c for c in runner.comments)
    restarted = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == "Codex"
        and r.metadata.reviewer_board_change_digest == board_change_link_digest(second_link)
    ]
    assert restarted and [item.item_id for item in restarted[0].prior_items] == ["item-1"]
    _qualify(runner, second)


@pytest.mark.parametrize("antigravity_state", ["pending", "cleared"])
def test_co_owned_carried_obligation_through_primary_restart_and_panel(
    tmp_path, monkeypatch, antigravity_state
):
    """Item-9.s3 (row pr-primary-change): a pre-epoch item co-owned by two
    secondaries is carried through a same-seat primary model change.  Its
    owners keep their states; each still-pending owner dispositions it when
    the panel reopens, and the run qualifies."""
    from test_orchestrator_pr import _rewrite_pr_metadata, _staged_config, _staged_review

    runner = _staged_panel_history(tmp_path, monkeypatch)

    def co_own(payload):
        changed = False
        for key in ("new_items", "prior_items"):
            items = []
            for item in payload.get(key) or ():
                if item.get("item_id") == "item-1":
                    item = {
                        **item, "resolution_owners": ["Gemini", "Antigravity"],
                        "owner_states": [["Gemini", "pending"], ["Antigravity", antigravity_state]],
                    }
                    changed = True
                items.append(item)
            if items:
                payload[key] = items
        return payload if changed else None

    _rewrite_pr_metadata(runner, co_own)
    changed = _staged_config(
        tmp_path, reviewer_codex_model="gpt-primary-next", pre_review_tests=False, max_rounds=8,
    )
    note = [{"item_id": "item-1", "disposition": "resolved",
             "note": "Looks addressed to me; the panel owners decide."}]
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=note))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", dispositions=resolved))
    runner.antigravity_outputs.append(_staged_review(
        reviewer="Antigravity", dispositions=resolved if antigravity_state == "pending" else note,
    ))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=changed) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["Codex"]
    agents = _agents(runner, start)
    assert agents[0] == "codex" and "gemini" in agents[1:]
    assert not any("pre-panel safety" in c for c in runner.comments)
    restarted = next(
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == "Codex"
        and r.metadata.reviewer_board_change_digest == board_change_link_digest(payload)
    )
    (carried,) = restarted.prior_items
    # The restarted primary saw the obligation with both owners' states intact.
    assert carried.resolution_owners == ("Gemini", "Antigravity")
    assert dict(carried.owner_states) == {"Gemini": "pending", "Antigravity": antigravity_state}
    _qualify(runner, changed)


def test_primary_seat_backend_change_restarts_the_primary_phase(tmp_path, monkeypatch):
    """Item-9.s3 (row pr-primary-change): named primary seat ``p`` moves from
    Codex to Claude under the same seat ID.  The primary phase restarts on the
    rebound seat, the carried secondary finding is exempt from the pre-panel
    stop, its owner dispositions it when the panel reopens, and the run
    qualifies."""
    from test_orchestrator_pr import _audit_phases, _staged_review
    from coding_review_agent_loop.review_scheduling import TransitionClassification

    monkeypatch.setattr(
        orchestrator, "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    p = _seat(tmp_path, "p", "codex", "gpt-p")
    g = _seat(tmp_path, "g", "gemini", "gem-g")
    a = _seat(tmp_path, "a", "antigravity", "Model A")
    staged = {"pr_review_policy": "primary-then-panel", "max_rounds": 8}
    label = {"p": "p (OpenAI Codex: gpt-p)", "g": "g (Google Gemini: gem-g)",
             "a": "a (Google Antigravity: Model A)", "p2": "p (Anthropic Claude: claude-p)"}
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer=label["p"])],
        gemini_outputs=[_staged_review(
            reviewer=label["g"], state="blocking",
            blocking_items=[{"text": "panel gap", "fix_scope": ["src/worker.py"]}],
        )],
        antigravity_outputs=[(_staged_review(reviewer=label["a"]), 0)],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    # The run stops when the next round's scripted reviews run out.
    with pytest.raises((AgentLoopError, AgentInvocationError)):
        run_pr_loop(runner, pr_number=77, config=_named(tmp_path, p, g, a, primary_reviewer=p, **staged))
    moved = _seat(tmp_path, "p", "claude", "claude-p")
    note = [{"item_id": "item-1", "disposition": "resolved",
             "note": "Looks addressed to me; g owns the finding."}]
    # The rebound primary (on Claude) reviews alone first, then g at the reopened panel.
    runner.claude_outputs.append(_staged_review(reviewer=label["p2"], dispositions=note))
    runner.gemini_outputs.append(_staged_review(
        reviewer=label["g"], dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    runner.antigravity_outputs.append((_staged_review(reviewer=label["a"], dispositions=note), 0))
    changed = _named(tmp_path, moved, g, a, primary_reviewer=moved, **staged)
    phases_before = len(_audit_phases(runner))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=changed) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["p"] and payload["new_board"]["primary"] == "p"
    assert [seat["backend"] for seat in payload["new_board"]["seats"] if seat["id"] == "p"] == ["claude"]
    assert _agents(runner, start)[0] == "claude" and "codex" not in _agents(runner, start)
    phases = [phase for phase, _head in _audit_phases(runner)][phases_before:]
    assert phases[0] == "primary" and "secondary-audit" in phases
    assert not any("pre-panel safety" in c for c in runner.comments)
    _qualify(runner, changed)


def test_co_owned_pre_epoch_item_is_carried_while_an_owner_is_pending():
    """Row pr-primary-change: co-owned items follow the carry rule per owner."""
    from coding_review_agent_loop.board_change import carried_secondary_obligations

    def co_owned(gemini_state):
        return _item(
            1, "Gemini", owners=("Gemini", "Antigravity"),
            states=(("Gemini", gemini_state), ("Antigravity", "cleared")),
        )

    assert carried_secondary_obligations(
        [co_owned("pending")], epoch_round=4, primary="Codex", eligible=frozenset({"item-1"}),
    ) == ("item-1",)
    assert carried_secondary_obligations(
        [co_owned("cleared")], epoch_round=4, primary="Codex", eligible=frozenset({"item-1"}),
    ) == ()


# --- prelaunch-only anchors and successive changes (row pr-link-crash-recovery) -------------


def _selective_prelaunch_history():
    """Round 1 reviewed at abc123 (Gemini raised item-1); round 2 holds only its
    scheduler-prelaunch record (the run stopped right after posting it)."""
    from coding_review_agent_loop.review_scheduling import make_contract

    contract = make_contract(("Codex", "Gemini"), "selective-intermediate", None)
    item = _item(1, "Gemini")

    def prelaunch(round_number, prior):
        return _legacy(
            "summary", "Orchestrator", round_number, phase="scheduler-prelaunch",
            prior_items=prior, scheduler_contract=contract.as_dict(),
            scheduler_previous_sha=None, scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16, scheduler_selected_reviewers=("Codex", "Gemini"),
            scheduler_reasons=("full board",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=0,
        )

    return [
        prelaunch(1, ()),
        _legacy("reviewer", "Codex", 1, state="approved"),
        _legacy("reviewer", "Gemini", 1, state="blocking", new_items=(item,)),
        prelaunch(2, (item,)),
    ]


@pytest.mark.parametrize("path", ["recover", "supersede"])
def test_prelaunch_only_anchor_survives_a_crash_and_a_superseding_change(
    tmp_path, monkeypatch, path
):
    """Row pr-link-crash-recovery: a genuinely prelaunch-only round R > 1."""
    runner = _legacy_runner(_selective_prelaunch_history())
    anchor = board_anchor(_records(runner))
    in_round = [r.metadata for r in _records(runner) if r.metadata.round_number == 2]
    assert anchor.round_number == 2 and [m.phase for m in in_round] == ["scheduler-prelaunch"]
    assert not any(m.role in {"coder", "reviewer"} for m in in_round)
    first = make_config(tmp_path, reviewer=("codex", "antigravity"),
                        pr_review_policy="selective-intermediate")
    final = first
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=first)
        if path == "supersede":
            final = make_config(tmp_path, reviewer=("codex", "claude"), coder="codex",
                                pr_review_policy="selective-intermediate")
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=final)
    payloads = _link_payloads(runner)
    assert all((p["anchor_round"], p["advance"], p["effective_from_round"]) == (2, True, 3)
               for p in payloads)
    assert len(payloads) == (2 if path == "supersede" else 1)
    reviews = {
        "antigravity": "antigravity_outputs", "claude": "claude_outputs", "codex": "codex_outputs",
    }
    for backend in ("codex", "antigravity" if path == "recover" else "claude"):
        getattr(runner, reviews[backend]).append(structured_pr_review(
            reviewer={"codex": "OpenAI Codex", "antigravity": "Antigravity",
                      "claude": "Anthropic Claude"}[backend],
            prior_item_dispositions=_RESOLVED_1,
        ))
    assert run_pr_loop(runner, pr_number=77, config=final) == 0
    assert len(_link_payloads(runner)) == len(payloads)
    new = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    # Re-entered round 3 exactly once, with round 2's ledger carried.
    assert {m.round_number for m in new} == {3}
    reviewer = next(m for m in new if m.role == "reviewer")
    assert [item.item_id for item in reviewer.prior_items] == ["item-1"]
    assert {m.reviewer_board_change_digest for m in new} == {
        board_change_link_digest(payloads[-1])
    }
    _qualify(runner, final)


def test_further_change_after_prelaunch_only_activation_takes_effect_next_round(
    tmp_path, monkeypatch
):
    """Item-9.s4 (row pr-link-crash-recovery): the first link is activated at
    round 3 by nothing but its scheduler-prelaunch record (the run stops right
    after posting it); a second board change then takes effect at round 4."""
    runner = _legacy_runner(_selective_prelaunch_history())
    first = make_config(tmp_path, reviewer=("codex", "antigravity"),
                        pr_review_policy="selective-intermediate")
    real_post = orchestrator.post_pr_comment

    def stop_after_prelaunch(*args, **kwargs):
        real_post(*args, **kwargs)
        (record,) = _extract_round_metadata_records(
            [SimpleNamespace(body=str(kwargs["body"]))], flow="pr"
        ) or (None,)
        if record is not None and record.metadata.phase == "scheduler-prelaunch":
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "post_pr_comment", stop_after_prelaunch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=first)
    (link_one,) = _link_payloads(runner)
    assert link_one["effective_from_round"] == 3
    round_three = [r.metadata for r in _records(runner) if r.metadata.round_number == 3]
    assert [m.phase for m in round_three] == ["scheduler-prelaunch"]
    assert round_three[0].reviewer_board_change_digest == board_change_link_digest(link_one)
    second = make_config(tmp_path, reviewer=("codex", "claude"), coder="codex",
                         pr_review_policy="selective-intermediate")
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    runner.claude_outputs.append(structured_pr_review(
        reviewer="Anthropic Claude", prior_item_dispositions=_RESOLVED_1,
    ))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=second) == 0
    _, link_two = _link_payloads(runner)
    assert link_two["predecessor"] == board_change_link_digest(link_one)
    assert (link_two["anchor_round"], link_two["advance"], link_two["effective_from_round"]) == (3, True, 4)
    assert "agy" not in _agents(runner, start)
    new = [r.metadata for r in _records(runner) if r.metadata.round_number >= 4]
    assert new and {m.round_number for m in new} == {4}
    assert {m.reviewer_board_change_digest for m in new} == {board_change_link_digest(link_two)}
    # Round 2's carried ledger survives both changes.
    reviewer = next(m for m in new if m.role == "reviewer")
    assert [item.item_id for item in reviewer.prior_items] == ["item-1"]
    _qualify(runner, second)


def test_empty_anchor_link_recovery_re_enters_the_anchor_round(tmp_path):
    """Item-9.s4 (row pr-link-crash-recovery): a durable link whose anchor round
    holds no record (advance clear) is recovered by the loop at the anchor
    round itself, without a second link, and the run qualifies."""
    root = identity_from_config(make_config(tmp_path, reviewer=("codex", "gemini")))
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    link, digest = _link_body(
        root, root.digest, root, identity_from_config(swapped), anchor=1, advance=False,
    )
    runner = _legacy_runner([link])
    runner.codex_outputs.append(structured_pr_review())
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1
    assert "gemini" not in _agents(runner, start)
    new = _records(runner)
    assert new and {r.metadata.round_number for r in new} == {1}
    assert {r.metadata.reviewer_board_change_digest for r in new} == {digest}
    _qualify(runner, swapped)


def test_further_change_after_coder_only_activation_takes_effect_next_round(tmp_path):
    """Row pr-link-crash-recovery: a link activated by a coder-only round, then another change."""
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    keep_open = [{"item_id": "item-2", "disposition": "blocking",
                  "note": "The retry is still unbounded in src/worker.py; bound it and test it."}]
    runner.codex_outputs.append(structured_pr_review(state="blocking", prior_item_dispositions=keep_open))
    runner.antigravity_outputs.append(structured_pr_review(
        reviewer="Antigravity", state="blocking", prior_item_dispositions=keep_open,
    ))
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-2"]))
    # Round 3 reviews, the coder answers in round 4, and round 4's reviews run out.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=swapped)
    (first,) = _link_payloads(runner)
    round_four = [r.metadata for r in _records(runner) if r.metadata.round_number == 4]
    assert round_four and {m.role for m in round_four} <= {"coder", "summary"}
    assert all(m.reviewer_board_change_digest == board_change_link_digest(first) for m in round_four)
    again = make_config(tmp_path, reviewer=("codex", "gemini"))
    resolved = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved))
    runner.gemini_outputs.append(
        structured_pr_review(reviewer="Google Gemini", prior_item_dispositions=resolved)
    )
    assert run_pr_loop(runner, pr_number=77, config=again) == 0
    _, second = _link_payloads(runner)
    assert second["predecessor"] == board_change_link_digest(first)
    assert (second["anchor_round"], second["advance"], second["effective_from_round"]) == (4, True, 5)
    _qualify(runner, again)


# --- pending signed removal after a rebinding or seat-adding link (row pr-floor-candidate-link)


_LABELS = {"codex": "OpenAI Codex", "gemini": "Google Gemini", "antigravity": "Google Antigravity"}
_OUTPUTS = {"codex": "codex_outputs", "gemini": "gemini_outputs", "antigravity": "antigravity_outputs"}


def _queue_reviews(runner, seats):
    for seat in seats:
        review = structured_pr_review(
            reviewer=f"{seat} ({_LABELS[seat.backend]}: {seat.model_chain[0]})"
        )
        getattr(runner, _OUTPUTS[seat.backend]).append(
            (review, 0) if seat.backend == "antigravity" else review
        )


@pytest.mark.parametrize("link", ["backend-rebinding", "seat-adding"])
def test_pending_signed_removal_after_a_link_is_authorized_from_the_link_board(tmp_path, link):
    """Row pr-floor-candidate-link, end to end: preflight refusal, signed
    activation, resume and qualification all use the frozen authorization
    taken from the predecessor link's board."""
    a = _seat(tmp_path, "a", "codex", "gpt-a")
    b = _seat(tmp_path, "b", "gemini", "gem-b")
    c = _seat(tmp_path, "c", "antigravity", "Model C")
    selective = {"pr_review_policy": "selective-intermediate"}
    runner = FakeRunner()
    _queue_reviews(runner, (a, b, c))
    assert run_pr_loop(runner, pr_number=77, config=_named(tmp_path, a, b, c, **selective)) == 0
    if link == "backend-rebinding":
        changed = (a, _seat(tmp_path, "b", "codex", "gpt-b"), c)
        floor = {"min_distinct_providers": 2}
        refusal = "1 distinct provider\\(s\\) is below the floor of 2"
    else:
        changed = (a, b, c, _seat(tmp_path, "d", "codex", "gpt-d"))
        floor = {"min_reviewers": 4}
        refusal = "3 reviewer\\(s\\) is below the floor of 4"
    runner.pr_payload["headRefOid"] = "def456"
    _queue_reviews(runner, changed)
    assert run_pr_loop(runner, pr_number=77, config=_named(tmp_path, *changed, **selective, **floor)) == 0
    (first,) = _link_payloads(runner)
    reduced_seats = tuple(seat for seat in changed if str(seat) != "c")
    reduced = _named(tmp_path, *reduced_seats, **selective, **floor)
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match=refusal) as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    # Refused at preflight: no record, link or agent.
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner, start) == []
    template = refused.value.args[0][refused.value.args[0].index("Reviewer board amendment:"):]
    assert '"c"' in template.split('"removed_reviewers"')[1].split("]")[0]
    _append(runner, template.replace("<why the removed reviewer cannot be reached>", "c is out of quota."))
    runner.pr_payload["headRefOid"] = "fed789"
    _queue_reviews(runner, reduced_seats)
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert len(_link_payloads(runner)) == 1
    assert any("Reviewer board amendment applied." in c for c in runner.comments)
    assert "agy" not in _agents(runner, start)
    lineage = _board_state(runner, reduced).lineage
    assert [element.kind for element in lineage.elements] == ["link", "amendment"]
    (authorization,) = lineage.authorizations
    assert [seat.seat_id for seat in authorization.seats] == [str(seat) for seat in reduced_seats]
    _qualify(runner, reduced)
    # A resume with the same board derives the same lineage and posts nothing.
    # (A named Gemini seat is re-asked on every rerun, as on main.)
    _queue_reviews(runner, [seat for seat in reduced_seats if seat.backend == "gemini"])
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert len(_link_payloads(runner)) == 1
    assert first == _link_payloads(runner)[0]


_LABELS["claude"] = "Anthropic Claude"
_OUTPUTS["claude"] = "claude_outputs"


@pytest.mark.parametrize("floor", ["reviewers", "providers"])
@pytest.mark.parametrize("flags", ["reduced", "original"])
def test_named_legacy_all_reviewers_signed_reduction(tmp_path, floor, flags):
    """Rows pr-floor-unsigned-vs-signed and pr-legacy-root-anchor: a legacy
    named all-reviewers PR (recorded seat binding, no scheduler contract)."""
    sol = _seat(tmp_path, "sol", "codex", "gpt-sol")
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    opus = _seat(tmp_path, "opus", "claude", "claude-opus")
    board = (sol, flash, opus)
    runner = FakeRunner()
    _queue_reviews(runner, board)
    assert run_pr_loop(runner, pr_number=77, config=_named(tmp_path, *board)) == 0
    _strip_board_identity(runner)
    assert all(r.metadata.pr_board_root is None and r.metadata.seat_binding for r in _records(runner))
    floor_kwargs = {"min_reviewers": 3} if floor == "reviewers" else {"min_distinct_providers": 3}
    kept = (sol, opus)
    reduced = _named(tmp_path, *kept, **floor_kwargs)
    original = _named(tmp_path, *board, **floor_kwargs)
    runner.pr_payload["headRefOid"] = "def456"
    comments = len(runner.pr_payload["comments"])
    before = len(runner.commands)
    with pytest.raises(AgentLoopError, match="below the floor of 3") as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner, before) == []
    _append(runner, _amendment_from(str(refused.value)))
    start = len(runner.commands)
    _queue_reviews(runner, kept)
    assert run_pr_loop(runner, pr_number=77, config=reduced if flags == "reduced" else original) == 0
    assert "agy" not in _agents(runner, start) and _links(runner) == []
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded and {
        BoardIdentity.from_payload(m.pr_board_root).names for m in upgraded
    } == {("sol", "flash", "opus")}
    _qualify(runner, reduced)


def test_signed_first_upgrade_initializes_bindings_through_a_crash_then_links_a_model_change(
    tmp_path, monkeypatch
):
    """Row pr-legacy-root-anchor: a plain signed reduction under reduced flags is
    the first upgrade; the crash after its first record keeps the initialization,
    and a later model change is exactly one link needing fresh approval."""
    runner = _legacy_runner([
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="approved"),
        _legacy("coder", "Claude", 2),
    ])
    _append(runner, _signed_body(("Codex", "Gemini"), removed=("Gemini",), round_number=2))
    reduced = make_config(tmp_path, reviewer=("codex",), min_reviewers=2)
    real_post = orchestrator.post_pr_comment

    def crash_after_first_record(*args, **kwargs):
        real_post(*args, **kwargs)
        if "AGENT_LOOP_META" in str(kwargs["body"]):
            raise KeyboardInterrupt

    runner.codex_outputs.append(structured_pr_review())
    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "post_pr_comment", crash_after_first_record)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=reduced)
    (first,) = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert first.reviewer_board_amendment_digest is not None
    # Gemini (signed off) stays unknown; Codex is initialized by this record.
    assert set(first.pr_board_binding_init) == {"Codex"}
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert _links(runner) == []
    moved = dataclasses.replace(reduced, reviewer_codex_model="gpt-later")
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=moved) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["Codex"] and payload["binding_init"] == {}
    assert _agents(runner, start) == ["codex"]
    _qualify(runner, moved)


# =====================================================================================
# Recorded plan board changes (#1377): the same lineage on the issue owning a plan.
# =====================================================================================

from agent_loop_helpers import structured_plan_review, structured_v1_plan_state  # noqa: E402
from coding_review_agent_loop.board_change import (  # noqa: E402
    PLAN_BOARD_CHANGE_LINK_HEADING,
    PLAN_BOARD_CHANGE_LINK_MARKER,
    PLAN_SURFACE,
)
from coding_review_agent_loop.cli import run_issue_loop  # noqa: E402

_PLAN_ACTOR = ("agent-loop", 4242)
_PLAN_AGENTS = {"claude", "codex", "gemini", "agy"}


def _plan_runner(**outputs):
    return FakeRunner(
        authenticated_actor=_PLAN_ACTOR, serve_rest_issue_comments=True, **outputs
    )


def _plan_cfg(tmp_path, reviewers, **overrides):
    values = dict(
        reviewer=tuple(reviewers), max_rounds=6, plan_execution_mode="plan-only",
        agent_max_retries=0,
    )
    values.update(overrides)
    return make_config(tmp_path, **values)


def _plan_comments(runner):
    return [
        SimpleNamespace(body=c["body"], author_id=c.get("_rest_author_id"))
        for c in runner.issue_comments
    ]


def _plan_records(runner):
    return _extract_round_metadata_records(_plan_comments(runner), flow="plan")


def _plan_links(runner):
    return [
        (index, c["body"]) for index, c in enumerate(runner.issue_comments)
        if c["body"].startswith(PLAN_BOARD_CHANGE_LINK_HEADING)
    ]


def _plan_link_payloads(runner):
    pattern = PLAN_SURFACE.link_re
    return [
        json.loads(base64.urlsafe_b64decode(pattern.search(body).group("payload")))
        for _index, body in _plan_links(runner)
    ]


def _plan_agents(runner, start=0):
    return [cmd[0] for cmd, _cwd in runner.commands[start:] if cmd and cmd[0] in _PLAN_AGENTS]


def _gemini_blocks_round_one(tmp_path, reviewers=("codex", "gemini")):
    """Round 1: Codex approves, Gemini raises item-1; the planner revision never starts."""
    reviews = {
        "codex": ("codex_outputs", structured_plan_review(state="approved")),
        "gemini": ("gemini_outputs", structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="Missing owner.",
            blocking_plan_issues=["Name the rollout owner."],
        )),
        "antigravity": ("antigravity_outputs", structured_plan_review(
            state="approved", reviewer="Google Antigravity",
        )),
    }
    outputs = {"claude_outputs": [structured_v1_plan_state()]}
    for name in reviewers:
        key, review = reviews[name]
        outputs[key] = [review]
    runner = _plan_runner(**outputs)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56, config=_plan_cfg(tmp_path, reviewers), plan_first=True,
        )
    return runner


def _append_issue_comment(runner, body, *, login="operator", author_id=81):
    """An operator comment created now, after every comment the run posted."""
    import datetime

    runner.issue_comments.append({
        "body": body, "author": {"login": login}, "_rest_author_id": author_id,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def _review(name, state="approved", dispositions=_RESOLVED_1, **extra):
    signature = {
        "codex": "OpenAI Codex", "gemini": "Google Gemini", "antigravity": "Google Antigravity",
    }[name]
    return structured_plan_review(
        state=state, reviewer=signature, prior_plan_item_dispositions=dispositions, **extra,
    )


def test_plan_midrun_swap_posts_one_link_and_requires_the_new_reviewer(tmp_path):
    """Rows plan-board-change-midrun / plan-all-reviewers-board-change: Gemini -> Antigravity."""
    runner = _gemini_blocks_round_one(tmp_path)
    before_comments = len(runner.issue_comments)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    start = len(runner.commands)
    swapped = _plan_cfg(tmp_path, ("codex", "antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0

    (payload,) = _plan_link_payloads(runner)
    assert payload["flow"] == "plan" and payload["issue_number"] == 56
    assert "pr_number" not in payload
    assert payload["added"] == ["Antigravity"] and payload["removed"] == ["Gemini"]
    assert payload["reason"] == "operator-reconfigured"
    assert payload["anchor_round"] == 1 and payload["advance"] is True
    assert payload["effective_from_round"] == 2
    ((link_index, _body),) = _plan_links(runner)
    # The link is the first comment of the rerun: posted before any agent turn.
    assert link_index == before_comments
    # Round 2 re-enters reviewer-only on the byte-identical plan: no planner turn.
    assert _plan_agents(runner, start) == ["codex", "agy"]
    records = _plan_records(runner)
    round_two = [r for r in records if r.metadata.round_number == 2]
    assert {r.metadata.role for r in round_two} == {"reviewer"}
    assert {r.metadata.subject for r in records} == {records[0].metadata.subject}
    # The removed reviewer's open finding was carried, never dropped.
    assert all(
        [item.item_id for item in r.metadata.prior_items] == ["item-1"] for r in round_two
    )
    assert all(
        r.metadata.reviewer_board_change_digest == board_change_link_digest(payload)
        and r.metadata.plan_board_root is not None
        for r in round_two
    )
    # Idempotent resume: the approved plan is never reopened and no link repeats.
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0
    assert _plan_agents(runner, start) == []
    assert len(_plan_links(runner)) == 1


def test_plan_swapped_out_reviewer_approval_never_counts(tmp_path):
    """Row plan-board-change-midrun: the new seat must approve; the old one's approval is gone."""
    runner = _gemini_blocks_round_one(tmp_path)
    # Swap Codex (which approved round 1) for Antigravity, which then blocks.
    runner.gemini_outputs.append(_review("gemini"))
    runner.antigravity_outputs.append(_review(
        "antigravity", state="blocking", dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
        ],
        summary="Still unclear.", blocking_plan_issues=["Clarify the rollback."],
    ))
    swapped = _plan_cfg(tmp_path, ("antigravity", "gemini"))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True)
    (payload,) = _plan_link_payloads(runner)
    assert payload["removed"] == ["Codex"] and payload["added"] == ["Antigravity"]
    # Codex's round-1 approval did not approve the plan: a planner revision followed.
    assert not any("plan approved" in c["body"] for c in runner.issue_comments)
    assert _plan_agents(runner)[-1] == "claude"


def test_plan_removed_reviewer_finding_stays_open_and_blocks_approval(tmp_path):
    """Row plan-all-reviewers-board-change: removing the finding's owner reassigns it."""
    runner = _gemini_blocks_round_one(tmp_path, reviewers=("codex", "gemini", "antigravity"))
    # Gemini (owner of item-1) leaves; its finding stays open for the remaining
    # board, which keeps it blocking, so the plan is not approved.
    still_open = [{"item_id": "item-1", "disposition": "blocking"}]
    runner.codex_outputs.append(_review(
        "codex", state="blocking", dispositions=still_open, summary="Owner still missing.",
    ))
    runner.antigravity_outputs.append(_review(
        "antigravity", state="blocking", dispositions=still_open, summary="Owner still missing.",
    ))
    reduced = _plan_cfg(tmp_path, ("codex", "antigravity"))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=reduced, plan_first=True)
    (payload,) = _plan_link_payloads(runner)
    assert payload["removed"] == ["Gemini"] and payload["added"] == []
    assert not any("plan approved" in c["body"] for c in runner.issue_comments)
    round_two = [r for r in _plan_records(runner) if r.metadata.round_number == 2]
    assert round_two and all(
        [item.item_id for item in r.metadata.prior_items] == ["item-1"] for r in round_two
    )
    # The next planner revision is asked to address the carried item.
    assert _plan_agents(runner)[-1] == "claude"


def test_plan_added_reviewer_is_required_from_the_effective_round(tmp_path):
    """Row plan-board-change-midrun: an added reviewer reviews the current plan."""
    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.gemini_outputs.append(_review("gemini"))
    runner.antigravity_outputs.append(_review("antigravity"))
    widened = _plan_cfg(tmp_path, ("codex", "gemini", "antigravity"))
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=widened, plan_first=True) == 0
    (payload,) = _plan_link_payloads(runner)
    assert payload["added"] == ["Antigravity"] and payload["removed"] == []
    assert sorted(_plan_agents(runner, start)) == ["agy", "codex", "gemini"]


def test_plan_named_model_rebind_is_a_rebound_seat_link(tmp_path):
    """Row plan-board-change-midrun: a same-ID seat on another model is a swap."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        antigravity_outputs=[structured_plan_review(
            state="blocking", reviewer="flash (Google Antigravity: Model A)", summary="x",
            blocking_plan_issues=["Name the owner."],
        )],
    )
    config = _plan_cfg(tmp_path, ("codex", flash), reviewer_seats=(flash,))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    rebound = _seat(tmp_path, "flash", "antigravity", "Model B")
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="flash (Google Antigravity: Model B)",
        prior_plan_item_dispositions=_RESOLVED_1,
    ))
    rebound_config = _plan_cfg(tmp_path, ("codex", rebound), reviewer_seats=(rebound,))
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=rebound_config, plan_first=True) == 0
    (payload,) = _plan_link_payloads(runner)
    assert payload["rebound"] == ["flash"] and payload["added"] == [] == payload["removed"]
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands[start:]
              if cmd[0] == "agy" and "--model" in cmd]
    assert models == ["Model B"]


def test_plan_plain_to_named_and_back_each_post_one_link(tmp_path):
    """Row plan-all-reviewers-board-change: plain-to-named, then named-to-plain."""
    runner = _gemini_blocks_round_one(tmp_path)
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    runner.codex_outputs.append(_review("codex", state="blocking", summary="More.",
                                        blocking_plan_issues=["Add rollback steps."]))
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="flash (Google Antigravity: Model A)",
        prior_plan_item_dispositions=_RESOLVED_1,
    ))
    named = _plan_cfg(tmp_path, ("codex", flash), reviewer_seats=(flash,))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=named, plan_first=True)
    first = _plan_link_payloads(runner)
    assert len(first) == 1 and first[0]["new_seat_binding"] is not None
    assert first[0]["old_seat_binding"] is None
    # Back to the plain board while the planner revision is still pending: a
    # second link at the re-entry round; nothing is posted twice.
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "gemini")),
                       plan_first=True)
    payloads = _plan_link_payloads(runner)
    assert len(payloads) == 2
    assert payloads[1]["new_seat_binding"] is None
    assert payloads[1]["predecessor"] == board_change_link_digest(payloads[0])


def test_plan_link_crash_before_effective_round_is_recovered_once(tmp_path, monkeypatch):
    """Row plan-link-crash-resume: the link is recovered, never reposted."""
    import coding_review_agent_loop.plan_first_loop as plan_first_loop

    runner = _gemini_blocks_round_one(tmp_path)
    swapped = _plan_cfg(tmp_path, ("codex", "antigravity"))
    real = plan_first_loop.set_plan_board_stamp

    def crash(stamp):
        if stamp is not None:
            raise AgentLoopError("simulated crash after the link post")
        return real(stamp)

    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", crash)
    with pytest.raises(AgentLoopError, match="simulated crash"):
        run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True)
    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", real)
    assert len(_plan_links(runner)) == 1
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0
    assert len(_plan_links(runner)) == 1
    rounds = sorted({r.metadata.round_number for r in _plan_records(runner)})
    assert rounds == [1, 2]


def test_plan_further_change_while_a_link_is_pending_supersedes_it(tmp_path, monkeypatch):
    """Row plan-link-crash-resume: a further differing board adds exactly one link."""
    import coding_review_agent_loop.plan_first_loop as plan_first_loop

    runner = _gemini_blocks_round_one(tmp_path)
    real = plan_first_loop.set_plan_board_stamp

    def crash(stamp):
        if stamp is not None:
            raise AgentLoopError("simulated crash after the link post")
        return real(stamp)

    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", crash)
    with pytest.raises(AgentLoopError, match="simulated crash"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
                       plan_first=True)
    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", real)
    runner.codex_outputs.append(_review("codex"))
    runner.gemini_outputs.append(_review("gemini"))
    runner.antigravity_outputs.append(_review("antigravity"))
    widened = _plan_cfg(tmp_path, ("codex", "gemini", "antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=widened, plan_first=True) == 0
    first, second = _plan_link_payloads(runner)
    assert second["predecessor"] == board_change_link_digest(first)
    assert (second["anchor_round"], second["advance"]) == (first["anchor_round"], first["advance"])
    assert second["effective_from_round"] == 2


def _plan_link_body(runner, **overrides):
    payload = dict(_plan_link_payloads(runner)[0])
    payload.update(overrides)
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    return f"{PLAN_BOARD_CHANGE_LINK_HEADING}\n\n<!-- {PLAN_BOARD_CHANGE_LINK_MARKER}: {encoded} -->"


@pytest.mark.parametrize("defect", ["foreign-author", "malformed", "wrong-issue", "unknown-digest"])
def test_forged_or_malformed_plan_link_fails_closed(tmp_path, defect):
    """Row plan-link-forged: resume refuses before any agent invocation or post."""
    runner = _gemini_blocks_round_one(tmp_path)
    swapped = _plan_cfg(tmp_path, ("codex", "antigravity"))
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0
    link = next(c for c in runner.issue_comments if c["body"].startswith(PLAN_BOARD_CHANGE_LINK_HEADING))
    if defect == "foreign-author":
        link["_rest_author_id"] = 99
        link["author"] = {"login": "someone-else"}
    elif defect == "malformed":
        link["body"] = f"{PLAN_BOARD_CHANGE_LINK_HEADING}\n<!-- {PLAN_BOARD_CHANGE_LINK_MARKER}: e30= -->"
    elif defect == "wrong-issue":
        link["body"] = _plan_link_body(runner, issue_number=57)
    else:
        _rewrite_last_plan_record(runner, reviewer_board_change_digest="f" * 64)
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="lineage is inconsistent"):
        run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True)
    assert _plan_agents(runner, start) == []
    assert len(runner.issue_comments) == comments


def _planned_round_one(tmp_path, reviewers=("codex", "gemini")):
    """Round 1 holds only the published plan: the first reviewer turn never ran."""
    runner = _plan_runner(claude_outputs=[structured_v1_plan_state()])
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56,
            config=_plan_cfg(tmp_path, reviewers, review_parallel=False), plan_first=True,
        )
    return runner


def test_plan_floor_shrink_refused_unsigned_and_allowed_signed(tmp_path):
    """Rows plan-floor-shrink / plan-all-reviewers-board-change (signed floor case)."""
    runner = _planned_round_one(tmp_path)
    one = _plan_cfg(tmp_path, ("codex",), min_reviewers=2)
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="below the floor of 2") as refusal:
        run_issue_loop(runner, issue_number=56, config=one, plan_first=True)
    # The refusal prints the filled-in signed plan amendment template and posts nothing.
    assert '"kind": "reviewer-board-amendment"' in str(refusal.value)
    assert '"flow": "plan"' in str(refusal.value)
    assert _plan_agents(runner, start) == [] and len(runner.issue_comments) == comments
    assert _plan_links(runner) == []

    signed = format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    )
    _append_issue_comment(runner, signed)
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    # Explicitly reduced flags are accepted under the signed reduction.
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=one, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["codex"]
    assert _plan_links(runner) == []


def test_plan_signed_reduction_with_original_flags_rebinds_to_the_signed_board(tmp_path):
    """Row plan-all-reviewers-board-change: original flags are rebound (#1133) under the floor."""
    runner = _planned_round_one(tmp_path)
    signed = format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    )
    _append_issue_comment(runner, signed)
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    start = len(runner.commands)
    original = _plan_cfg(tmp_path, ("codex", "gemini"), min_reviewers=2, min_distinct_providers=2)
    assert run_issue_loop(runner, issue_number=56, config=original, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["codex"]
    assert _plan_links(runner) == []


def test_plan_provider_floor_refuses_an_unsigned_same_provider_board(tmp_path):
    """Row plan-all-reviewers-board-change: the distinct-provider floor on a plan link."""
    runner = _gemini_blocks_round_one(tmp_path)
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    opus = _seat(tmp_path, "opus", "antigravity", "Model B")
    same_provider = _plan_cfg(
        tmp_path, (flash, opus), reviewer_seats=(flash, opus), min_distinct_providers=2,
    )
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="distinct provider"):
        run_issue_loop(runner, issue_number=56, config=same_provider, plan_first=True)
    assert _plan_agents(runner, start) == [] and len(runner.issue_comments) == comments


def _rewrite_last_plan_record(runner, **changes):
    """Rewrite the newest plan round record carrying a board-change digest."""
    import re

    from coding_review_agent_loop.round_state import _decode_round_metadata

    for comment in reversed(runner.issue_comments):
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow != "plan" or metadata.reviewer_board_change_digest is None:
            continue
        body = comment["body"][: match.start()].rstrip("\n")
        comment["body"] = _attach_round_metadata(body, dataclasses.replace(metadata, **changes))
        return
    raise AssertionError("no stamped plan record")


def _strip_plan_stamps(runner):
    """Rewrite every plan record as pre-upgrade (unstamped) history."""
    from coding_review_agent_loop.round_state import _decode_round_metadata

    import re

    for comment in runner.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow != "plan":
            continue
        stripped = dataclasses.replace(
            metadata, plan_board_root=None, plan_board_identity=None,
            plan_board_binding_init=None, reviewer_board_change_digest=None,
            reviewer_board_amendment_digest=None,
        )
        body = comment["body"][: match.start()].rstrip("\n")
        comment["body"] = _attach_round_metadata(body, stripped)


def test_plan_legacy_plain_history_upgrades_without_any_record(tmp_path):
    """Row plan-legacy-regression: unchanged board on unstamped history posts nothing."""
    runner = _gemini_blocks_round_one(tmp_path)
    _strip_plan_stamps(runner)
    assert all(r.metadata.plan_board_root is None for r in _plan_records(runner))
    runner.claude_outputs.append(structured_plan_review())  # unused guard
    runner.claude_outputs.clear()
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "gemini")),
                       plan_first=True)
    # Exactly today's continuation: the planner revision, no link, no new record.
    assert _plan_agents(runner, start) == ["claude"]
    assert _plan_links(runner) == []
    assert len(runner.issue_comments) == comments


def test_plan_legacy_plain_history_migrates_its_root_from_posters(tmp_path):
    """Row plan-all-reviewers-board-change: legacy root = configured board plus posters."""
    runner = _gemini_blocks_round_one(tmp_path)
    _strip_plan_stamps(runner)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(
        runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
        plan_first=True,
    ) == 0
    (payload,) = _plan_link_payloads(runner)
    root = BoardIdentity.from_payload(payload["root"])
    # The configured board plus every historical plan reviewer poster, read-only.
    assert root.names == ("Codex", "Antigravity", "Gemini")
    assert root.policy == "all-reviewers" and root.primary is None
    assert payload["removed"] == ["Gemini"] and payload["added"] == []
    # Persisted legacy rounds are never rewritten.
    assert all(
        r.metadata.plan_board_root is None
        for r in _plan_records(runner) if r.metadata.round_number == 1
    )


def test_plan_policy_change_stays_drift_including_legacy_history(tmp_path):
    """Row plan-policy-drift: legacy plain history roots at all-reviewers, no primary."""
    runner = _gemini_blocks_round_one(tmp_path)
    for strip in (False, True):
        if strip:
            _strip_plan_stamps(runner)
        staged = _plan_cfg(
            tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer="codex",
        )
        start, comments = len(runner.commands), len(runner.issue_comments)
        with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
            run_issue_loop(runner, issue_number=56, config=staged, plan_first=True)
        assert _plan_agents(runner, start) == []
        assert len(runner.issue_comments) == comments
        assert _plan_links(runner) == []


def test_plan_legacy_partial_approval_is_undetermined_and_resumes_as_today(tmp_path):
    """Row plan-legacy-partial-approval: posters alone never prove completion."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"), review_parallel=False)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    _strip_plan_stamps(runner)
    # Only Codex approved; Gemini never posted.
    assert {r.metadata.agent for r in _plan_records(runner) if r.metadata.role == "reviewer"} == {
        "Codex"
    }
    classification = classify_plan_approval(
        _plan_comments(runner), config=config, issue_number=56, human_requirements=(),
    )
    assert classification.state == "undetermined"
    runner.gemini_outputs.append(structured_plan_review(state="approved", reviewer="Google Gemini"))
    start = len(runner.commands)
    floor = _plan_cfg(tmp_path, ("codex", "gemini"), min_reviewers=2, review_parallel=False)
    assert run_issue_loop(runner, issue_number=56, config=floor, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["gemini"]
    assert _plan_links(runner) == []


def test_approved_plan_is_never_reopened_by_a_different_board(tmp_path):
    """Row plan-approved-board-change: approved by B, resumed with C and below the floor."""
    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    board_b = _plan_cfg(tmp_path, ("codex", "antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=board_b, plan_first=True) == 0
    links = len(_plan_links(runner))
    for board_c in (
        _plan_cfg(tmp_path, ("gemini", "antigravity")),
        _plan_cfg(tmp_path, ("gemini", "codex"), plan_review_policy="primary-then-panel",
                  primary_plan_reviewer="gemini", min_reviewers=2),
    ):
        start, comments = len(runner.commands), len(runner.issue_comments)
        assert run_issue_loop(runner, issue_number=56, config=board_c, plan_first=True) == 0
        assert _plan_agents(runner, start) == []
        assert len(_plan_links(runner)) == links
        assert len(runner.issue_comments) == comments
    # A below-floor invocation board meets only the stage 1 handoff floor of
    # the approved plan, never a planning candidate floor or a plan link.
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="Issue #56 reviewer board") as refusal:
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("gemini",), min_reviewers=2),
                       plan_first=True)
    assert "plan reviewer board" not in str(refusal.value)
    assert _plan_agents(runner, start) == [] and len(runner.issue_comments) == comments


def test_approval_classification_is_independent_of_the_invocation_board(tmp_path):
    """Row plan-approved-board-change (unit parity): the approving identity governs."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(
        runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
        plan_first=True,
    ) == 0
    comments = _plan_comments(runner)
    results = set()
    for invocation in (
        _plan_cfg(tmp_path, ("codex", "antigravity")),
        _plan_cfg(tmp_path, ("gemini",)),
        _plan_cfg(tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
                  primary_plan_reviewer="gemini"),
    ):
        classification = classify_plan_approval(
            comments, config=invocation, issue_number=56, human_requirements=(),
            trusted_actor=_PLAN_ACTOR,
        )
        results.add((classification.state, classification.context.effective_reviewers))
    assert results == {("approved", ("Codex", "Antigravity"))}


def test_unanimous_approvals_with_an_open_carried_obligation_are_not_approved():
    """Row plan-approval-gate-open-obligation (unit): a reassigned item blocks completion."""
    from coding_review_agent_loop.plan_verification import plan_open_obligations
    from coding_review_agent_loop.round_state import PostedRoundRecord, ResumedReviewRound

    gemini_item = UnresolvedReviewItem(
        item_id="item-1", reviewer="Gemini", source_round=1, text="Name the owner.",
        status="blocking",
    )

    def approval(index, agent, dispositions=()):
        return PostedRoundRecord(
            index=index, body="",
            metadata=PostedRoundMetadata(
                flow="plan", role="reviewer", agent=agent, round_number=2, subject="s",
                state="approved", prior_items=(gemini_item,), dispositions=tuple(dispositions),
            ),
        )

    unanimous = ResumedReviewRound(
        round_number=2, prior_items=(gemini_item,), coder_output=None,
        completed_reviews=(approval(1, "Codex"), approval(2, "Antigravity")),
        next_unresolved_item_number=2,
    )
    # Every remaining reviewer approved, but the removed reviewer's finding was
    # never dispositioned: it is reassigned to the remaining board and stays open.
    assert plan_open_obligations(unanimous, board=("Codex", "Antigravity")) == ("item-1",)
    from coding_review_agent_loop.protocol import ReviewItemDisposition

    resolved = ResumedReviewRound(
        round_number=2, prior_items=(gemini_item,), coder_output=None,
        completed_reviews=(
            approval(1, "Codex", [ReviewItemDisposition(item_id="item-1", disposition="resolved",
                                                        reviewer="Codex")]),
            approval(2, "Antigravity"),
        ),
        next_unresolved_item_number=2,
    )
    assert plan_open_obligations(resolved, board=("Codex", "Antigravity")) == ()


def test_requirement_evidence_is_recorded_under_all_reviewers_and_binds_approval(tmp_path):
    """Row plan-approval-requirement-evidence: recorded keys bind; a change unbinds."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    reviews = [r for r in _plan_records(runner) if r.metadata.role == "reviewer"]
    # Generation-1 reviewer records carry the exact-plan key as evidence only.
    assert all(r.metadata.plan_candidate_key is not None for r in reviews)
    assert all(r.metadata.scheduler_contract is None for r in _plan_records(runner))
    comments = _plan_comments(runner)
    unchanged = classify_plan_approval(
        comments, config=config, issue_number=56, human_requirements=(),
        trusted_actor=_PLAN_ACTOR,
    )
    assert unchanged.state == "approved"
    from coding_review_agent_loop.github import HumanReviewRequirement

    added = HumanReviewRequirement(
        source_type="issue_comment", author="operator", created_at="2026-05-23T01:00:00Z",
        url=None, body="Keep the rollout reversible.",
    )
    # A signed requirement added after the approvals: the recorded key no longer
    # matches, so the approval cannot be shortcut.
    changed = classify_plan_approval(
        comments, config=config, issue_number=56, human_requirements=(added,),
        trusted_actor=_PLAN_ACTOR,
    )
    assert changed.state in {"unfinished", "undetermined"}


def test_plan_verification_resolves_the_lineage_and_judges_the_approving_board(tmp_path):
    """Row plan-lineage-to-pr-verification (unit): A -> B approved by B verifies against B."""
    from coding_review_agent_loop.plan_verification import derive_plan_verification_context

    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(
        runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
        plan_first=True,
    ) == 0
    context = derive_plan_verification_context(
        _plan_comments(runner), issue_number=56, trusted_actor=_PLAN_ACTOR,
    )
    assert context.derivation == "lineage"
    assert context.effective_reviewers == ("Codex", "Antigravity")
    # Gemini's round-1 record is not an approving-board approval; nothing else is stale.
    with pytest.raises(AgentLoopError, match="not authored by the authenticated orchestrator"):
        derive_plan_verification_context(
            _plan_comments(runner), issue_number=56, trusted_actor=("intruder", 7),
        )


def test_plan_primary_change_restarts_the_primary_phase_and_carries_the_panel_item(tmp_path):
    """Row plan-primary-change-midrun: Codex -> Antigravity as primary-then-panel primary."""
    def staged(primary, **extra):
        return _plan_cfg(
            tmp_path, ("codex", "gemini", "antigravity"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer=primary, **extra,
        )

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="x",
            blocking_plan_issues=["Name the owner."],
        )],
        antigravity_outputs=[structured_plan_review(state="approved", reviewer="Google Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    records = _plan_records(runner)
    gemini_items = [
        item.item_id for r in records if r.metadata.agent == "Gemini" for item in r.metadata.new_items
    ]
    assert gemini_items == ["item-1"]
    anchor = max(r.metadata.round_number for r in records)
    # The new primary approves the unchanged plan; the panel then needs Codex and
    # Gemini, whose outputs are not scripted, so the run stops there.
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="Google Antigravity", prior_plan_item_dispositions=_RESOLVED_1,
    ))
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56, config=staged("antigravity", plan_primary_stall_rounds=1),
            plan_first=True,
        )
    (payload,) = _plan_link_payloads(runner)
    assert payload["old_board"]["primary"] == "Codex"
    assert payload["new_board"]["primary"] == "Antigravity"
    assert payload["effective_from_round"] == anchor + 1
    # The next round is primary-only on the new primary: no pre-panel safety
    # stop for Gemini's carried item, and the old primary's opening is not reused.
    assert _plan_agents(runner, start)[0] == "agy"
    prelaunch = [
        r.metadata for r in _plan_records(runner)
        if r.metadata.phase == "scheduler-prelaunch" and r.metadata.round_number == anchor + 1
    ]
    assert prelaunch and prelaunch[0].scheduler_phase == "primary"
    assert prelaunch[0].scheduler_selected_reviewers == ("Antigravity",)
    assert prelaunch[0].scheduler_primary_reviewer == "Antigravity"
    assert [item.item_id for item in prelaunch[0].prior_items] == ["item-1"]
    assert not any("Plan review scheduling diagnostic" in c["body"] for c in runner.issue_comments)
