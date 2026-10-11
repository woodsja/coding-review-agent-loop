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
from coding_review_agent_loop.errors import AgentLoopError
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


@pytest.mark.parametrize("corruption", ["foreign-author", "malformed", "broken-predecessor", "wrong-old-board"])
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
    else:
        wrong = identity_from_config(make_config(tmp_path, reviewer=("gemini",))).payload()
        _append(runner, _forged_link_body(runner, smaller, old_board=wrong), author_id=_ACTOR_ID)
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


def _link_body(root, predecessor, old, new, *, anchor, advance, inits=None):
    payload = board_change_link_payload(
        pr_number=77, root=root, predecessor=predecessor, old_board=old, new_board=new,
        old_seat_binding=old.seat_binding(), binding_init=inits or {},
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


def test_primary_change_restarts_the_primary_phase_and_carries_secondary_obligations(
    tmp_path, monkeypatch
):
    """Row pr-primary-change: the panel's open item survives the restarted primary phase."""
    from test_orchestrator_pr import _audit_phases, _staged_config, _staged_review
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
    # Codex (primary) approves, the panel opens, Gemini raises item-1, the
    # coder answers, and the next round runs out of scripted output.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pre_review_tests=False))
    assert _agents(runner)[:3] == ["codex", "gemini", "agy"]
    changed = _staged_config(tmp_path, primary_reviewer="antigravity", pre_review_tests=False)
    # A non-owner's disposition is a note under owner-scoped reconciliation:
    # item-1 stays pending on its owner, Gemini.
    carried = [{"item_id": "item-1", "disposition": "resolved",
                "note": "Looks addressed to me; Gemini owns the finding."}]
    # Round 4: the new primary alone; round 5: the reopened panel at the same
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
    assert (payload["old_board"]["primary"], payload["new_board"]["primary"]) == ("Codex", "Antigravity")
    # The new primary runs alone first -- no pre-panel stop for the carried
    # secondary obligation -- then the panel reopens and Gemini dispositions it.
    agents = _agents(runner, start)
    assert agents[0] == "agy" and "gemini" in agents[1:] and "claude" not in agents
    phases = [phase for phase, _head in _audit_phases(runner)][phases_before:]
    assert phases[0] == "primary" and "secondary-audit" in phases
    restarted = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == "Antigravity"
        and r.metadata.reviewer_board_change_digest == board_change_link_digest(payload)
    ]
    # item-1 was still pending on Gemini when the new primary reviewed.
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
                         anchor=2, advance=True)
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
