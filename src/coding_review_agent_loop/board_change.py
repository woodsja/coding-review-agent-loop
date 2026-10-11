"""Recorded in-run PR reviewer-board changes (#1376, stage 2 of #1373).

A PR run's reviewer board is a per-invocation choice.  When the configured
board differs from the durable PR board, the orchestrator records an
orchestrator-authored *board-change link* on the PR and continues; no human
signature is needed for additions, swaps, or model changes.  The guarantee is
kept by the lineage, not by a frozen roster: every reviewer on the board in
force approves the exact artifact, and every change is visible and auditable.

The lineage has three kinds of element, ordered by comment position:

* the *root*: the board identity C0 of the run.  Once any identity-bearing
  round record or link exists, the root is the identity those artifacts carry,
  and it never changes.  Before that, it is derived read-only from legacy
  history (see :func:`derive_legacy_root`).
* *board-change links*: digest-chained records whose payload names the root,
  the predecessor element, the old and new boards, and the round the change
  takes effect from.  Only comments authored by the authenticated orchestrator
  actor are admitted.
* *signed amendments* (#943/#984): unchanged parser and digests.  A signed
  amendment after a link amends the link's new board, and a signed
  restoration can only re-add a seat a signed amendment removed since the
  latest link.

Every PR round record is governed by exactly one element (the latest whose
comment precedes it and whose effective round is at or before its round) and
must carry that element's digest; identity-bearing records also carry the
root and the governing identity.  Records that do not match fail closed at
startup and at qualification.

Floor authorizations come only from signed segments, frozen from the board
identity immediately before the segment; a link never creates or enlarges an
exception.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .errors import AgentLoopError
from .reviewer_floor import BoardSeat, SignedBoardAuthorization, distinct_providers

BOARD_CHANGE_LINK_MARKER = "AGENT_PR_BOARD_CHANGE"
BOARD_CHANGE_LINK_KIND = "reviewer-board-change"
BOARD_CHANGE_LINK_SCHEMA_VERSION = 2
BOARD_CHANGE_LINK_REASON = "operator-reconfigured"
BOARD_CHANGE_LINK_HEADING = "Reviewer board change recorded."
BOARD_CHANGE_LINK_RE = re.compile(
    rf"<!--\s*{BOARD_CHANGE_LINK_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I
)
_LINK_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "flow",
        "pr_number",
        "root",
        "predecessor",
        "old_board",
        "new_board",
        "old_seat_binding",
        "new_seat_binding",
        "added",
        "removed",
        "rebound",
        "binding_init",
        "anchor_round",
        "advance",
        "effective_from_round",
        "reason",
    }
)
_IDENTITY_KEYS = frozenset({"policy", "primary", "broad_rules", "seats"})
_SEAT_KEYS = frozenset({"id", "backend", "model_chain", "effort"})
_INIT_KEYS = frozenset({"model_chain", "effort"})
_BACKENDS = frozenset({"claude", "codex", "gemini", "antigravity"})
_POLICIES = frozenset({"all-reviewers", "selective-intermediate", "primary-then-panel"})
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _fail(detail: str) -> AgentLoopError:
    return AgentLoopError(
        f"PR reviewer-board lineage is inconsistent: {detail}. Rerun is refused until the "
        "history is corrected; no agent was invoked and no comment was posted."
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _plain_backends() -> dict[str, str]:
    from .agents.registry import agent_display_name

    return {agent_display_name(backend): backend for backend in sorted(_BACKENDS)}


def is_named_seat(seat: BoardSeat) -> bool:
    """A plain reviewer's seat ID is the display name of its backend."""
    return _plain_backends().get(seat.seat_id) != seat.backend


# --- Board identity -----------------------------------------------------------


def seat_payload(seat: BoardSeat) -> dict[str, object]:
    return {
        "id": seat.seat_id,
        "backend": seat.backend,
        "model_chain": list(seat.model_chain) if seat.model_chain is not None else None,
        "effort": seat.effort,
    }


def _seat_from_payload(entry: object) -> BoardSeat:
    if not isinstance(entry, dict) or set(entry) != _SEAT_KEYS:
        raise _fail("a board seat has missing or unknown fields")
    seat_id = entry["id"]
    backend = entry["backend"]
    chain = entry["model_chain"]
    effort = entry["effort"]
    if not isinstance(seat_id, str) or not seat_id.strip() or backend not in _BACKENDS:
        raise _fail("a board seat has an invalid identity or backend")
    if chain is not None:
        if not isinstance(chain, list) or not chain or any(
            model is not None and (not isinstance(model, str) or not model.strip())
            for model in chain
        ):
            raise _fail(f"board seat {seat_id!r} has an invalid model chain")
        chain = tuple(chain)
    if effort is not None and not isinstance(effort, str):
        raise _fail(f"board seat {seat_id!r} has an invalid effort")
    return BoardSeat(seat_id=seat_id, backend=backend, model_chain=chain, effort=effort)


@dataclass(frozen=True)
class BoardIdentity:
    """The canonical PR board identity: policy, primary, broad rules, and seats."""

    policy: str
    primary: str | None
    broad_rules: tuple[str, ...]
    seats: tuple[BoardSeat, ...]

    def __post_init__(self) -> None:
        names = [seat.seat_id for seat in self.seats]
        if not names or len(set(names)) != len(names):
            raise _fail("a board identity needs a unique, non-empty seat list")
        if self.policy not in _POLICIES:
            raise _fail(f"unknown review policy {self.policy!r}")
        if self.primary is not None and self.primary not in names:
            raise _fail(f"primary {self.primary!r} is not on its board")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(seat.seat_id for seat in self.seats)

    def seat(self, seat_id: str) -> BoardSeat | None:
        return next((seat for seat in self.seats if seat.seat_id == seat_id), None)

    def payload(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "primary": self.primary,
            "broad_rules": list(self.broad_rules),
            "seats": [seat_payload(seat) for seat in self.seats],
        }

    @property
    def digest(self) -> str:
        return _digest(self.payload())

    def label(self) -> str:
        """Seat IDs, with the primary when there is one, for log lines."""
        return ", ".join(self.names) + (f" (primary {self.primary})" if self.primary else "")

    @property
    def named(self) -> bool:
        return any(is_named_seat(seat) for seat in self.seats)

    def contract(self) -> object:
        from .review_scheduling import make_contract

        try:
            return make_contract(self.names, self.policy, self.broad_rules, self.primary)
        except AgentLoopError as exc:
            raise _fail(f"board {list(self.names)} is not a valid {self.policy} board: {exc}") from exc

    def seat_binding(self) -> dict[str, object] | None:
        """The seat binding PR round records carry under this board.

        Named boards persist a versioned binding (``reviewer_seat_binding``);
        plain boards persist none.
        """
        if not self.named:
            return None
        return {"version": 1, "seats": [seat_payload(seat) for seat in self.seats]}

    def with_seats(self, seats: Sequence[BoardSeat]) -> "BoardIdentity":
        primary = self.primary if self.primary in {seat.seat_id for seat in seats} else None
        return dataclasses.replace(self, seats=tuple(seats), primary=primary)

    @classmethod
    def from_payload(cls, payload: object) -> "BoardIdentity":
        if not isinstance(payload, dict) or set(payload) != _IDENTITY_KEYS:
            raise _fail("a board identity has missing or unknown fields")
        rules = payload["broad_rules"]
        if not isinstance(rules, list) or any(not isinstance(rule, str) for rule in rules):
            raise _fail("a board identity has invalid broad rules")
        seats = payload["seats"]
        if not isinstance(seats, list):
            raise _fail("a board identity has an invalid seat list")
        primary = payload["primary"]
        if primary is not None and not isinstance(primary, str):
            raise _fail("a board identity has an invalid primary")
        policy = payload["policy"]
        if not isinstance(policy, str):
            raise _fail("a board identity has an invalid policy")
        return cls(
            policy=policy,
            primary=primary,
            broad_rules=tuple(rules),
            seats=tuple(_seat_from_payload(entry) for entry in seats),
        )


def identity_from_config(config: object) -> BoardIdentity:
    """The configured PR board identity, with every seat's current binding."""
    from .agents.registry import agent_display_name
    from .review_scheduling import normalize_broad_rules
    from .reviewer_floor import board_seats_from_config

    primary = getattr(config, "primary_reviewer", None)
    return BoardIdentity(
        policy=getattr(config, "pr_review_policy", "all-reviewers"),
        primary=agent_display_name(primary) if primary is not None else None,
        broad_rules=normalize_broad_rules(getattr(config, "pr_review_broad_rules", None)),
        seats=board_seats_from_config(config),
    )


def identity_agents(identity: BoardIdentity) -> tuple[object, ...]:
    """Reviewer agents for a board identity, keyed by the same display names."""
    from pathlib import Path

    from .reviewer_seats import ReviewerSeat, SeatAgent

    agents: list[object] = []
    for seat in identity.seats:
        if not is_named_seat(seat):
            agents.append(seat.backend)
        else:
            agents.append(
                SeatAgent(
                    ReviewerSeat(seat.seat_id, seat.backend, tuple(seat.model_chain or ()), seat.effort),
                    Path("."),
                )
            )
    return tuple(agents)


def name_agents(names: Sequence[str]) -> tuple[object, ...]:
    """Reviewer keys for display names alone, for history readers that match by name."""
    from pathlib import Path

    from .reviewer_seats import ReviewerSeat, SeatAgent

    plain = _plain_backends()
    return tuple(
        plain[name] if name in plain
        else SeatAgent(ReviewerSeat(name, "codex", (), None), Path("."))
        for name in names
    )


def board_reentry_round(comments: Sequence[object], resumed_round: int | None) -> int:
    """The round a PR resume re-enters (#1376).

    The resume reader matches the latest round's records by the configured
    board.  When a recorded board change has left that round with no record by
    a current seat, the reader finds nothing; with link history the run then
    re-enters after the highest recorded round instead of restarting at round 1.
    """
    if resumed_round is not None:
        return resumed_round
    if comments_bear_board_change_links(comments):
        from .round_state import _extract_round_metadata_records

        records = _extract_round_metadata_records(comments, flow="pr")
        if records:
            return board_anchor(records).round_number + 1
    return 1


def seats_compatible(old: BoardSeat, new: BoardSeat) -> bool:
    """Same seat and backend; models and effort compared only where both are known."""
    if old.seat_id != new.seat_id or old.backend != new.backend:
        return False
    if old.model_chain is None or new.model_chain is None:
        return True
    return old.model_chain == new.model_chain and old.effort == new.effort


def binding_compatible(a: BoardIdentity, b: BoardIdentity) -> bool:
    """Seat IDs, order, primary, policy, rules and backends match; known bindings match."""
    return (
        a.policy == b.policy
        and a.primary == b.primary
        and a.broad_rules == b.broad_rules
        and a.names == b.names
        and all(seats_compatible(x, y) for x, y in zip(a.seats, b.seats))
    )


def apply_inits(identity: BoardIdentity, inits: Mapping[str, BoardSeat]) -> BoardIdentity:
    """The initialization-aware view: unknown seats take their durable initialization."""
    seats = []
    changed = False
    for seat in identity.seats:
        init = inits.get(seat.seat_id)
        if seat.model_chain is None and init is not None and init.backend == seat.backend:
            seats.append(init)
            changed = True
        else:
            seats.append(seat)
    return dataclasses.replace(identity, seats=tuple(seats)) if changed else identity


def pending_initializations(
    view: BoardIdentity, configured: BoardIdentity
) -> dict[str, BoardSeat]:
    """Configured bindings for every seat whose binding is still unknown in ``view``."""
    result: dict[str, BoardSeat] = {}
    for seat in view.seats:
        if seat.model_chain is not None:
            continue
        target = configured.seat(seat.seat_id)
        if target is not None and target.backend == seat.backend and target.model_chain is not None:
            result[seat.seat_id] = target
    return result


def init_payload(inits: Mapping[str, BoardSeat]) -> dict[str, object]:
    return {
        seat_id: {"model_chain": list(seat.model_chain or ()), "effort": seat.effort}
        for seat_id, seat in sorted(inits.items())
    }


def _inits_from_payload(payload: object, *, board: BoardIdentity) -> dict[str, BoardSeat]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise _fail("a binding initialization is not an object")
    result: dict[str, BoardSeat] = {}
    for seat_id, entry in payload.items():
        seat = board.seat(seat_id)
        if seat is None:
            raise _fail(f"binding initialization names {seat_id!r}, which is not on its board")
        if not isinstance(entry, dict) or set(entry) != _INIT_KEYS:
            raise _fail(f"binding initialization of {seat_id!r} is malformed")
        parsed = _seat_from_payload(
            {"id": seat_id, "backend": seat.backend, **entry}
        )
        if parsed.model_chain is None:
            raise _fail(f"binding initialization of {seat_id!r} has no model chain")
        result[seat_id] = parsed
    return result


# --- Board-change link records ------------------------------------------------


@dataclass(frozen=True)
class BoardChangeLink:
    """One admitted orchestrator-authored board-change link."""

    payload: Mapping[str, object]
    digest: str
    comment_index: int
    root: BoardIdentity
    predecessor: str
    old_board: BoardIdentity
    new_board: BoardIdentity
    anchor_round: int
    advance: bool
    effective_from_round: int

    @property
    def added(self) -> tuple[str, ...]:
        return tuple(self.payload["added"])  # type: ignore[arg-type]

    @property
    def removed(self) -> tuple[str, ...]:
        return tuple(self.payload["removed"])  # type: ignore[arg-type]

    @property
    def rebound(self) -> tuple[str, ...]:
        return tuple(self.payload["rebound"])  # type: ignore[arg-type]

    @property
    def binding_init(self) -> dict[str, BoardSeat]:
        return _inits_from_payload(self.payload["binding_init"], board=self.new_board)


def describe_board_change(
    old: BoardIdentity, new: BoardIdentity
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """(added, removed, rebound) seat IDs between two board identities."""
    old_by_id = {seat.seat_id: seat for seat in old.seats}
    new_by_id = {seat.seat_id: seat for seat in new.seats}
    added = tuple(name for name in new.names if name not in old_by_id)
    removed = tuple(name for name in old.names if name not in new_by_id)
    rebound = tuple(
        name
        for name in new.names
        if name in old_by_id and not seats_compatible(old_by_id[name], new_by_id[name])
    )
    return added, removed, rebound


def board_change_link_payload(
    *,
    pr_number: int,
    root: BoardIdentity,
    predecessor: str,
    old_board: BoardIdentity,
    new_board: BoardIdentity,
    old_seat_binding: Mapping[str, object] | None,
    binding_init: Mapping[str, BoardSeat],
    anchor_round: int,
    advance: bool,
) -> dict[str, object]:
    added, removed, rebound = describe_board_change(old_board, new_board)
    return {
        "kind": BOARD_CHANGE_LINK_KIND,
        "schema_version": BOARD_CHANGE_LINK_SCHEMA_VERSION,
        "flow": "pr",
        "pr_number": pr_number,
        "root": root.payload(),
        "predecessor": predecessor,
        "old_board": old_board.payload(),
        "new_board": new_board.payload(),
        "old_seat_binding": dict(old_seat_binding) if old_seat_binding is not None else None,
        "new_seat_binding": new_board.seat_binding(),
        "added": list(added),
        "removed": list(removed),
        "rebound": list(rebound),
        "binding_init": init_payload(binding_init),
        "anchor_round": anchor_round,
        "advance": advance,
        "effective_from_round": anchor_round + (1 if advance else 0),
        "reason": BOARD_CHANGE_LINK_REASON,
    }


def board_change_link_digest(payload: Mapping[str, object]) -> str:
    return _digest(dict(payload))


def _encode_payload(payload: Mapping[str, object]) -> str:
    raw = json.dumps(dict(payload), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _describe_seat(seat: BoardSeat) -> str:
    if seat.model_chain is None:
        return f"{seat.seat_id} ({seat.backend})"
    models = " -> ".join(model or "default model" for model in seat.model_chain)
    effort = f", effort {seat.effort}" if seat.effort else ""
    return f"{seat.seat_id} ({seat.backend}: {models}{effort})"


def render_board_change_link(payload: Mapping[str, object]) -> object:
    """The orchestrator-authored link comment: audit text plus the reserved record."""
    from .protocol_markers import TrustedBody

    digest = board_change_link_digest(payload)
    old = BoardIdentity.from_payload(payload["old_board"])
    new = BoardIdentity.from_payload(payload["new_board"])
    rebound = []
    for name in payload["rebound"]:  # type: ignore[union-attr]
        before, after = old.seat(name), new.seat(name)
        if before is not None and after is not None:
            rebound.append(f"{_describe_seat(before)} -> {_describe_seat(after)}")
    lines = [
        BOARD_CHANGE_LINK_HEADING,
        "",
        f"- Record digest: `{digest}`",
        f"- Reason: {BOARD_CHANGE_LINK_REASON}",
        f"- Effective from round: {payload['effective_from_round']}",
        "- Previous board: " + "; ".join(_describe_seat(seat) for seat in old.seats),
        "- New board: " + "; ".join(_describe_seat(seat) for seat in new.seats),
        f"- Added reviewer(s): {', '.join(payload['added']) or 'none'}",  # type: ignore[arg-type]
        f"- Removed reviewer(s): {', '.join(payload['removed']) or 'none'}",  # type: ignore[arg-type]
        f"- Rebound reviewer(s): {'; '.join(rebound) or 'none'}",
    ]
    if old.primary != new.primary or new.policy == "primary-then-panel":
        lines.append(f"- Primary reviewer: {old.primary or '(none)'} -> {new.primary or '(none)'}")
    lines += [
        "- Added and rebound reviewers must approve the current head; their earlier "
        "approvals do not count. Removed reviewers' approvals no longer count, and their "
        "unresolved findings stay open, reassigned to the remaining board.",
        "",
        f"<!-- {BOARD_CHANGE_LINK_MARKER}: {_encode_payload(payload)} -->",
        "",
        "-- Orchestrator",
    ]
    return TrustedBody.canonical("\n".join(lines), expected_tokens=(BOARD_CHANGE_LINK_MARKER,))


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _name_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and len(set(value)) == len(value)
    )


def parse_board_change_link(
    payload: object, *, comment_index: int, pr_number: int
) -> BoardChangeLink:
    locator = f"PR #{pr_number} comment {comment_index + 1}"
    if not isinstance(payload, dict) or set(payload) != _LINK_KEYS:
        raise _fail(f"board-change record at {locator} has missing or unknown fields")
    if payload["kind"] != BOARD_CHANGE_LINK_KIND or payload["schema_version"] != (
        BOARD_CHANGE_LINK_SCHEMA_VERSION
    ) or isinstance(payload["schema_version"], bool):
        raise _fail(f"board-change record at {locator} has an unsupported kind or version")
    if payload["flow"] != "pr" or payload["pr_number"] != pr_number:
        raise _fail(f"board-change record at {locator} names another surface")
    if payload["reason"] != BOARD_CHANGE_LINK_REASON:
        raise _fail(f"board-change record at {locator} has an unknown reason")
    if not isinstance(payload["predecessor"], str) or not _HEX64.match(payload["predecessor"]):
        raise _fail(f"board-change record at {locator} has an invalid predecessor digest")
    anchor = payload["anchor_round"]
    advance = payload["advance"]
    effective = payload["effective_from_round"]
    if (
        not _is_positive_int(anchor)
        or not isinstance(advance, bool)
        or not _is_positive_int(effective)
        or effective != anchor + (1 if advance else 0)
    ):
        raise _fail(f"board-change record at {locator} has inconsistent rounds")
    for key in ("added", "removed", "rebound"):
        if not _name_list(payload[key]):
            raise _fail(f"board-change record at {locator} has an invalid {key} list")
    root = BoardIdentity.from_payload(payload["root"])
    old = BoardIdentity.from_payload(payload["old_board"])
    new = BoardIdentity.from_payload(payload["new_board"])
    added, removed, rebound = describe_board_change(old, new)
    if [list(added), list(removed), list(rebound)] != [
        payload["added"], payload["removed"], payload["rebound"],
    ]:
        raise _fail(f"board-change record at {locator} misdescribes its change")
    if payload["new_seat_binding"] != new.seat_binding():
        raise _fail(f"board-change record at {locator} has an inconsistent seat binding")
    if payload["old_seat_binding"] is not None and not isinstance(payload["old_seat_binding"], dict):
        raise _fail(f"board-change record at {locator} has an invalid previous seat binding")
    new.contract()
    _inits_from_payload(payload["binding_init"], board=new)
    return BoardChangeLink(
        payload=payload,
        digest=board_change_link_digest(payload),
        comment_index=comment_index,
        root=root,
        predecessor=payload["predecessor"],
        old_board=old,
        new_board=new,
        anchor_round=anchor,
        advance=advance,
        effective_from_round=effective,
    )


def comments_bear_board_change_links(comments: Sequence[object]) -> bool:
    return any(
        isinstance(body := _body(comment), str) and BOARD_CHANGE_LINK_RE.search(body)
        for comment in comments
    )


def _body(comment: object) -> object:
    if isinstance(comment, dict):
        return comment.get("body")
    return getattr(comment, "body", None)


def collect_board_change_links(
    comments: Sequence[object],
    *,
    pr_number: int,
    trusted_actor: tuple[str, int] | None,
) -> tuple[BoardChangeLink, ...]:
    """Admit board-change links authored by the authenticated orchestrator actor.

    A marker-bearing comment by any other author, a comment whose author
    cannot be authenticated, or a malformed record fails closed.  Identical
    records collapse by digest, keeping the earliest.
    """
    links: dict[str, BoardChangeLink] = {}
    for index, comment in enumerate(comments):
        body = _body(comment)
        if not isinstance(body, str):
            continue
        matches = list(BOARD_CHANGE_LINK_RE.finditer(body))
        if not matches:
            continue
        locator = f"PR #{pr_number} comment {index + 1}"
        if len(matches) > 1:
            raise _fail(f"{locator} carries more than one board-change record")
        author_id = (
            comment.get("author_id") if isinstance(comment, dict)
            else getattr(comment, "author_id", None)
        )
        if trusted_actor is None or author_id is None or author_id != trusted_actor[1]:
            raise _fail(
                f"the board-change record at {locator} was not authored by the authenticated "
                "orchestrator actor; only the orchestrator's record path can change the board"
            )
        try:
            raw = json.loads(
                base64.urlsafe_b64decode(matches[0].group("payload").encode("ascii")).decode("utf-8")
            )
        except (ValueError, UnicodeDecodeError) as exc:
            raise _fail(f"the board-change record at {locator} cannot be decoded") from exc
        link = parse_board_change_link(raw, comment_index=index, pr_number=pr_number)
        links.setdefault(link.digest, link)
    return tuple(sorted(links.values(), key=lambda link: link.comment_index))


# --- Lineage --------------------------------------------------------------------


@dataclass(frozen=True)
class LineageElement:
    """One element of the PR board lineage after the root."""

    kind: str  # "link" | "amendment"
    comment_index: int
    effective_from_round: int
    digest: str
    identity: BoardIdentity
    contract: object
    segment_base: BoardIdentity
    link: BoardChangeLink | None = None
    amendment: object | None = None


@dataclass(frozen=True)
class BoardLineage:
    """The validated PR board lineage: root, links and signed amendments."""

    root: BoardIdentity
    anchored: bool
    elements: tuple[LineageElement, ...]
    carried: frozenset[str]
    superseded: frozenset[str]
    pending_links: tuple[LineageElement, ...]
    pending_amendments: tuple[object, ...]
    inits: Mapping[str, BoardSeat]
    authorizations: tuple[SignedBoardAuthorization, ...]
    # Seat IDs every PR round record has published as a reviewer.
    posters: tuple[str, ...] = ()
    last_seen: Mapping[str, BoardSeat] = field(default_factory=dict)

    # -- ContractLineage-compatible surface ---------------------------------

    @property
    def contracts(self) -> tuple[object, ...]:
        return (self.root.contract(), *(element.contract for element in self.elements))

    @property
    def amendments(self) -> tuple[object, ...]:
        return tuple(
            element.amendment for element in self.elements if element.kind == "amendment"
        )

    @property
    def latest(self) -> LineageElement | None:
        return self.elements[-1] if self.elements else None

    @property
    def active_amendment(self) -> object | None:
        latest = self.latest
        return latest.amendment if latest is not None and latest.kind == "amendment" else None

    @property
    def active_digest(self) -> str | None:
        active = self.active_amendment
        return getattr(active, "digest", None)

    @property
    def active_change_digest(self) -> str | None:
        latest = self.latest
        return latest.digest if latest is not None and latest.kind == "link" else None

    @property
    def current_identity(self) -> BoardIdentity:
        base = self.latest.identity if self.latest is not None else self.root
        return apply_inits(base, self.inits)

    @property
    def current_contract(self) -> object:
        return self.current_identity.contract()

    @property
    def current_board(self) -> tuple[str, ...]:
        return self.current_identity.names

    @property
    def segment_base(self) -> BoardIdentity:
        return self.latest.segment_base if self.latest is not None else self.root

    @property
    def segment_base_board(self) -> tuple[str, ...]:
        """The board a signed amendment of the current segment amends from."""
        return self.segment_base.names

    @property
    def removed_reviewers(self) -> tuple[str, ...]:
        """Seats that left the board (signed amendment or link) and are still off it."""
        board = set(self.current_board)
        removed: list[str] = []
        previous = self.root.names
        for element in self.elements:
            for name in previous:
                if name not in element.identity.names and name not in board:
                    removed.append(name)
            previous = element.identity.names
        # Historical root members a link never carried forward are removed too.
        return tuple(dict.fromkeys(removed))

    @property
    def restoration_rounds(self) -> dict[str, int]:
        """Currently required seats restored by a signed amendment -> its round (#984)."""
        board = set(self.current_board)
        rounds: dict[str, int] = {}
        for element in self.elements:
            if element.kind != "amendment":
                continue
            for name in getattr(element.amendment, "restored_reviewers", ()):
                if name in board:
                    rounds[name] = element.effective_from_round
        return rounds

    @property
    def restored_reviewers(self) -> tuple[str, ...]:
        return tuple(self.restoration_rounds)

    @property
    def fresh_from_rounds(self) -> dict[str, int]:
        """Seat -> round from which its approvals count again.

        A seat added, rebound (backend, model or effort), restored, or whose
        primary/secondary role changed must approve from that round on; its
        earlier approvals never count (#1376).
        """
        board = set(self.current_board)
        rounds: dict[str, int] = {}
        for element in self.elements:
            if element.kind == "amendment":
                for name in getattr(element.amendment, "restored_reviewers", ()):
                    rounds[name] = element.effective_from_round
                continue
            link = element.link
            assert link is not None
            changed = set(link.added) | set(link.rebound)
            if link.new_board.policy == "primary-then-panel" and (
                link.old_board.primary != link.new_board.primary
                or link.new_board.primary in link.rebound
            ):
                changed |= {
                    name for name in (link.old_board.primary, link.new_board.primary) if name
                }
            for name in changed:
                rounds[name] = element.effective_from_round
        return {name: value for name, value in rounds.items() if name in board}

    @property
    def primary_epoch_round(self) -> int | None:
        """Effective round of the latest link that restarted the primary phase."""
        epoch = None
        for element in self.elements:
            link = element.link
            if link is None or link.new_board.policy != "primary-then-panel":
                continue
            if (
                link.old_board.primary != link.new_board.primary
                or link.new_board.primary in link.rebound
            ):
                epoch = element.effective_from_round
        return epoch

    def contract_before(self, amendment: object) -> object:
        for position, element in enumerate(self.elements):
            if element.amendment is amendment:
                return self.contracts[position]
        raise AgentLoopError("Unknown reviewer-board amendment.")

    def governing(self, *, record_index: int, round_number: int) -> LineageElement | None:
        governing = None
        for element in self.elements:
            if (
                element.comment_index < record_index
                and element.effective_from_round <= round_number
            ):
                governing = element
        return governing

    def binding_for_round(self, round_number: int) -> dict[str, object] | None:
        """Seat binding of the lineage element governing ``round_number``.

        Signed segments keep their segment base's binding (the #1133 rebind
        retains the configured binding), so only links change it.
        """
        base = self.root
        for element in self.elements:
            if element.kind == "link" and element.effective_from_round <= round_number:
                base = element.segment_base
        return apply_inits(base, self.inits).seat_binding()

    def describe(self) -> str:
        return "; ".join(_describe_seat(seat) for seat in self.current_identity.seats)


def lineage_floor_authorizations(lineage: BoardLineage) -> tuple[SignedBoardAuthorization, ...]:
    """The only source of floor authorizations: frozen signed segments."""
    return lineage.authorizations


def _segment_authorization(
    pre: BoardIdentity, post: BoardIdentity
) -> SignedBoardAuthorization:
    return SignedBoardAuthorization(
        seats=post.seats,
        lowered_reviewers=len(post.seats) < len(pre.seats),
        lowered_providers=distinct_providers(post.seats) < distinct_providers(pre.seats),
    )


def derive_legacy_root(
    records: Sequence[object],
    *,
    configured: BoardIdentity,
    amendments: Sequence[object],
    contract_from_metadata: Callable[[object], object | None],
) -> BoardIdentity:
    """Derive the root of history that carries no root anchor yet (D1a).

    Scheduler-contract history: the earliest persisted contract, with the
    latest legacy seat binding when one exists.  Named all-reviewers history:
    the recorded seat binding, which names the whole board.  Plain
    all-reviewers history records no board, so its membership is never
    inferred from round completeness: the earliest signed amendment's
    original board when signed amendments exist, otherwise the configured
    board plus every historical reviewer poster.  No reviewer history means a
    fresh run whose root is the configured board.
    """
    plain = _plain_backends()
    contract = next(
        (
            contract
            for record in sorted(records, key=lambda item: item.index)
            if record.metadata.scheduler_metadata_status != "invalid"
            and (contract := contract_from_metadata(record.metadata)) is not None
        ),
        None,
    )
    bindings = [
        record.metadata.seat_binding
        for record in sorted(records, key=lambda item: item.index)
        if record.metadata.seat_binding is not None
    ]
    binding_entries: dict[str, dict] = {}
    if bindings:
        from .reviewer_seats import validated_binding_entries

        try:
            binding_entries = validated_binding_entries(bindings[-1])
        except AgentLoopError as exc:
            raise _fail(f"the recorded seat binding is malformed ({exc})") from exc

    def seat_for(name: str) -> BoardSeat:
        entry = binding_entries.get(name)
        if entry is not None:
            return BoardSeat(
                seat_id=name,
                backend=entry["backend"],
                model_chain=tuple(entry["model_chain"]),
                effort=entry.get("effort"),
            )
        if name in plain and not binding_entries:
            return BoardSeat(seat_id=name, backend=plain[name])
        configured_seat = configured.seat(name)
        if configured_seat is not None and not binding_entries and not is_named_seat(configured_seat):
            return BoardSeat(seat_id=name, backend=configured_seat.backend)
        raise _fail(f"historical reviewer {name!r} has no verifiable backend binding")

    posters = [
        record.metadata.agent
        for record in sorted(records, key=lambda item: item.index)
        if record.metadata.role == "reviewer" and record.metadata.agent
    ]
    posters = list(dict.fromkeys(posters))
    if contract is not None:
        return BoardIdentity(
            policy=contract.policy,
            primary=contract.primary_reviewer,
            broad_rules=tuple(contract.broad_rules),
            seats=tuple(seat_for(name) for name in contract.required_reviewers),
        )
    if binding_entries:
        return BoardIdentity(
            policy=configured.policy,
            primary=configured.primary if configured.primary in binding_entries else None,
            broad_rules=configured.broad_rules,
            seats=tuple(seat_for(name) for name in binding_entries),
        )
    if not posters:
        return configured
    if amendments:
        original = tuple(getattr(amendments[0], "original_required_reviewers"))
        outside = [name for name in posters if name not in original]
        if outside:
            raise _fail(
                f"historical reviewer(s) {outside} are not on the earliest signed amendment's "
                f"original board {list(original)}"
            )
        names = original
    else:
        names = tuple(dict.fromkeys((*configured.names, *posters)))
    primary = configured.primary if configured.primary in names else None
    return BoardIdentity(
        policy=configured.policy,
        primary=primary,
        broad_rules=configured.broad_rules,
        seats=tuple(seat_for(name) for name in names),
    )


def _contract_identity(contract: object, previous: BoardIdentity, last_seen: Mapping[str, BoardSeat]) -> BoardIdentity:
    seats = []
    for name in getattr(contract, "required_reviewers"):
        seat = previous.seat(name) or last_seen.get(name)
        if seat is None:
            raise _fail(f"signed amendment restores {name!r}, whose binding is unknown")
        seats.append(seat)
    return BoardIdentity(
        policy=getattr(contract, "policy"),
        primary=getattr(contract, "primary_reviewer"),
        broad_rules=tuple(getattr(contract, "broad_rules")),
        seats=tuple(seats),
    )


def resolve_board_lineage(
    records: Sequence[object],
    comments: Sequence[object],
    *,
    amendments: Sequence[object],
    configured: BoardIdentity,
    pr_number: int,
    trusted_actor: tuple[str, int] | None,
    contract_from_metadata: Callable[[object], object | None],
    drift_error: Callable[[object, str], AgentLoopError],
) -> BoardLineage:
    """Resolve and validate the complete PR board lineage, read-only (D3).

    The configured board is used only to derive a legacy root.  Comparing the
    configured board with the effective board (a board-change candidate, or
    policy drift) is the caller's job.
    """
    from .board_amendment import build_amendment_chain
    from .reviewer_seats import validate_backend_outage_amendments_for_binding

    ordered = sorted(records, key=lambda record: record.index)
    links = collect_board_change_links(comments, pr_number=pr_number, trusted_actor=trusted_actor)
    identity_records = [r for r in ordered if r.metadata.pr_board_root is not None]
    for record in ordered:
        metadata = record.metadata
        if metadata.pr_board_root is None and (
            metadata.pr_board_identity is not None
            or metadata.pr_board_binding_init is not None
            or metadata.reviewer_board_change_digest is not None
        ):
            raise _fail(
                f"round record at comment {record.index + 1} carries board identity fields "
                "without a root anchor"
            )
    anchors: list[tuple[int, BoardIdentity]] = [
        (record.index, BoardIdentity.from_payload(record.metadata.pr_board_root))
        for record in identity_records
    ] + [(link.comment_index, link.root) for link in links]
    anchors.sort(key=lambda item: item[0])
    if anchors:
        root = anchors[0][1]
        for index, candidate in anchors[1:]:
            if candidate != root:
                raise _fail(
                    f"the artifact at comment {index + 1} carries a different root board; "
                    "the root anchor never changes once persisted"
                )
        anchored = True
    else:
        root = derive_legacy_root(
            ordered,
            configured=configured,
            amendments=amendments,
            contract_from_metadata=contract_from_metadata,
        )
        anchored = False
    root_contract = root.contract()
    posters = tuple(
        dict.fromkeys(
            record.metadata.agent
            for record in ordered
            if record.metadata.role == "reviewer" and record.metadata.agent
        )
    )

    # Pass 1: elements in comment order.  Consecutive signed amendments form a
    # segment chained from the segment base with the existing chain builder.
    raw_elements = sorted(
        [("link", link.comment_index, link) for link in links]
        + [("amendment", amendment.comment_index, amendment) for amendment in amendments],
        key=lambda item: item[1],
    )
    elements: list[LineageElement] = []
    authorizations: list[SignedBoardAuthorization] = []
    superseded: set[str] = set()
    last_seen: dict[str, BoardSeat] = {seat.seat_id: seat for seat in root.seats}
    current = root
    current_contract = root_contract
    segment_base = root
    position = 0
    while position < len(raw_elements):
        kind, _index, value = raw_elements[position]
        if kind == "amendment":
            segment = []
            while position < len(raw_elements) and raw_elements[position][0] == "amendment":
                segment.append(raw_elements[position][2])
                position += 1
            contracts, chain = build_amendment_chain(current_contract, segment)
            # Named outage amendments are checked against the binding of the
            # board they amended (the segment base), never the invocation's.
            validate_backend_outage_amendments_for_binding(chain, segment_base.seat_binding())
            if elements and chain and chain[0].effective_from_round <= elements[-1].effective_from_round:
                raise _fail(
                    f"signed amendment at {chain[0].comment_locator} is effective from round "
                    f"{chain[0].effective_from_round}, which does not follow round "
                    f"{elements[-1].effective_from_round} of the element it amends"
                )
            pre = current
            for contract, amendment in zip(contracts[1:], chain):
                identity = _contract_identity(contract, current, last_seen)
                elements.append(
                    LineageElement(
                        kind="amendment",
                        comment_index=amendment.comment_index,
                        effective_from_round=amendment.effective_from_round,
                        digest=amendment.digest,
                        identity=identity,
                        contract=contract,
                        segment_base=segment_base,
                        amendment=amendment,
                    )
                )
                current, current_contract = identity, contract
            if chain:
                authorizations.append(_segment_authorization(pre, current))
            continue
        link: BoardChangeLink = value
        position += 1
        if link.root != root:
            raise _fail(f"the board-change record at comment {link.comment_index + 1} names another root")
        predecessor = elements[-1].digest if elements else root.digest
        if link.predecessor != predecessor:
            raise _fail(
                f"the board-change record at comment {link.comment_index + 1} does not chain "
                "from the preceding lineage element"
            )
        if (
            link.new_board.policy != root.policy
            or link.new_board.broad_rules != root.broad_rules
        ):
            raise _fail(
                f"the board-change record at comment {link.comment_index + 1} changes the review "
                "policy or broad-path rules, which stay immutable"
            )
        previous = elements[-1] if elements else None
        if previous is not None:
            if (
                previous.kind == "link"
                and previous.effective_from_round == link.effective_from_round
            ):
                assert previous.link is not None
                if (
                    previous.link.anchor_round != link.anchor_round
                    or previous.link.advance != link.advance
                ):
                    raise _fail(
                        f"the board-change record at comment {link.comment_index + 1} "
                        "supersedes a pending record but does not copy its anchor round and "
                        "advance flag"
                    )
                superseded.add(previous.digest)
            elif link.effective_from_round <= previous.effective_from_round:
                raise _fail(
                    f"the board-change record at comment {link.comment_index + 1} is effective "
                    f"from round {link.effective_from_round}, which does not follow round "
                    f"{previous.effective_from_round} of its predecessor"
                )
        elements.append(
            LineageElement(
                kind="link",
                comment_index=link.comment_index,
                effective_from_round=link.effective_from_round,
                digest=link.digest,
                identity=link.new_board,
                contract=link.new_board.contract(),
                segment_base=link.new_board,
                link=link,
            )
        )
        for seat in link.new_board.seats:
            last_seen[seat.seat_id] = seat
        current = link.new_board
        current_contract = elements[-1].contract
        segment_base = link.new_board

    _validate_legacy_seat_bindings([r for r in ordered if r.metadata.pr_board_root is None])

    # Pass 2: walk records and links in comment order, accumulating binding
    # initializations and validating every record against its governing element.
    inits: dict[str, BoardSeat] = {}
    carried: set[str] = set()
    by_index = {element.comment_index: element for element in elements}
    events = sorted(
        [(record.index, "record", record) for record in ordered]
        + [(element.comment_index, "element", element) for element in elements],
        key=lambda item: (item[0], item[1] != "element"),
    )
    previous_identity = root
    for index, kind, value in events:
        if kind == "element":
            element: LineageElement = value
            if element.kind == "link":
                link = element.link
                assert link is not None
                view = apply_inits(previous_identity, inits)
                if not binding_compatible(link.old_board, view):
                    raise _fail(
                        f"the board-change record at comment {index + 1} names an old board "
                        "that differs from the board derived at that point"
                    )
                _merge_inits(inits, link.binding_init, locator=f"comment {index + 1}")
            previous_identity = element.identity
            continue
        record = value
        metadata = record.metadata
        governing = None
        for element in elements:
            if element.comment_index < record.index and element.effective_from_round <= metadata.round_number:
                governing = element
        expected_amendment = (
            governing.digest if governing is not None and governing.kind == "amendment" else None
        )
        expected_change = (
            governing.digest if governing is not None and governing.kind == "link" else None
        )
        governing_identity = governing.identity if governing is not None else root
        governing_contract = governing.contract if governing is not None else root_contract
        # A malformed optimization record is not a contract claim; recovery
        # handles it, and qualification checks it separately.
        contract = (
            contract_from_metadata(metadata)
            if metadata.scheduler_metadata_status != "invalid"
            else None
        )
        locator = f"round {metadata.round_number} record (comment {record.index + 1})"
        if metadata.pr_board_root is not None:
            if metadata.reviewer_board_amendment_digest is not None and metadata.reviewer_board_change_digest is not None:
                raise _fail(f"{locator} carries both a signed-amendment and a board-change digest")
            known = {element.digest for element in elements}
            for digest in (metadata.reviewer_board_amendment_digest, metadata.reviewer_board_change_digest):
                if digest is not None and digest not in known:
                    raise _fail(f"{locator} carries an unknown board-change or amendment digest")
            if (
                metadata.reviewer_board_amendment_digest != expected_amendment
                or metadata.reviewer_board_change_digest != expected_change
            ):
                position = elements.index(governing) if governing is not None else -1
                previous_element = elements[position - 1] if position > 0 else None
                previous_digest = previous_element.digest if previous_element is not None else None
                if (
                    governing is not None
                    and governing.kind == "amendment"
                    and previous_digest in {
                        metadata.reviewer_board_amendment_digest,
                        metadata.reviewer_board_change_digest,
                    }
                    or governing is not None
                    and governing.kind == "amendment"
                    and previous_element is None
                    and metadata.reviewer_board_amendment_digest is None
                    and metadata.reviewer_board_change_digest is None
                ):
                    # Posted after the amendment comment by a run that never
                    # read it: never reinterpreted (#1133).
                    from .board_amendment import STALE_UNREAD_AMENDMENT_HINT

                    raise drift_error(
                        contract if contract is not None else governing_contract,
                        f"round {metadata.round_number} scheduler record (comment "
                        f"{record.index + 1}) was posted under the amendment at "
                        f"{governing.amendment.comment_locator} but does not carry its amended "  # type: ignore[union-attr]
                        "contract and digest: it was posted after that amendment comment without "
                        "the run reading it, and such records are never reinterpreted "
                        f"[{STALE_UNREAD_AMENDMENT_HINT} at {governing.amendment.comment_locator}]",  # type: ignore[union-attr]
                    )
                raise _fail(
                    f"{locator} does not carry the digest of the lineage element that governs it"
                )
            own = _inits_from_payload(metadata.pr_board_binding_init, board=governing_identity)
            _merge_inits(inits, own, locator=locator, view=apply_inits(governing_identity, inits))
            expected_identity = apply_inits(governing_identity, inits)
            try:
                recorded_identity = BoardIdentity.from_payload(metadata.pr_board_identity)
            except AgentLoopError:
                raise _fail(f"{locator} carries a malformed board identity") from None
            if recorded_identity != expected_identity:
                raise _fail(f"{locator} carries a board identity that differs from its governing element")
            base = governing.segment_base if governing is not None else root
            allowed_bindings = (
                apply_inits(base, inits).seat_binding(),
                expected_identity.seat_binding(),
            )
            if metadata.seat_binding not in allowed_bindings:
                raise _fail(
                    f"{locator} carries a seat binding that differs from the board that governs it"
                )
            if metadata.role == "reviewer" and metadata.agent not in expected_identity.names and (
                metadata.agent not in base.names
            ):
                raise _fail(f"{locator} is a review by {metadata.agent!r}, which is not on its board")
            if contract is not None and contract != governing_contract:
                raise drift_error(
                    contract,
                    f"round {metadata.round_number} scheduler record (comment {record.index + 1}) "
                    "does not carry the contract of the lineage element that governs it",
                )
            if governing is not None:
                carried.add(governing.digest)
            continue
        # A legacy-shaped record: never under a link.
        if governing is not None and governing.kind == "link":
            raise _fail(
                f"{locator} was posted under a board-change record but carries no board identity"
            )
        if contract is not None:
            if contract != governing_contract or metadata.reviewer_board_amendment_digest != expected_amendment:
                if metadata.reviewer_board_amendment_digest is not None and (
                    metadata.reviewer_board_amendment_digest not in {e.digest for e in elements}
                ):
                    detail = (
                        f"round {metadata.round_number} scheduler record (comment "
                        f"{record.index + 1}) carries an unknown reviewer-board amendment digest"
                    )
                elif governing is not None:
                    detail = (
                        f"round {metadata.round_number} scheduler record (comment "
                        f"{record.index + 1}) was posted under the amendment at "
                        f"{governing.amendment.comment_locator} but does not carry its amended "  # type: ignore[union-attr]
                        "contract and digest"
                    )
                else:
                    detail = (
                        f"round {metadata.round_number} scheduler record (comment "
                        f"{record.index + 1}) does not match the run's original contract"
                    )
                raise drift_error(contract, detail)
            if governing is not None:
                carried.add(governing.digest)
        elif metadata.reviewer_board_amendment_digest is not None:
            raise AgentLoopError(
                "A contract-neutral round record carries a reviewer-board amendment "
                f"digest (comment {record.index + 1}); the scheduler contract history is "
                "unexplained. Rerun is refused until the history is corrected."
            )
    del by_index

    # Pending elements: superseded links stay uncarried; every other uncarried
    # element must be part of the trailing pending run.
    for element in elements:
        if element.digest in superseded and element.digest in carried:
            raise _fail(
                f"the board-change record at comment {element.comment_index + 1} was activated "
                "but a later record supersedes it at the same round"
            )
    trailing: list[LineageElement] = []
    for element in reversed(elements):
        if element.digest in carried:
            break
        trailing.append(element)
    trailing.reverse()
    trailing_digests = {element.digest for element in trailing}
    for element in elements:
        if (
            element.kind == "link"
            and element.digest not in carried
            and element.digest not in superseded
            and element.digest not in trailing_digests
        ):
            raise _fail(
                f"the board-change record at comment {element.comment_index + 1} never took "
                "effect, but a later element did"
            )
    pending_links = tuple(element for element in trailing if element.kind == "link")
    if pending_links:
        first = next(element for element in trailing if element.kind == "link")
        if any(element.kind == "amendment" for element in trailing if element.comment_index < first.comment_index):
            raise _fail(
                "a board-change record follows a pending signed amendment; activate the signed "
                "amendment first"
            )
        late = [record for record in ordered if record.index > first.comment_index]
        if late:
            raise _fail(
                f"round records were posted after the pending board-change record at comment "
                f"{first.comment_index + 1} without taking it into effect"
            )
        link = first.link
        assert link is not None
        highest = max((record.metadata.round_number for record in ordered), default=None)
        if link.advance != (highest is not None) or (
            highest is not None and highest != link.anchor_round
        ) or (highest is None and link.anchor_round != 1):
            raise _fail(
                f"the pending board-change record at comment {first.comment_index + 1} is "
                f"anchored at round {link.anchor_round} (advance {str(link.advance).lower()}), "
                "but durable history no longer matches it"
            )
    pending_amendments = tuple(
        element.amendment for element in trailing if element.kind == "amendment"
    )
    return BoardLineage(
        root=root,
        anchored=anchored,
        elements=tuple(elements),
        carried=frozenset(carried),
        superseded=frozenset(superseded),
        pending_links=pending_links,
        pending_amendments=pending_amendments,
        inits=dict(inits),
        authorizations=tuple(authorizations),
        posters=posters,
        last_seen=dict(last_seen),
    )


def _validate_legacy_seat_bindings(records: Sequence[object]) -> None:
    """Structural checks of pre-upgrade named history (the #1373 rules).

    Legacy history may change a seat's model chain (that was fresh review),
    but never its backend or its recorded board, and never mixes bound and
    unbound round records.
    """
    from .reviewer_seats import validated_binding_entries

    backends: dict[str, str] = {}
    board: set[str] | None = None
    bound = unbound = False
    for record in records:
        metadata = record.metadata
        binding = metadata.seat_binding
        if binding is None:
            if metadata.role in {"reviewer", "coder", "summary"}:
                unbound = True
            continue
        bound = True
        try:
            entries = validated_binding_entries(binding)
        except AgentLoopError as exc:
            raise _fail(f"PR reviewer seat binding at comment {record.index + 1} is malformed ({exc})") from exc
        for seat_id, entry in entries.items():
            if backends.setdefault(seat_id, entry["backend"]) != entry["backend"]:
                raise _fail(f"reviewer seat {seat_id!r} changed backend in PR history")
        if board is None:
            board = set(entries)
        elif set(entries) != board:
            raise _fail("a reviewer seat binding changed its recorded board without a recorded change")
        if metadata.role == "reviewer" and metadata.agent not in entries:
            raise _fail("a PR reviewer record is not represented in its seat binding")
    if bound and unbound:
        raise _fail("PR history mixes bound and unbound round records")


def _merge_inits(
    inits: dict[str, BoardSeat],
    new: Mapping[str, BoardSeat],
    *,
    locator: str,
    view: BoardIdentity | None = None,
) -> None:
    for seat_id, seat in new.items():
        existing = inits.get(seat_id)
        if existing is not None:
            if existing != seat:
                raise _fail(f"{locator} initializes {seat_id!r} differently from an earlier record")
            continue
        if view is not None:
            current = view.seat(seat_id)
            if current is None or current.model_chain is not None:
                raise _fail(f"{locator} initializes {seat_id!r}, whose binding is already known")
        inits[seat_id] = seat


# --- Candidate links and re-entry -------------------------------------------------


@dataclass(frozen=True)
class BoardAnchor:
    """The lineage-aware anchor: the highest recorded round and its ledger."""

    round_number: int
    has_records: bool
    ledger: tuple[object, ...]


def board_anchor(records: Sequence[object]) -> BoardAnchor:
    """Highest round among all PR round records of any role or phase (D4).

    The ledger is the latest record in that round that carries one, including
    a scheduler-prelaunch summary; with no record the run starts at round 1.
    """
    ordered = sorted(records, key=lambda record: record.index)
    if not ordered:
        return BoardAnchor(round_number=1, has_records=False, ledger=())
    highest = max(record.metadata.round_number for record in ordered)
    in_round = [record for record in ordered if record.metadata.round_number == highest]
    ledger: tuple[object, ...] = ()
    for record in in_round:
        if record.metadata.prior_items or record.metadata.role in {"coder", "summary"}:
            ledger = tuple(record.metadata.prior_items)
    return BoardAnchor(round_number=highest, has_records=True, ledger=ledger)


@dataclass(frozen=True)
class BoardChangeCandidate:
    payload: dict[str, object]
    digest: str
    old_board: BoardIdentity
    new_board: BoardIdentity
    effective_from_round: int
    advance: bool
    anchor_round: int


def configured_differs(configured: BoardIdentity, view: BoardIdentity) -> bool:
    """Whether the configured board is a board change against the current view.

    Unknown bindings in ``view`` compare compatibly (they are initialized by
    the next durable artifact); initialized seats compare exactly.
    """
    if configured.names != view.names or configured.primary != view.primary:
        return True
    return not all(seats_compatible(a, b) for a, b in zip(view.seats, configured.seats))


def policy_differs(configured: BoardIdentity, root: BoardIdentity) -> bool:
    return configured.policy != root.policy or configured.broad_rules != root.broad_rules


def build_board_change_candidate(
    lineage: BoardLineage,
    *,
    configured: BoardIdentity,
    records: Sequence[object],
    pr_number: int,
) -> BoardChangeCandidate:
    """The link a differing configured board would post (D4)."""
    if lineage.pending_amendments:
        raise AgentLoopError(
            "Human decision required: the configured reviewer board differs from the board "
            "of a pending signed reviewer-board amendment. Rerun once with the amended board "
            "so the signed amendment takes effect, then change the board."
        )
    view = lineage.current_identity
    latest = lineage.latest
    if lineage.pending_links:
        pending = lineage.pending_links[-1].link
        assert pending is not None
        anchor_round, advance = pending.anchor_round, pending.advance
    else:
        anchor = board_anchor(records)
        anchor_round, advance = anchor.round_number, anchor.has_records
    inits = {
        seat_id: seat
        for seat_id, seat in pending_initializations(view, configured).items()
        if seat_id in configured.names
    }
    payload = board_change_link_payload(
        pr_number=pr_number,
        root=lineage.root,
        predecessor=latest.digest if latest is not None else lineage.root.digest,
        old_board=view,
        new_board=configured,
        old_seat_binding=lineage.binding_for_round(
            max(anchor_round, latest.effective_from_round if latest is not None else 1)
        ),
        binding_init=inits,
        anchor_round=anchor_round,
        advance=advance,
    )
    return BoardChangeCandidate(
        payload=payload,
        digest=board_change_link_digest(payload),
        old_board=view,
        new_board=configured,
        effective_from_round=int(payload["effective_from_round"]),  # type: ignore[arg-type]
        advance=advance,
        anchor_round=anchor_round,
    )


@dataclass(frozen=True)
class _SyntheticComment:
    body: str
    author_id: int | None


def comments_with_candidate(
    comments: Sequence[object], candidate: BoardChangeCandidate, *, actor_id: int
) -> tuple[object, ...]:
    """``comments`` with the unposted candidate appended, for read-only validation."""
    return (
        *comments,
        _SyntheticComment(body=str(render_board_change_link(candidate.payload)), author_id=actor_id),
    )


def link_already_posted(comments: Sequence[object], digest: str) -> bool:
    return any(
        isinstance(body := _body(comment), str)
        and BOARD_CHANGE_LINK_HEADING in body
        and digest in body
        for comment in comments
    )


# --- Round-record stamping ----------------------------------------------------------


@dataclass
class PrBoardStamp:
    """Board identity fields stamped on every PR round record of this run."""

    root: dict[str, object]
    identity: dict[str, object]
    change_digest: str | None
    amendment_digest: str | None
    pending_init: dict[str, object] | None = None

    def apply(self, metadata: object) -> object:
        # The initialization rides on every record of the run: a render that
        # is never posted must not consume it, and an identical repeat is a
        # no-op when the lineage is resolved.
        if getattr(metadata, "flow", None) != "pr":
            return metadata
        return dataclasses.replace(
            metadata,
            pr_board_root=self.root,
            pr_board_identity=self.identity,
            pr_board_binding_init=self.pending_init or None,
            reviewer_board_change_digest=self.change_digest,
            reviewer_board_amendment_digest=self.amendment_digest,
        )


def stamp_for_lineage(
    lineage: BoardLineage, *, configured: BoardIdentity
) -> PrBoardStamp:
    """The stamp for records this run posts under the lineage's latest element."""
    view = lineage.current_identity
    inits = pending_initializations(view, configured)
    identity = apply_inits(view, inits)
    return PrBoardStamp(
        root=lineage.root.payload(),
        identity=identity.payload(),
        change_digest=lineage.active_change_digest,
        amendment_digest=lineage.active_digest,
        pending_init=init_payload(inits) if inits else None,
    )


# --- Carry-over (D4b) -------------------------------------------------------------------


@dataclass(frozen=True)
class CarryOver:
    items: tuple[object, ...]
    next_item_number: int
    evidence_clearances: tuple[tuple[str, str], ...]


def carry_over_anchor_round(
    *,
    prior_items: Sequence[object],
    completed_reviews: Sequence[object],
    settled_new_items: Sequence[object],
    records: Sequence[object],
    round_number: int,
    reconciliation_mode: str,
    evidence_clearances: Sequence[tuple[str, str]],
    next_item_number: int,
    board_seats: Sequence[str] = (),
) -> CarryOver:
    """Collect, reconcile, merge and rebuild evidence for the anchor round.

    ``completed_reviews`` are the anchor round's reviewer records collected
    under the *predecessor* board, so an outgoing reviewer's findings are
    never filtered out.  Dispositions are applied with the existing
    reconciliation rules (items resolved or partially cleared there stay
    settled), new items are merged, and recorded evidence requests are
    rebuilt with the accumulated head-scoped clearances.  New-board ownership
    is applied later by the caller's ledger view.
    """
    from .round_state import (
        _aggregate_record_dispositions,
        _max_unresolved_item_number_from_records,
    )
    from .unresolved_items import _apply_unresolved_item_dispositions, _upsert_evidence_obligation

    # A review may only carry findings in its own name: one naming another
    # board seat as its author is never adopted.
    seats = set(board_seats)
    for review in completed_reviews:
        for item in review.metadata.new_items:
            if item.reviewer != review.metadata.agent and item.reviewer in seats:
                raise AgentLoopError(
                    "PR board change cannot verify the owner of a carried finding."
                )
    clearances = list(evidence_clearances)
    reconciled_clearances: list[tuple[str, str]] = []
    items, _future = _apply_unresolved_item_dispositions(
        prior_items,
        _aggregate_record_dispositions(completed_reviews),
        retain_future=False,
        reconciliation_mode=reconciliation_mode,
        round_number=round_number,
        evidence_clearances=reconciled_clearances,
    )
    items = [item for item in items if item.status in {"blocking", "same-pr"}]
    by_id = {item.item_id: item for item in items}
    new_items = list(settled_new_items) or [
        item for review in completed_reviews for item in review.metadata.new_items
    ]
    for item in new_items:
        existing = by_id.get(item.item_id)
        if existing is not None and existing != item and existing.reviewer != item.reviewer:
            raise AgentLoopError(
                "PR board change found conflicting provenance for a carried finding."
            )
        if item.status in {"blocking", "same-pr"} and item.item_id not in by_id:
            by_id[item.item_id] = item
            items.append(item)
    for clearance in reconciled_clearances:
        if clearance not in clearances:
            clearances.append(clearance)
    number = max(
        next_item_number,
        _max_unresolved_item_number_from_records(list(records)) + 1,
        max(
            (
                int(match.group(1)) + 1
                for item in items
                if (match := re.fullmatch(r"item-(\d+)", item.item_id))
            ),
            default=1,
        ),
    )
    for review in completed_reviews:
        for request in review.metadata.evidence_requests:
            items, consumed = _upsert_evidence_obligation(
                items,
                item_number=number,
                reviewer=review.metadata.agent,
                text=request,
                source_round=review.metadata.round_number,
                current_head_sha=review.metadata.subject,
                clearances=clearances,
            )
            if consumed:
                number += 1
    return CarryOver(
        items=tuple(items),
        next_item_number=number,
        evidence_clearances=tuple(clearances),
    )


def carried_secondary_obligations(
    items: Sequence[object], *, epoch_round: int | None, primary: str | None
) -> tuple[str, ...]:
    """Active items pending on secondaries that predate the primary epoch (D5a)."""
    if epoch_round is None:
        return ()
    carried: list[str] = []
    for item in items:
        if item.status not in {"blocking", "same-pr"} or item.source_round >= epoch_round:
            continue
        owners = tuple(item.resolution_owners or (item.reviewer,))
        states = dict(item.owner_states or ((owner, "pending") for owner in owners))
        if any(owner != primary and states.get(owner) != "cleared" for owner in owners):
            carried.append(item.item_id)
    return tuple(carried)


def governing_seat_binding_from_comments(
    comments: Sequence[object],
    *,
    round_number: int,
    actor_id: int | None,
    current: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """The seat binding that governed ``round_number``, from link records alone.

    Used by managed-CI history checks that read raw REST comments.  Without
    any admitted link the current binding governs every round, as before.
    Links not authored by ``actor_id`` are ignored here: startup and
    qualification fail closed on them.
    """
    admitted: list[tuple[int, object, object]] = []
    for comment in comments:
        body = _body(comment)
        match = BOARD_CHANGE_LINK_RE.search(body) if isinstance(body, str) else None
        if match is None:
            continue
        user = comment.get("user") if isinstance(comment, dict) else None
        author_id = (
            user.get("id") if isinstance(user, dict) else getattr(comment, "author_id", None)
        )
        if actor_id is None or author_id != actor_id:
            continue
        try:
            payload = json.loads(base64.urlsafe_b64decode(match.group("payload").encode("ascii")))
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("kind") != BOARD_CHANGE_LINK_KIND:
            continue
        effective = payload.get("effective_from_round")
        if isinstance(effective, int) and not isinstance(effective, bool):
            admitted.append(
                (effective, payload.get("old_seat_binding"), payload.get("new_seat_binding"))
            )
    if not admitted:
        return current
    governing: object = ...
    for effective, _old, new in admitted:
        if effective <= round_number:
            governing = new
    binding = admitted[0][1] if governing is ... else governing
    return binding if isinstance(binding, dict) else None
