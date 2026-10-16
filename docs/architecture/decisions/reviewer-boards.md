# Reviewer seats and boards — decisions

Decision log for [Reviewer seats and boards](../reviewer-boards.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- The PR reviewer board is a per-invocation choice: a plan approved by one board can hand off to PR review by another, with an optional reviewer floor and a plain board-change audit record, and plan approvals never count as PR approvals (#1373).
- In-run PR board changes are recorded as orchestrator-authored links in one ordered lineage with signed amendments, instead of a frozen roster, under every review policy (#1376).
- Named boards over unanchored hand-off history are adopted only after every such record is authenticated as the orchestrator's own (#1365).
- Issue plan review uses the same board lineage on the issue that owns the plan, and an approved plan is never reopened by a board change (#1377).
