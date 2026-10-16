# Round state and resume — decisions

Decision log for [Round state and resume](../round-state-and-resume.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Evidence-conversion authority lives in round metadata and is admitted only from comments authored by the authenticated orchestrator actor (#1325).
- Exactly one byte-exact host comment footer is tolerated, once, at read-back and at three strict-reader boundaries, because some hosts such as Claude Code cloud sessions append it to every comment (#1043).
- A coder follow-up rejected after it pushed is recoverable through `coder-dispatch` and `coder-followup-rejected` records, because the rejection discards the response but not the push (#1292).
- Signed human requirements are honour-system unless trusted actors are configured, and board amendments, child-disposition overrides and child-plan supersessions stay textual (#1022).
