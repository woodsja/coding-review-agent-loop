# Execution and concurrency — decisions

Decision log for [Execution and concurrency](../execution-and-concurrency.md). Entries are append-only: a
replaced entry gets a `Superseded by #M: <clause>.` note appended and is
never rewritten or deleted. The detail doc describes the current design.

- Concurrent default-dir invocations on one repository are supported by giving each run its own detached worktree over one shared store per repository and agent (#1162).
- `agent-loop transport-check` runs one production fetch with the operator's credentials and is required before merging transport changes (#1383).
- SSH transport honours a non-executing subset of `~/.ssh/config` for the literal URL host, refusing rather than escaping any unsafe value, and never runs `ssh -G` (#1380).
- Same-UID live races against `~/.ssh` are left out of scope for the SSH transport (#1371).
- Antigravity tracks quota exhaustion per quota group for the logical run, so a verified exhaustion skips the whole group with no retry instead of retrying each model (#1236).
- The Antigravity `semantic-dedupe` turn reuses the isolated `repair` path (sibling GEMINI lock, no checkout snapshot, strict empty allowlist) with a stdout-only classification contract, so a classifier never creates git metadata in its temporary directory or reads a stale response file (#1472).
- The fork's checkpoint 4A local editor handoff starts as an unwired Git-only primitive: host-owned expected trees and direct raw-object verification remain separate from disposable operator objects/indexes; ignore-rule edits and unsupported inputs fail closed until broader integration is tested (#1).
- The fork's partial checkpoint 4C permits explicitly attested native local Claude executables via descriptor-bound read-only mounts and digest/inode checks before and after execution. Explicit session-file authentication uses an O_PATH descriptor and a read-only mount into disposable CLAUDE_CONFIG_DIR; host Python never reads session contents. Host settings and install/login parent directories remain unmounted. Actual CLI subscription compatibility requires an owner-run disposable smoke test. Issue/PR workflow integration and durable recovery remain outstanding (#1).

### 2026-10-18 — Checkpoint 4D: direct issue transaction only

Reuse the public issue-loop validated turn and review/handoff contracts around
one small `SplitIssueHandoff` adapter. Freeze only an explicit versioned editor
new-file declaration, authenticate evidence before publication, and synchronize
only the verified commit. Keep the public CLI gate blocked: progress messages
are not durable phase state, and PR corrections/crash recovery are outstanding.
