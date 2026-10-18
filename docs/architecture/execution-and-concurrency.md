# Execution and concurrency

How agent and Git processes run: per-run worktrees over a shared store, the agent-loop-owned Git boundary and private fetch transport with SSH identity selection, pre-turn checkout verification, process-tree containment, test interpreter and worker budgets, pre-collection launch-failure provenance, the Antigravity quota-group fallback, and the other entry paths outside the review loop.

Source paths: `src/coding_review_agent_loop/run_worktrees.py`, `src/coding_review_agent_loop/workdirs.py`, `src/coding_review_agent_loop/workdir_claims.py`, `src/coding_review_agent_loop/checkout_verification.py`, `src/coding_review_agent_loop/secure_git.py`, `src/coding_review_agent_loop/git_transport.py`, `src/coding_review_agent_loop/transport_check.py`, `src/coding_review_agent_loop/runner.py`, `src/coding_review_agent_loop/containment.py`, `src/coding_review_agent_loop/test_workers.py`, `src/coding_review_agent_loop/test_runtime.py`, `src/coding_review_agent_loop/lifecycle_probe.py`, `src/coding_review_agent_loop/agents/antigravity.py`, `src/coding_review_agent_loop/discuss_loop.py`, `helpers/`, `tests/test_run_worktrees.py`, `tests/test_containment.py`, `tests/test_test_workers.py`.

## Execution and Concurrency

Concurrent default-dir invocations on one repository are supported.
Each run with an omitted `--<agent>-dir` gets its own detached `git worktree`
under `<scratch>/OWNER-REPO/<agent>/runs/<run-token>`, backed by one shared
clone per repo/agent (the *store*, `<scratch>/OWNER-REPO/<agent>/repo`) that
owns the object store. Explicit `--<agent>-dir` paths are untouched. Optional `--worktree-link PATH` (repeatable, off by default) symlinks (never copies) an untracked repository-relative path from the shared store, for example `.venv`, into each per-run worktree. The path must exist in the store (otherwise the run fails, naming it) and its parent directories must be tracked. Explicit `--<agent>-dir` checkouts ignore it. The link is exempted from checkout verification, sync cleaning and test-evidence attribution by agent-loop's in-process registry only (no git ignore state is written); a removed, retargeted, staged or committed link is refused. Linked state is shared and mutable across concurrent runs. Keep it ignored with `/.venv` (a dir-only `.venv/` rule does not match a symlink). `.venv/bin/python` works through the link; bare `python` reaches the linked venv only if the agent environment activates it. Parallel
reviewers within one invocation are supported; the orchestrator verifies
distinct reviewer workdirs. Coder and reviewer turns are separate lifecycle
stages even when they use the same CLI.

Named seats with no explicit `--seat-dir` use the same per-run worktree lifecycle,
with a separate shared store and finalizer keyed by each seat ID.

**Per-run worktrees (`run_worktrees.py`).** Store mutations (initialization,
private fetch and pack import, local-base fast-forward, administrative link
creation, prune, remove) run under a short,
non-reentrant host `flock` (`<lock root>/workdir-stores/`) that is never held
across an agent turn; a nested acquisition fails fast. A probe of the store
path's own run claim (`workdir_claims.probe_free_claim`) interlocks with an
explicit-dir or older-version run that checked out a branch in the store: while
it is held the store's HEAD, branch, index and tree are untouched and the run
only imports objects and updates tracking refs. Every worktree has a durable owner record (`<store-key>/<token>.json`,
pending then ready, with the root and admin-dir identities) under the lock
root. Base and PR syncs check out an immutable SHA pinned under the store lock,
so a concurrent fetch cannot move another run's checkout; the PR path still
compares the pin with the advertised head. The local `<base>` ref moves only by
a guarded compare-and-swap `git update-ref` fast-forward, skipped when `<base>` is checked out in
any worktree. The outermost `workdir_claim_scope` runs registered finalizers
(identity-verified worktree removal) before releasing claims, on success and
failure. At startup a run prunes worktrees whose owner record matches and whose
claim lock is free; unrecognized entries are logged and left untouched.

**Agent-loop-owned Git boundary (`secure_git.py`, `git_transport.py`).** Every
local text, binary, and cancellable Git call that can read an agent checkout
uses the pinned executable, a closed child environment, and command-scoped
neutralizers. Linux loads a sealed preload library whose constructor installs
no-new-privileges and an inherited seccomp filter denying exec and network;
macOS loads a private library whose constructor applies Seatbelt before Git
main, denying process exec, fork, and network without a pinned-Git exception;
Windows starts Git suspended, assigns a kill-on-close one-process Job Object,
then resumes it. Each backend performs a real canary before the first checkout
probe and fails closed if confinement cannot be verified. Existing process
group containment is not an execution boundary. The isolated inspector keeps
its standard-library-only main process and delegates Git through verified
helper bytes executed by an isolated interpreter using the same local boundary.
Guard C source and the Windows launcher are checked against hashes held by the
already loaded controller before use. The compiler receives verified C bytes
through stdin, and Windows passes verified launcher code directly to a trusted
Python executable, leaving no helper file to replace between check and launch.
Each local Git invocation also binds `--work-tree` and `core.worktree` to its
assigned checkout, so repository config cannot redirect status or cleanup to
another directory. Repository initialization has no existing worktree to bind.

Fetches run only in a fresh private transport repository, with clean Git
configuration and an exact operator-selected GitHub HTTPS or SSH URL. An
explicit `--trusted-local-origin` is required for local paths. Private HTTPS
may obtain a token from an operator-owned GitHub CLI installation resolved
independently of inherited `PATH`, passed only as an in-memory scoped
header to that private fetch. On POSIX hosts, both pinned executable paths,
including their parent directories, must be root-owned so an unrestricted
same-user agent cannot replace them before launch.
Transport failures raise `TransportFailure`, whose `label` carries the fixed
class. `agent-loop transport-check` (wrapped by `scripts/transport_smoke.sh`)
runs a local configuration preflight and then one production `import_ref` fetch
into a private scratch repository with the operator's credentials. It reports
only that class or a fixed fallback (`configuration`, `local import`,
`unclassified`). It is required before merging transport changes.
The `--trusted-origin-protocol` setting selects HTTPS or SSH when creating a
new default checkout; the pinned SSH client is validated before transport.
Its command keeps `-F /dev/null` and executable-option hardening, retains
default keys and `SSH_AUTH_SOCK`, and pins `StrictHostKeyChecking=yes` before
any other option, so an unknown or changed host key fails closed without a
prompt (`BatchMode` stays unset for passphrase prompts). A trusted host with a
port uses an `ssh://` origin.
SSH identity selection: on POSIX, `git_transport` reads a non-executing
subset of `~/.ssh/config` for the literal URL host (`Host` blocks only; never
`ssh -G`). It honours `IdentityFile`, `HostName`, `IdentitiesOnly`, a single
`UserKnownHostsFile`, and `Port` only when the URL has no port. `User`,
`Match`, `Include`, `ProxyCommand`, `ProxyJump`, `LocalCommand`,
`KnownHostsCommand` and all other keys are ignored. Values are emitted as
explicit `-o`/`-i` options only when their final form (after `~/` expansion and
realpath) matches `[A-Za-z0-9._/+,:@-]+` without a leading `-`; anything else is
refused rather than escaped, so neither the shell nor ssh's option grammar or
`%`/`${}`/`~` expansion can reinterpret it. The config file, identity files and
the known-hosts file pass one trust walk (`_trusted_operator_file`). It
resolves the path component by component, expanding each symlink in place
(bounded hops), so every component traversed, including intermediate links
and the directories their targets pass through, is owned by the current uid
or root and, unless it is a symlink, not group- or world-writable. Root-owned
sticky directories are exempt, but never the final file or the parent of an
accepted missing file. The supplied path and every traversed component must
lie outside `transport_forbidden_roots`, which covers every agent checkout,
named-seat workdir, store, run-worktree root and import destination, so an
agent-controlled link cannot select an otherwise trusted file. The target must
be a regular file. The config
is opened `O_NOFOLLOW|O_NONBLOCK`, inode-checked against the walk and read with
a 256 KiB bound. An explicit `--git-ssh-identity` / `AGENT_LOOP_GIT_SSH_IDENTITY`
replaces config identities with exactly one `-i` and `IdentitiesOnly=yes`, and
an untrusted override is a fixed error. Windows does not read the config, keeps
the previous command byte-for-byte and refuses the override. Transport failures
report a fixed class chosen by evidence precedence (`remote not found`,
`host-key failure`, `network or timeout`, then `authentication failure` from
specific evidence before Git's generic access-rights footer, else
`unclassified`) with no stderr or argv; SSH authentication and host-key
failures add a fixed hint. No HTTPS fallback is attempted after SSH failure. Same-UID live races against `~/.ssh` remain out of
scope.
The complete reachable pack for a pinned commit
is streamed to confined `index-pack --stdin`; the destination verifies the
commit, then compare-and-swaps the tracking ref while its checkout or store
lock is held. Base ancestry and advertised PR head are checked before cleaning
or checkout. Git 2.43 spawns `update-ref` even for `worktree add --no-checkout`;
fresh linked worktrees therefore create only the administrative link under
`store_lock`, run a separate confined `reset --hard` at the pinned SHA, and
publish a ready owner record only after administrative identity, HEAD, and
clean status verify. Any setup failure removes only the recorded root and
administrative inode identities; incomplete cleanup is reported as an error.

**Local editor handoff primitive (fork #1, checkpoint 4A).**
`git_handoff.py` is an internal local-only seam, not connected to the issue or
PR loops. Its caller must retain the existing worktree ownership claim and
verify the assigned checkout. Call `Handoff.begin` before editing with a trusted
expected base SHA and fixed commit identity/message, then `capture` after the
editor exits with explicit approved new paths. The host reads candidate bytes
through descriptors and builds the complete expected tree using native Git in
separate trusted scratch metadata. Writes do not use alternates, since even
writing an existing object can refresh timestamps in an alternate store.

`run_local_commit` offers only fixed, candidate-allowlisted staging and native
`commit-tree` operations in a Linux bubblewrap namespace. The checkout, shared
Git administration and host scratch metadata are read-only; a separate disposable
index/object directory is writable. Its Git administration/configuration remains
host-owned and read-only. `verify` reads operator loose objects directly,
checks their hashes, exact parent/tree and fixed metadata, and compares every
reachable candidate tree/blob with host expectations. It never loads operator
configuration, replacements, grafts or alternate paths. Source, pointer and
shared-administration changes block verification.

This slice rejects no-change candidates, ignore-rule additions/edits (including
self-hiding ignore files), submodules, escaping links, oversized inputs, shared
alternate stores and operator objects available only in packs. Ignored artifacts
are excluded; untracked enumeration is bounded even for ignored entries.
Registered environment links are excluded only while intact. The Git-only
namespace disables networking and must not be reused as a Claude runtime. Missing confinement fails closed; synthetic tests that cannot create
namespaces skip the OS probes and provide no source-immutability claim.

**Claude operator and host publication modules (fork #1, checkpoint 4B).**
`ClaudeGitOperator(executable).run(handoff, auth=...)` in
`claude_git_operator.py` is the live model entry point, separate from both
lifecycle loops. `prepare` exposes its launch for inspection and is a context
manager: callers must close it to release pinned descriptors. Linux bubblewrap
and trusted system Git/Python remain mandatory. The default executable policy
requires a root-owned executable under `/usr`. An explicit
`local_executable_sha256` constructor argument also permits a native local
Claude installation, such as the canonical file under
`~/.local/share/claude/versions/`. The host supplies the expected SHA-256;
launcher scripts, symlinks (including ancestor links), group/other-writable
files and executables inside handoff storage are refused. The host opens the
file through directory descriptors, pins its inode, mode, ownership, timestamps
and digest, and binds that descriptor read-only at `/opt/reviewloop/claude`.
The installation's directory and host home are not mounted. Pins are checked
before and after invocation, and descriptors close on failure as well as success.
A fixed isolated Python bootstrap closes all non-stdio descriptors before
executing Claude, so host bind-source descriptors cannot bypass the read-only
namespace mounts. The bootstrap grants no shell or additional model tools.
Source mount layouts overlapping the isolated executable or runtime paths are
refused before role setup, so a later mount cannot hide an attested file.

Authentication requires exactly one explicit in-memory `ANTHROPIC_API_KEY` or
`CLAUDE_CODE_OAUTH_TOKEN`, or an explicit `session_file` constructor argument
pointing to an existing `.credentials.json`. Session files must be private,
owned by the invoking user, regular, single-link files outside handoff storage.
The host opens only an `O_PATH` descriptor and never reads, copies, extracts or
logs the contents. The exact inode is mounted read-only at
`/tmp/operator-home/.claude/.credentials.json`; the rest of that disposable
`CLAUDE_CONFIG_DIR` is writable ephemeral scratch. No host Claude settings,
parent directory or other login files are mounted. Token and session-file
authentication cannot be combined. File replacement or modification blocks
the run. A login that requires credential refresh/writes or other host files
must fail closed; this implementation has no fallback or token extraction.
Existing subscription login compatibility still requires the owner-run smoke
test; the synthetic tests do not establish actual CLI authentication.

Missing auth/confinement fails closed. A fresh
mount root exposes system runtime files, the canonical read-only source, trusted
read-only administration and read-only shared Git administration; the host home
is unavailable. Only private index/objects and ephemeral runtime
scratch are writable. The model retains network access for Anthropic. Project
credential/settings directories are masked; symlinked environment masks are
currently refused. Persistent Claude settings, SSH sockets, GitHub credentials
and provider pass-through environment variables are absent.

Built-in Claude tools are disabled (`--tools ''`). A strict, explicit stdio MCP
configuration exposes only `status`, `diff`, `stage` and `commit`, each accepting
zero arguments. This small adapter dispatches the host-generated immutable Git
wrapper, with fixed paths, Git environment, author/committer/message and no shell.
It avoids Claude's implicit read-only Bash auto-approvals. Restricted mode,
noninteractive permission denial, isolated settings, disabled hooks and
`--no-session-persistence` apply independently. Native staging and `commit-tree`
use the private index/loose objects and the pinned initial HEAD. A returned SHA
is only usable after `Handoff.verify` succeeds. Runtime files have a per-file
32 MiB limit, and ephemeral `/tmp` has a 256 MiB bound. The private object/index
area retains the candidate's existing input bounds; it has no aggregate quota.

`git_publication.publish_verified(handoff, sha, repo=..., origin=..., branch=...,
expected_head=...)` is a host-only gate. It re-verifies raw objects and imports
them, plus pinned trusted base history, into a disposable host-owned bare store.
It reuses `git_transport` URL validation, sanitized transport environments,
trusted authentication and fixed-error transport commands. No outbound command
runs in operator metadata or the source checkout. The push names only the exact
SHA/ref with an absent-head lease for a new issue, or the known parent SHA for a
correction. A retry accepts the verified published SHA; any foreign head blocks.
The helper checks the remote ref after push. Caller-held worktree/store claims
must span capture, model execution, verification and publication.

`bind_ordinary_pr(..., api=OrdinaryPrApi(runner, config))` creates or binds one
ordinary PR using the existing managed-PR API transport and GitHub closing-issue
parsers, without managed labels, draft policy or CI waivers. A fake API is an
executable test seam. The host supplies a durable handoff ID, base, branch, SHA
and exact closing-issue set. Binding validates the authenticated creator,
same-repository head/base, SHA, open state and contract. It refuses closed,
foreign or duplicate PRs. An uncertain POST is reconciled by listing again;
retrying the same recorded identity does not blindly create another PR. PR
binding currently supports `owner/repository` identities only.

These modules are **not wired into issue_loop.py or pr_loop.py**. Checkpoint
4C is partial: local executable/session attestation and its internal synthetic
publication slice are implemented. Remaining work must add the explicit
editor/operator option and role validation,
retain claims across the seam, write EDITED/COMMITTED/PUSHED/PR_BOUND into existing
round state, persist authenticated provenance/closing-contract records, and
schedule a separate reviewer on the published head. The caller must preflight
repository policy (including unsupported signed-commit requirements) and use
trusted round-state values, never model-selected destinations or handoff IDs.
Ordinary PR correction binding reuses the original handoff ID; existing legacy
PRs need a trusted adoption path before using this identity contract.

LFS, submodules, ignore-rule changes, escaping links, oversized files (including
binary files above 16 MiB), total candidate content above 128 MiB and packed-only
operator output remain unsupported. No live Claude/GitHub invocation was used
for checkpoint 4B. Synthetic tests cover wrapper/MCP dispatch, raw verification,
local-bare publication, exact leases/races, credential absence and PR retries.
Namespace denial tests skip when bubblewrap is unavailable. The host must still
run the separately authorized disposable **new MCP-role** Claude proof; the
previous Bash proof does not establish this invocation's CLI/auth behavior.

**Pre-turn checkout verification.** A claim protects against another
agent-loop run, not against an operator, editor or stray script writing to the
same directory. `run_agent_result` therefore verifies the assigned checkout
before every checked agent turn (every attempt, including retries,
executable-replacement replays and Antigravity fallbacks) and fails closed with
`CheckoutVerificationError`, naming the expected and observed branch/HEAD and
each unexpected path. The in-process ledger in `checkout_verification.py` keeps
one fingerprint per checkout: branch, HEAD, staged blob identity and a content
identity (sha256, or stat identity beyond the 16 MiB per-file / 512 MiB total
hash budget) for every dirty path, with each untracked file listed
individually and paths handled as raw bytes. Probes use `--no-renames`, go
through the hardened git runner in sandboxed mode, and never open a symlinked
ancestor, FIFO, socket or device. Verification never resets, cleans, stashes
or deletes foreign changes. The baseline is established by startup preparation,
verified before every between-turn sync (before any reset or clean) and every
real test gate, and refreshed only afterwards; a gate adopts only new
artifacts. Each completed turn refreshes it, and a post-turn capture that
fails, or exceeds the 100,000-path cap, poisons the entry so a later
clean-but-wrong checkout is refused rather than adopted. `semantic-dedupe` and
`repair` (tool-owned temporary directories) and dry-run are exempt. The
Antigravity backend treats `semantic-dedupe` like `repair`: a GEMINI lock
beside the temporary directory instead of in git metadata, no checkout
snapshot or replacement classification, and the strict empty-allowlist
settings injection for the turn. Explicit
agent directories must be existing git checkouts. Foreign writes *during* a
turn are absorbed into the refreshed baseline; excluding them is the claim's
responsibility.

The Antigravity `GEMINI.md` injection carries a self-describing header
(`orig`, body byte length and sha256) so a leftover from a killed run is
recognised as tool-owned in both the per-turn check and the startup dirty
check, and only the verified header and body are removed. Every agent-loop
access to `GEMINI.md` goes through one fd-relative accessor that touches only a
regular, non-symlink file, and stale recovery takes the backend's non-blocking
`GEMINI.md` lock first.

On supported Linux/systemd/cgroup-v2 hosts, containment bounds aggregate and
role-specific process trees. The portable process-group fallback can terminate
descendants but does not impose memory ceilings. Test-wrapper lane locks prevent
recognized duplicate commands, not all overlapping work in arbitrary shells.
Backend turn timeouts, whole-test-command watchdogs, and framework per-test
timeouts are independent limits. See [containment](../local_agent_loop.md#process-tree-containment).

An operator-configured `--test-python` interpreter is validated, exported to
every coder as `AGENT_LOOP_TEST_PYTHON`, import-checked at each coder-prompt
render and, in sandboxed mode, becomes the single granted test invocation. Claude
coder turns get `BASH_DEFAULT_TIMEOUT_MS`/`BASH_MAX_TIMEOUT_MS` sized by
`run_tests_foreground_budget_seconds` (watchdog ceiling, host-wait bound,
parent overhead, headroom); every other backend and role carries an `unknown`
shell-cap fact. Broker-owned detached runs are deferred. See
[Running a long or full suite](../local_agent_loop.md#running-a-long-or-full-suite).

A containment-aware test-worker budget (`test_workers.py`) is derived after
admission from the limits that apply on each path (managed handle limits,
cgroup ancestry, usable host memory, CPU affinity/quota) and exported to coder
and repair agents as `AGENT_LOOP_TEST_WORKERS`. The parent-owned broker is the
enforcement boundary: it applies the stricter of the parent and client values
and injects a stdlib-only pytest plugin that clamps or refuses the final
resolved pytest-xdist worker count and reports the gateways actually created.
The local fallback is advisory by comparison, and the plugin is a prompt-slip
safety net, not a sandbox. In clamp and refuse one test command per invocation
holds a worker-budget lock. Host-wide admission is exclusive with a bounded
wait (`AGENT_LOOP_TEST_WORKER_HOST_WAIT_SECONDS`): a run waits for the shared
pool inside `run_foreground_test`, around the test command only, and on timeout
falls back to the free workers (or one oversubscribed worker) instead of
refusing; the fallback lowers launched workers after the configured-budget
judgement, so refuse-mode enforcement is not weakened. A broker request's
pending admission is cancelled when the broker stops, synchronised with target
launch so no target starts after the owning turn ends. See
[Parallel test-worker budget](../local_agent_loop.md#parallel-test-worker-budget).

The same test-execution boundary distinguishes positive lifecycle provenance
from vetoes when deciding that a direct-pytest run exited before any test could
run. Provenance is parent-owned: a launch eligibility the parent validated for a
native, non-script, non-isolated interpreter that passed a probe-import
preflight, the `_agent_loop_lifecycle_probe` plugin's strictly parsed report
(injected ahead of later plugins; option parsing never finished and no test-code
file was touched first; a clean `final` record), or an independent parent
importability check when pytest itself is absent. The whole-stream output scan
and the worker-session report are vetoes only, and a missing, blocked or
unreadable signal always fails closed. The probe is a cooperative observation
channel, not a sandbox. A classified run is a persisted non-evidence
`launch-failed` row; see
[Pre-collection launch failures](../local_agent_loop.md#pre-collection-launch-failures).

**Antigravity quota-group fallback.** `agents/antigravity.py` derives a
quota group per chain entry (an explicit override or the model-name family) and
keeps `AntigravityQuotaGroupMemory`, a locked map of exhausted groups with
finite monotonic expiries: the parsed reset, or a bounded
`--antigravity-quota-cooldown-seconds` cooldown. The memory is owned by the
logical run: it is keyed by the `workdir_claims` owner's `run_id`, created on
first use, dropped by a run finalizer when the outermost scope ends, shared by
nested loops, and reached from reviewer threads because `review_rounds.py` and
`discuss_loop.py` submit with `contextvars.copy_context()`. With no owner
(library use, the skill helper process) the memory is local to one attempt
state. Exhaustion is classified in `transient.py` only from the matched
provider error frame, extracted from raw unstripped lines (never the whole
transcript), with capacity wording taking precedence. `AntigravityAttemptState`
is the single policy shared by `validated_agent.py` and `helpers/run_external.py`:
transient failures retry then fall back to a later eligible entry; a verified
exhaustion skips the whole group with no retry or sleep and makes at most one
cross-group jump per turn; a second verified exhaustion is a shared-limit stop;
and an all-groups-cooling state stops before invoking. Only when every group
carries a parsed long reset is `QuotaResetExceededError` raised. Memory is not
persisted across processes.

## Other Entry Paths

- `discuss` coordinates non-implementation debate, with optional research,
  agenda/synthesis, and evidence reconciliation. It does not implicitly enter
  implementation or merge a PR.
- `managed-pr` prepares a managed PR from an existing branch before opening it;
  it is distinct from explicitly adopting an already-open PR.
- `managed-ci preflight`, `containment-preflight`, and `run-tests` expose
  supporting checks and execution helpers outside the full review loop.
- `history export` is a read-only reporting path: it reads issue and linked-PR
  conversations, edit history and closing-PR state from GitHub and writes one
  local JSON or CSV file. It shares no state with a running loop and never
  writes to GitHub.
- `issue --triage` adds one advisory model call before an issue run's first
  agent dispatch and one issue comment. It reads an optional local history
  export file but never changes the route, execution mode or policy the
  operator selected.
- Claude Code skill mode uses an interactive host plus [`helpers/`](../../helpers/)
  for validation, external agents, GitHub operations, and local session state.
  It reuses library functionality but is not an identical CLI lifecycle and
  never auto-merges. Host-session work is outside the external-agent process
  boundary. See [skill architecture](../skill_mode.md#architecture).

## Decisions

[Decision log](decisions/execution-and-concurrency.md)
