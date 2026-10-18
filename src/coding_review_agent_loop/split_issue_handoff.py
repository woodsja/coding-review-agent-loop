"""Direct-issue and PR-correction transactions with durable phase recovery.

The issue and PR loops own the claim, response validation, provenance and
reviewer schedule. This adapter never runs an editor, reviewer or model
publication tool. Each phase is journaled (``handoff_journal``) so a restarted
run restores the frozen candidate, re-derives the deterministic commit and
completes the idempotent push and PR binding without a second editor turn.
"""
from __future__ import annotations

from pathlib import Path
import re
import time
from urllib.parse import quote

from .claude_git_operator import ClaudeGitOperator
from .agent_failure import ValidatedAgentResponse
from .git_handoff import CommitIdentity, Handoff, HandoffError, _run, private_trusted_dir, state_base
from .handoff_journal import HandoffJournal, JournalRecord
from .git_publication import OrdinaryPrApi, bind_ordinary_pr, publish_verified, _verified_objects
from .git_transport import default_origin
from .github import unexpected_closing_issue_ids
from .protocol_markers import TrustedBody
from .logging import log
from .protocol import StructuredIssueImplementation, StructuredCoderFollowup
from .workdir_claims import current_claim_owner
from .workdir_guard import validate_test_commands_within_workdir, validate_test_observation_citations_within_workdir
from .workdirs import active_workdir


def preflight(config):
    """Only explicit owner strings; never discover credentials or inherit tokens."""
    from .claude_git_operator import _PinnedFile
    digest, session = config.git_operator_cli_sha256, config.git_operator_session_file
    if (not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)
            or not isinstance(session, str) or not session.startswith('/')
            or '\x00' in session or '\n' in session
            or Path(session).name != '.credentials.json'):
        raise HandoffError('Supply --git-operator-cli-sha256 and --git-operator-session-file '
                           'with an owner-attested native CLI and read-only session path before editing')
    if any(Path(value).is_relative_to(active_workdir(config).resolve())
           for value in (config.claude_cmd, session)):
        raise HandoffError('Owner-attested CLI and session must be outside the editor checkout')
    operator = ClaudeGitOperator(config.claude_cmd, local_executable_sha256=digest,
                                 session_file=Path(session))
    # O_PATH session pin: inspect metadata only, never read the session bytes.
    for path, sha in ((Path(config.claude_cmd), digest), (Path(session), None)):
        pin = _PinnedFile.open(path, digest=sha)
        pin.close()
    from .agent_permissions import validate_split_role_capabilities
    validate_split_role_capabilities(config)
    return operator


def require_unsigned_publication_policy(api, repo, base_branch, head_branch):
    """Refuse before editing when GitHub may require signed commits.

    Operator commits are unsigned by design (signing configuration is never
    read), so a required-signatures rule would only be discovered after the
    push. Anything this read cannot positively prove absent is a refusal.
    """
    def read(endpoint, keys):
        try:
            result = api('GET', f'repos/{repo}/{endpoint}')
        except HandoffError:
            result = None
        if keys is dict and not isinstance(result, dict) or keys is list and not isinstance(result, list):
            raise HandoffError('Cannot verify that signed commits are not required; refusing before editing')
        return result

    for name in dict.fromkeys((base_branch, head_branch)):
        rules = read(f'rules/branches/{quote(name, safe="")}', list)
        if not all(isinstance(rule, dict) for rule in rules):
            raise HandoffError('Cannot verify that signed commits are not required; refusing before editing')
        if any(rule.get('type') == 'required_signatures' for rule in rules):
            raise HandoffError('Signed commits are required by a repository ruleset, which is unsupported')
    branch = read(f'branches/{quote(base_branch, safe="")}', dict)
    if type(branch.get('protected')) is not bool:
        raise HandoffError('Cannot verify that signed commits are not required; refusing before editing')
    if branch['protected']:
        status = read(f'branches/{quote(base_branch, safe="")}/protection/required_signatures', dict)
        if type(status.get('enabled')) is not bool:
            raise HandoffError('Cannot verify that signed commits are not required; refusing before editing')
        if status['enabled']:
            raise HandoffError('Signed commits are required by branch protection, which is unsupported')


class SplitIssueHandoff:
    @classmethod
    def begin(cls, runner, *, config, issue_number, expected_base):
        if len(config.repo.split('/')) != 2:
            raise HandoffError('Ordinary split issue publication requires owner/repository')
        operator = preflight(config)
        owner = current_claim_owner()
        if owner is None or owner.command != 'issue' or owner.number != issue_number:
            raise HandoffError('Direct issue handoff requires the existing issue worktree claim')
        if config.dry_run:
            raise HandoffError('Split issue publication requires a real trusted transport')
        self = cls()
        self.owner = owner
        self.config, self.runner = config, runner
        self.operator = operator
        self.api = OrdinaryPrApi(runner, config)
        actor = self.api('GET', 'user')
        if not isinstance(actor, dict) or type(actor.get('id')) is not int or actor['id'] < 1:
            raise HandoffError('Trusted GitHub publication authentication is required before editing')
        self.issue_number = issue_number
        self.journal = HandoffJournal(config.repo, 'issue', issue_number, active_workdir(config))
        self.recovered = self.journal.load()
        self.handoff_id = owner.run_id
        self.branch = f'reviewloop/issue-{issue_number}-{owner.run_id}'
        if self.recovered is not None:
            self.branch = self.recovered.branch
        require_unsigned_publication_policy(self.api, config.repo, config.base, self.branch)
        self.base_branch = config.base
        self.timestamp = int(time.time())
        self.message = f'Implement issue #{issue_number}\n'
        self.title = f'Implement issue #{issue_number}'
        if self.recovered is not None:
            self._adopt(self.recovered, expected_base, config.base)
        # Private host evidence outside every editor-writable root (never /tmp); preserved on failure.
        trusted = private_trusted_dir(active_workdir(config), 'issue-handoff')
        self._begin_handoff(trusted, expected_base)
        return self

    def _adopt(self, record, base, base_branch):
        """Adopt an interrupted journal only if it still describes this exact target."""
        if record.base != base or record.base_branch != base_branch:
            raise HandoffError('Interrupted handoff journal was recorded against a different base; '
                               f'inspect, then remove {self.journal.path} to discard it before rerunning')
        self.handoff_id, self.branch = record.handoff_id, record.branch
        self.timestamp, self.message = record.timestamp, record.message
        if record.title is not None:
            self.title = record.title

    def _begin_handoff(self, trusted, expected_base):
        record = self.recovered
        self.handoff = Handoff.begin(active_workdir(self.config), trusted,
            CommitIdentity('ReviewLoop', 'reviewloop@localhost', self.timestamp, self.message),
            expected_base=expected_base)
        self.record = record
        if record is None:
            return
        previous = Path(record.trusted)
        if previous.parent.parent != state_base().resolve() or previous.name != 'trusted':
            raise HandoffError('Journaled evidence is outside trusted handoff state')
        log(self.config, f'Recovering interrupted {record.kind} handoff from {record.phase}')
        self.handoff.restore(previous / 'git' / 'objects', record.tree)
        if self.handoff.capture(allowed_new=set(record.new_paths)) != record.tree:
            raise HandoffError('Restored candidate differs from the journaled tree')

    def recovered_response(self, validate):
        """Re-validate the journaled editor response; no new editor turn is run."""
        record = self.recovered
        return ValidatedAgentResponse(text=record.editor_text, session_id=None,
                                      marker_value=validate(record.editor_text),
                                      model_used=record.model_used, role='editor')

    def _phase(self, phase, **changes):
        """Persist progress before any later phase is attempted."""
        self.record = self.journal.write(self.record.at(phase, **changes))
        log(self.config, f'{self._subject()}: {phase} {self.record.sha or self.record.tree}')

    def _subject(self):
        return f'Issue #{self.issue_number}'

    def complete(self):
        """Retire the journal and retained evidence once the checkout is synchronized."""
        self.journal.retire(str(self.handoff.trusted))

    def _freeze(self, parsed, *, editor_text, model_used, kind, number, branch, base_branch,
                allow_no_change=False):
        if self.recovered is not None:
            if set(parsed.editor_new_paths) != set(self.recovered.new_paths):
                raise HandoffError('Journaled editor declaration differs from the recovered candidate')
            return
        self.handoff.capture(allowed_new=set(parsed.editor_new_paths), allow_no_change=allow_no_change)
        if not self.handoff.changed_paths:
            return
        self.record = self.journal.write(JournalRecord(
            repo=self.config.repo, kind=kind, number=number, handoff_id=self.handoff_id,
            branch=branch, base_branch=base_branch, base=self.handoff.base, phase='EDITED',
            tree=self.handoff.tree, new_paths=tuple(sorted(parsed.editor_new_paths)),
            timestamp=self.timestamp, message=self.message, editor_text=editor_text,
            trusted=str(self.handoff.trusted), title=self.title, model_used=model_used).validated())
        log(self.config, f'{self._subject()}: EDITED tree {self.handoff.tree}; Claude committing frozen candidate')

    def _claim(self):
        if current_claim_owner() != self.owner:
            raise HandoffError('Worktree claim changed during handoff')

    def _validate_editor_evidence(self, parsed):
        if (not parsed.tests_run or parsed.test_observation_degradations
                or parsed.architecture_impact is None or parsed.architecture_impact_degradations
                or not parsed.architecture_impact_contract.satisfied):
            raise HandoffError('Valid editor test and architecture evidence is required before publication')
        validate_test_commands_within_workdir(parsed.tests_run, assigned_workdir=self.handoff.checkout)
        validate_test_observation_citations_within_workdir(parsed.test_observations,
                                                         assigned_workdir=self.handoff.checkout)

    def publish(self, parsed, *, editor_text=None, model_used=None):
        self._claim()
        if (not isinstance(parsed, StructuredIssueImplementation) or parsed.pr_number is not None
                or parsed.editor_outcome != 'ready' or parsed.editor_new_paths is None
                or any(d.disposition == 'blocked' for d in parsed.human_requirement_dispositions)):
            raise HandoffError('Editor is blocked or lacks an explicit ready handoff declaration')
        if self.recovered is None:  # Already validated, with live evidence, before journaling.
            self._validate_editor_evidence(parsed)
        # PR prose is editor output too: reject it before advancing any remote.
        TrustedBody.current_untrusted_visible(parsed.summary)
        if (len(parsed.summary.encode()) > 59000 or unexpected_closing_issue_ids(
                parsed.summary, repo=self.config.repo,
                expected_issue_ids=self.config.expected_closing_issue_ids)):
            raise HandoffError('Editor summary violates the ordinary PR closing contract')
        self._freeze(parsed, editor_text=editor_text if self.recovered is None else self.recovered.editor_text,
                     model_used=model_used, kind='issue', number=self.issue_number,
                     branch=self.branch, base_branch=self.base_branch)
        sha = self._commit()
        if self.record.phase == 'EDITED':
            self._phase('COMMITTED', sha=sha)
        self._push(expected_head=None)
        if self.record.phase == 'COMMITTED':
            self._phase('PUSHED')
        number = bind_ordinary_pr(repo=self.config.repo, branch=self.branch, base=self.config.base,
            sha=sha, issue_ids=self.config.expected_closing_issue_ids,
            title=self.title, body=parsed.summary,
            handoff_id=self.handoff_id, api=self.api)
        if self.record.phase == 'PUSHED':
            self._phase('PR_BOUND', pr_number=number)
        return number

    def _commit(self):
        sha = self.operator.run(self.handoff)
        self.handoff.verify(sha)  # Treat even the dedicated adapter's result as untrusted.
        if self.record.sha is not None and sha != self.record.sha:
            raise HandoffError('Recovered commit differs from the journaled commit')
        self.sha = sha
        return sha

    def _push(self, *, expected_head):
        sha = self.sha
        origin = (str(self.config.trusted_local_origin) if self.config.trusted_local_origin
                  else default_origin(self.config.repo, protocol=self.config.trusted_origin_protocol))
        publish_verified(self.handoff, sha, repo=self.config.repo, origin=origin,
                         branch=self.branch, expected_head=expected_head, gh_cmd=self.config.gh_cmd,
                         local_origin=self.config.trusted_local_origin,
                         ssh_identity=self.config.git_ssh_identity)

    def synchronize(self, metadata):
        """Advance the owned checkout without rewriting frozen content."""
        self._claim()
        if (metadata.head_sha != self.sha or metadata.head_branch != self.branch
                or metadata.base_branch != self.config.base):
            raise HandoffError('Authenticated PR head differs from verified issue publication')
        from .run_worktrees import store_lock
        with store_lock(self.handoff.common):
            objects = _verified_objects(self.handoff, self.sha)
            for oid, kind, raw in objects:
                actual = _run(None, self.handoff.checkout,
                              ['hash-object', '-w', '-t', kind.decode(), '--stdin'], data=raw).decode().strip()
                if actual != oid:
                    raise HandoffError('Verified checkout object import failed')
            # All raw objects have been checked before touching the local ref/index.
            _run(None, self.handoff.checkout, ['update-ref', '--no-deref', 'HEAD', self.sha, self.handoff.base])
            _run(None, self.handoff.checkout, ['read-tree', self.sha])
        from .checkout_verification import refresh_after_sync
        refresh_after_sync(self.config, self.runner, self.handoff.checkout)


class SplitPrHandoff(SplitIssueHandoff):
    """One ordinary correction on an authenticated existing PR; never creates PRs.

    The outer loop owns dispatch/ledger/evidence/review. Interrupted pushes
    recover from the phase journal without a second editor turn.
    """
    @classmethod
    def begin(cls, runner, *, config, pr_number, metadata):
        from .git_publication import _branch
        operator = preflight(config)
        owner = current_claim_owner()
        # Direct issue runs enter this loop under their existing outer claim.
        if owner is None or not (owner.command == 'pr' and owner.number == pr_number
                                 or owner.command == 'issue' and config.expected_closing_issue_ids):
            raise HandoffError('PR correction requires the existing worktree claim')
        if config.dry_run or metadata.number != pr_number or metadata.repo != config.repo:
            raise HandoffError('Invalid ordinary PR correction identity')
        _branch(metadata.head_branch or '')
        if not metadata.base_branch or metadata.head_branch == metadata.base_branch:
            raise HandoffError('Invalid ordinary PR correction branch')
        self = cls()
        self.owner, self.config, self.runner = owner, config, runner
        self.operator, self.pr_number = operator, pr_number
        self.branch, self.base_branch = metadata.head_branch, metadata.base_branch
        self.api = OrdinaryPrApi(runner, config)
        self.start_sha = metadata.head_sha
        self._pr(self.start_sha)
        require_unsigned_publication_policy(self.api, config.repo, self.base_branch, self.branch)
        self.issue_number = pr_number
        self.handoff_id = f'pr-{pr_number}'  # Corrections never create a PR; the ID is journal evidence only.
        self.timestamp = int(time.time())
        self.message = f'Address review on PR #{pr_number}\n'
        self.title = None
        self.journal = HandoffJournal(config.repo, 'pr', pr_number, active_workdir(config))
        self.recovered = self.journal.load()
        if self.recovered is not None and self.recovered.sha == self.start_sha:
            # The interrupted correction already reached the PR head: nothing is
            # left to publish, so the ordinary next-round logic takes over.
            self.journal.retire()
            self.recovered = None
        if self.recovered is not None:
            if self.recovered.branch != self.branch:
                raise HandoffError('Interrupted PR handoff journal names a different branch')
            self._adopt(self.recovered, self.start_sha, self.base_branch)
        trusted = private_trusted_dir(active_workdir(config), 'pr-handoff')
        self._begin_handoff(trusted, self.start_sha)
        log(config, f'PR #{pr_number}: Codex editing pinned head {self.start_sha}')
        return self

    def _subject(self):
        return f'PR #{self.pr_number}'

    def _pr(self, sha):
        self._claim()
        pr = self.api('GET', f'repos/{self.config.repo}/pulls/{self.pr_number}')
        try:
            valid = (pr['number'] == self.pr_number and pr['state'] == 'open'
                and pr['head']['sha'] == sha and pr['head']['ref'] == self.branch
                and pr['base']['ref'] == self.base_branch
                and pr['head']['repo']['full_name'].casefold() == self.config.repo.casefold()
                and pr['base']['repo']['full_name'].casefold() == self.config.repo.casefold())
        except (KeyError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise HandoffError('Authenticated ordinary PR identity or head changed during handoff')

    def publish(self, parsed, *, head_change_expected, test_observations, editor_text=None, model_used=None):
        import shlex
        from .local_test_evidence import LocalTestObservation
        self._claim()
        if (not isinstance(parsed, StructuredCoderFollowup) or parsed.editor_outcome != 'ready'
                or parsed.editor_new_paths is None
                or any(d.disposition == 'blocked' for d in parsed.human_requirement_dispositions)):
            raise HandoffError('Editor is blocked or lacks an explicit ready handoff declaration')
        if self.recovered is None:  # Already validated, with live evidence, before journaling.
            self._validate_editor_evidence(parsed)
        # Only live host acquisition receipts count, never model test prose or
        # historical receipts. A failed/incomplete run is a refusal in this slice.
        observations = tuple(test_observations)
        if self.recovered is None and (not observations or any(not isinstance(o, LocalTestObservation)
                or o.outcome != 'passed' or o.cwd != str(self.handoff.checkout)
                for o in observations)):
            raise HandoffError('Passing invocation-local editor test evidence is required')
        commands = {shlex.join(o.command) for o in observations}
        if self.recovered is None and any(
                shlex.join(shlex.split(command)) not in commands for command in parsed.tests_run):
            raise HandoffError('Reported editor tests were not delivered by this invocation')
        TrustedBody.current_untrusted_visible(parsed.summary)
        self._pr(self.start_sha)
        # No-change policy: only the existing host ledger classifier can allow
        # citation-only turns. Capture still checks paths, bytes, modes and Git.
        self._freeze(parsed, editor_text=editor_text if self.recovered is None else self.recovered.editor_text,
                     model_used=model_used, kind='pr', number=self.pr_number, branch=self.branch,
                     base_branch=self.base_branch, allow_no_change=not head_change_expected)
        if not self.handoff.changed_paths:
            self.sha = self.start_sha
            self._pr(self.sha)
            self.handoff._unchanged()
            if self.handoff._snapshot(self.handoff.allowed_new) != self.handoff.snapshot:
                raise HandoffError('Citation-only candidate changed after capture')
            log(self.config, f'PR #{self.pr_number}: citation-only follow-up at {self.sha}; no commit')
            return self.sha
        sha = self._commit()
        if self.record.phase == 'EDITED':
            self._phase('COMMITTED', sha=sha)
        self._pr(self.start_sha)
        self._push(expected_head=self.start_sha)
        if self.record.phase == 'COMMITTED':
            self._phase('PUSHED')
        self._pr(sha)
        log(self.config, f'PR #{self.pr_number}: PUSHED verified {sha} to the same PR')
        return sha

    def synchronize(self, metadata):
        if self.sha == self.start_sha:
            self._pr(self.sha)
            self.handoff._unchanged()
            return
        super().synchronize(metadata)
