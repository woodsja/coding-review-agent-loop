"""One-way direct-issue transaction; CLI stays blocked pending durable recovery.

The issue loop owns the claim, response validation, provenance and reviewer
schedule. This adapter never runs an editor, reviewer or model publication tool.
"""
from __future__ import annotations

from pathlib import Path
import re
import tempfile
import time

from .claude_git_operator import ClaudeGitOperator
from .git_handoff import CommitIdentity, Handoff, HandoffError, _run
from .git_publication import OrdinaryPrApi, bind_ordinary_pr, publish_verified, _verified_objects
from .git_transport import default_origin
from .github import unexpected_closing_issue_ids
from .protocol_markers import TrustedBody
from .logging import log
from .protocol import StructuredIssueImplementation
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
        self.handoff_id = owner.run_id
        self.branch = f'reviewloop/issue-{issue_number}-{owner.run_id}'
        # Private host evidence outside all model workspaces; preserved on failure.
        trusted = Path(tempfile.mkdtemp(prefix='reviewloop-issue-handoff-')) / 'trusted'
        self.handoff = Handoff.begin(active_workdir(config), trusted,
            CommitIdentity('ReviewLoop', 'reviewloop@localhost', int(time.time()),
                           f'Implement issue #{issue_number}\n'), expected_base=expected_base)
        log(config, f'Issue #{issue_number}: Codex editing pinned base {expected_base}')
        return self

    def _claim(self):
        if current_claim_owner() != self.owner:
            raise HandoffError('Issue worktree claim changed during handoff')

    def publish(self, parsed):
        self._claim()
        if (not isinstance(parsed, StructuredIssueImplementation) or parsed.pr_number is not None
                or parsed.editor_outcome != 'ready' or parsed.editor_new_paths is None
                or any(d.disposition == 'blocked' for d in parsed.human_requirement_dispositions)):
            raise HandoffError('Editor is blocked or lacks an explicit ready handoff declaration')
        if (not parsed.tests_run or parsed.test_observation_degradations
                or parsed.architecture_impact is None or parsed.architecture_impact_degradations
                or not parsed.architecture_impact_contract.satisfied):
            raise HandoffError('Valid editor test and architecture evidence is required before publication')
        validate_test_commands_within_workdir(parsed.tests_run, assigned_workdir=self.handoff.checkout)
        validate_test_observation_citations_within_workdir(parsed.test_observations,
                                                         assigned_workdir=self.handoff.checkout)
        # PR prose is editor output too: reject it before advancing any remote.
        TrustedBody.current_untrusted_visible(parsed.summary)
        if (len(parsed.summary.encode()) > 59000 or unexpected_closing_issue_ids(
                parsed.summary, repo=self.config.repo,
                expected_issue_ids=self.config.expected_closing_issue_ids)):
            raise HandoffError('Editor summary violates the ordinary PR closing contract')
        self.handoff.capture(allowed_new=set(parsed.editor_new_paths))
        log(self.config, f'Issue #{self.issue_number}: EDITED tree {self.handoff.tree}; Claude committing frozen candidate')
        sha = self.operator.run(self.handoff)
        self.handoff.verify(sha)  # Treat even the dedicated adapter's result as untrusted.
        self.sha = sha
        log(self.config, f'Issue #{self.issue_number}: COMMITTED verified {sha}')
        origin = (str(self.config.trusted_local_origin) if self.config.trusted_local_origin
                  else default_origin(self.config.repo, protocol=self.config.trusted_origin_protocol))
        publish_verified(self.handoff, sha, repo=self.config.repo, origin=origin,
                         branch=self.branch, expected_head=None, gh_cmd=self.config.gh_cmd,
                         local_origin=self.config.trusted_local_origin,
                         ssh_identity=self.config.git_ssh_identity)
        log(self.config, f'Issue #{self.issue_number}: PUSHED verified {sha}')
        number = bind_ordinary_pr(repo=self.config.repo, branch=self.branch, base=self.config.base,
            sha=sha, issue_ids=self.config.expected_closing_issue_ids,
            title=f'Implement issue #{self.issue_number}', body=parsed.summary,
            handoff_id=self.handoff_id, api=self.api)
        log(self.config, f'Issue #{self.issue_number}: PR_BOUND #{number} at {sha}')
        return number

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
