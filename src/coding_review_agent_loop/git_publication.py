"""Small host gate for verified commits and ordinary PR binding.

Call under the existing worktree/store claim. The expected head and handoff ID
must come from trusted round state, never from model output. This module does
not persist state or orchestrate rounds. No publication command sees an
operator-controlled Git config, index, refs, hooks or alternates.
"""
from __future__ import annotations

import base64
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from urllib.parse import quote, urlsplit

from . import git_transport as transport
from .git_handoff import Handoff, HandoffError, SHA, _run, ensure_state_base
from .errors import AgentLoopError
from .github import missing_expected_closing_issue_ids, unexpected_closing_issue_ids
from .managed_pr import _api, _json_payload
from .protocol_markers import TrustedBody


def _branch(branch: str):
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./-]{0,199}', branch)
            or any(x in branch for x in ('..', '//', '@{'))
            or any(p.startswith('.') or p.endswith(('.', '.lock')) for p in branch.split('/'))
            or branch.endswith('/') or branch.startswith('refs/')):
        raise HandoffError('Unsafe publication branch')


def _verified_objects(handoff: Handoff, sha: str):
    handoff.verify(sha)
    kind, raw = handoff._object(sha)
    objects = [(sha, kind, raw)]
    pending = [(handoff.tree, b'tree')]
    seen = set()
    while pending:
        oid, kind = pending.pop()
        if oid in seen:
            continue
        seen.add(oid)
        actual, raw = handoff._object(oid)
        if actual != kind or raw != handoff.git('cat-file', kind.decode(), oid):
            raise HandoffError('Verified object changed before import')
        objects.append((oid, kind, raw))
        if kind == b'tree':
            while raw:
                head, raw = raw.split(b'\0', 1)
                child, raw = raw[:20].hex(), raw[20:]
                pending.append((child, b'tree' if head.startswith(b'40000 ') else b'blob'))
    handoff.verify(sha)
    return objects


def publish_verified(handoff: Handoff, sha: str, *, repo: str, origin: str,
                     branch: str, expected_head: str | None, gh_cmd: str | None = None,
                     local_origin: Path | None = None, ssh_identity: Path | None = None) -> str:
    """Import only checked objects, then push SHA with an exact remote lease.

    None means an absent issue branch; a correction requires the pinned parent.
    A retry accepts only that starting head or this exact verified SHA. Import
    happens in a host-owned disposable bare store, leaving source/shared Git
    untouched. local_origin is an explicit host test seam, never agent input.
    """
    _branch(branch)
    if not SHA.fullmatch(sha) or (expected_head is not None and expected_head != handoff.base):
        raise HandoffError('Invalid pinned publication identity')
    url = transport.trusted_url(repo, origin, local_origin=local_origin)
    objects = _verified_objects(handoff, sha)
    # Export pinned trusted history through the existing local execution guard;
    # operator packs/config/alternates are never consulted.
    base_pack = _run(handoff.admin, handoff.checkout, ['pack-objects', '--stdout', '--revs'],
                     data=(handoff.base + '\n').encode(), alternates=handoff.shared)
    with TemporaryDirectory(prefix='reviewloop-publish-',
                            dir=ensure_state_base(handoff.checkout)) as raw:
        store = Path(raw)
        env = transport._transport_env(url, gh_cmd, ssh_identity=ssh_identity,
                                      forbidden_roots=(handoff.checkout, handoff.trusted,
                                                       handoff.operator_context().private))
        if url.startswith('https://'):
            if not gh_cmd:
                raise HandoffError('Trusted GitHub publication authentication is required')
            token = transport._token_from_gh(gh_cmd, handoff.checkout, urlsplit(url).hostname)
            header = 'AUTHORIZATION: basic ' + base64.b64encode(f'x-access-token:{token}'.encode()).decode()
            env.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0=f'http.{url}.extraheader',
                       GIT_CONFIG_VALUE_0=header)
        # Never discover a surrounding repository or use init templates.
        def git(*args, data=None):
            return transport._private_git(tuple(args), cwd=store, env=env, input_bytes=data,
                                          ssh=url.startswith(('ssh://', 'git@')))
        git('init', '--bare', '--template=', '-q')
        git('index-pack', '--stdin', data=base_pack)
        for oid, kind, content in objects:
            imported = git('hash-object', '-w', '-t', kind.decode(), '--stdin', data=content).decode().strip()
            if imported != oid:
                raise HandoffError('Verified raw object import failed')
        ref = f'refs/heads/{branch}'
        def remote_head():
            rows = git('ls-remote', '--refs', url, ref).decode('ascii').splitlines()
            if not rows:
                return None
            if len(rows) != 1 or rows[0].split() != [rows[0][:40], ref] or not SHA.fullmatch(rows[0][:40]):
                raise HandoffError('Invalid remote publication head')
            return rows[0][:40]
        handoff.verify(sha)  # Recheck the claim/candidate immediately at the publication gate.
        current = remote_head()
        if current != sha:
            if current != expected_head:
                raise HandoffError('Publication refused a foreign remote head')
            # No +refspec, broad force, branch-name expansion or implicit HEAD.
            git('push', '--porcelain', '--no-verify', '--recurse-submodules=no',
                f'--force-with-lease={ref}:{expected_head or ""}', url, f'{sha}:{ref}')
        if remote_head() != sha:
            raise HandoffError('Remote publication head did not verify')
    return sha


class OrdinaryPrApi:
    """Trusted Runner/gh adapter; tests may supply an in-memory API instead."""
    def __init__(self, runner, config):
        if config.dry_run:
            raise HandoffError('Ordinary publication cannot use a dry-run transport')
        self.runner, self.config = runner, config

    def __call__(self, method, endpoint, payload=None):
        result = _api(self.runner, config=self.config, method=method, endpoint=endpoint,
                      payload=payload, check=False)
        if result.returncode:
            raise HandoffError('Trusted GitHub PR operation failed')
        return _json_payload(result, operation='binding an ordinary PR')


def bind_ordinary_pr(*, repo: str, branch: str, base: str, sha: str, issue_ids: tuple[int, ...],
                     title: str, body: str, handoff_id: str, api) -> int:
    """Create or recover one ordinary same-repository PR for a recorded handoff.

    A failed POST is reconciled by listing again, never blindly repeated. A
    closed, duplicate or foreign PR blocks. Corrections should pass their
    original handoff ID and updated verified SHA to bind the same open PR.
    The caller persists the returned number through existing provenance and
    closing-contract writers before entering the independent review loop.
    """
    _branch(branch)
    _branch(base)
    transport.default_origin(repo)  # Validate the host repository identity.
    if len(repo.split('/')) != 2:
        raise HandoffError('Ordinary PR binding currently requires owner/repository')
    if (branch == base or not SHA.fullmatch(sha) or not issue_ids
            or any(type(i) is not int or i < 1 for i in issue_ids)
            or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', handoff_id)
            or not title.strip() or len(title) > 256 or '\n' in title or '\r' in title):
        raise HandoffError('Invalid ordinary PR contract')
    TrustedBody.current_untrusted_visible(title)
    TrustedBody.current_untrusted_visible(body)
    if len(body.encode()) > 60000 or any(ord(c) < 32 for c in title):
        raise HandoffError('Unsupported ordinary PR text')
    identity = f'ReviewLoop handoff: {handoff_id}'
    if 'ReviewLoop handoff:' in body:
        raise HandoffError('Untrusted PR text contains a handoff identity')
    rendered = body.rstrip() + '\n\n' + '\n'.join(f'Closes #{i}' for i in sorted(set(issue_ids)))
    rendered += '\n\n' + identity + '\n'
    if unexpected_closing_issue_ids(rendered, repo=repo, expected_issue_ids=issue_ids):
        raise HandoffError('PR text closes issues outside the expected contract')
    actor = api('GET', 'user')
    if not isinstance(actor, dict) or type(actor.get('id')) is not int or actor['id'] < 1:
        raise HandoffError('Trusted GitHub publication authentication is required')
    owner = repo.split('/')[-2]
    endpoint = f'repos/{repo}/pulls?state=all&head={quote(owner + ":" + branch, safe="")}&per_page=100'
    def lookup():
        prs = api('GET', endpoint)
        if not isinstance(prs, list) or len(prs) > 1:
            raise HandoffError('Ambiguous ordinary PR binding')
        return prs[0] if prs else None
    def validate(pr):
        if not isinstance(pr, dict) or type(pr.get('number')) is not int or pr['number'] < 1:
            raise HandoffError('Invalid ordinary PR response')
        pr = api('GET', f'repos/{repo}/pulls/{pr["number"]}')
        try:
            valid = (pr['state'] == 'open' and pr['head']['sha'] == sha
                     and pr['head']['ref'] == branch and pr['base']['ref'] == base
                     and pr['head']['repo']['full_name'].casefold() == repo.casefold()
                     and pr['base']['repo']['full_name'].casefold() == repo.casefold()
                     and pr['user']['id'] == actor['id']
                     and pr['body'].splitlines().count(identity) == 1)
            TrustedBody.current_untrusted_visible(pr['body'])
            valid = valid and not missing_expected_closing_issue_ids(pr['body'], repo=repo, expected_issue_ids=issue_ids)
            valid = valid and not unexpected_closing_issue_ids(pr['body'], repo=repo, expected_issue_ids=issue_ids)
        except (KeyError, TypeError, AttributeError) as exc:
            raise HandoffError('Incomplete ordinary PR identity') from exc
        if not valid:
            raise HandoffError('Ordinary PR differs from the recorded publication contract')
        return pr['number']
    existing = lookup()
    if existing is not None:
        return validate(existing)
    try:
        created = api('POST', f'repos/{repo}/pulls', dict(title=title.strip(), body=rendered,
                                                       head=branch, base=base, draft=False))
    except (AgentLoopError, OSError, TimeoutError):
        existing = lookup()
        if existing is None:
            raise HandoffError('Ordinary PR creation failed; retry binding the recorded handoff') from None
        return validate(existing)
    # Re-list even after a successful POST, detecting competing/duplicate PRs.
    existing = lookup()
    if existing is None or not isinstance(created, dict) or existing.get('number') != created.get('number'):
        raise HandoffError('Ordinary PR creation did not reconcile')
    return validate(existing)
