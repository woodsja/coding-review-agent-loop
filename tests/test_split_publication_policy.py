"""Unsigned operator commits must be refused before editing when signatures are required."""
import pytest

from coding_review_agent_loop import split_issue_handoff as split
from coding_review_agent_loop.git_handoff import HandoffError

REAL_POLICY = split.require_unsigned_publication_policy


def api_for(**responses):
    calls = []

    def api(method, endpoint, payload=None):
        assert method == 'GET' and payload is None
        calls.append(endpoint)
        result = responses.get(endpoint.removeprefix('repos/OWNER/REPO/'), HandoffError('404'))
        if isinstance(result, Exception):
            raise result
        return result
    api.calls = calls
    return api


def check(api, head='reviewloop/issue-1-run1'):
    REAL_POLICY(api, 'OWNER/REPO', 'main', head)


def test_unprotected_branch_without_rules_is_allowed():
    api = api_for(**{'rules/branches/main': [], f'rules/branches/reviewloop%2Fissue-1-run1': [],
                     'branches/main': {'protected': False}})
    check(api)
    assert not any('required_signatures' in call for call in api.calls)


def test_protected_branch_without_required_signatures_is_allowed():
    check(api_for(**{'rules/branches/main': [{'type': 'pull_request'}],
                     'rules/branches/reviewloop%2Fissue-1-run1': [],
                     'branches/main': {'protected': True},
                     'branches/main/protection/required_signatures': {'enabled': False}}))


@pytest.mark.parametrize('responses,match', [
    ({'rules/branches/main': [{'type': 'required_signatures'}]}, 'ruleset'),
    ({'rules/branches/main': [], 'rules/branches/reviewloop%2Fissue-1-run1': [{'type': 'required_signatures'}]},
     'ruleset'),
    ({'rules/branches/main': [], 'rules/branches/reviewloop%2Fissue-1-run1': [],
      'branches/main': {'protected': True},
      'branches/main/protection/required_signatures': {'enabled': True}}, 'branch protection'),
])
def test_required_signatures_are_refused(responses, match):
    with pytest.raises(HandoffError, match=match):
        check(api_for(**responses))


@pytest.mark.parametrize('responses', [
    {},                                                                   # rules unreadable
    {'rules/branches/main': {}, 'rules/branches/reviewloop%2Fissue-1-run1': []},   # wrong shape
    {'rules/branches/main': [None], 'rules/branches/reviewloop%2Fissue-1-run1': []},
    {'rules/branches/main': [], 'rules/branches/reviewloop%2Fissue-1-run1': [],
     'branches/main': {}},                                                # protection state unknown
    {'rules/branches/main': [], 'rules/branches/reviewloop%2Fissue-1-run1': [],
     'branches/main': {'protected': True}},                               # signature state unreadable
    {'rules/branches/main': [], 'rules/branches/reviewloop%2Fissue-1-run1': [],
     'branches/main': {'protected': True},
     'branches/main/protection/required_signatures': {'enabled': 'no'}},
])
def test_unverifiable_policy_fails_closed(responses):
    with pytest.raises(HandoffError, match='Cannot verify'):
        check(api_for(**responses))
