"""Unit tests for the managed-CI GitHub API retry helper (#1476).

The helper is the MANAGED_CI_HTTP_RETRY block inlined in the managed-CI
workflows; these tests drive the pinned fixture copy with fake openers.
"""

from __future__ import annotations

import email.message
import http.client
import io
import socket
import urllib.error
import urllib.request

import pytest

from fixtures.managed_ci import http_retry

URL = "https://api.github.com/repos/OWNER/REPO/pulls/7"
TOKEN = "ghs_secret_token_value"


class FakeResponse:
    def __init__(self, body=b"{}", headers=None, read_error=None):
        self._body = body
        self._read_error = read_error
        self.headers = headers if headers is not None else email.message.Message()

    def read(self):
        if self._read_error is not None:
            raise self._read_error
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, body=b"", headers=None):
    message = email.message.Message()
    for key, value in (headers or {}).items():
        message[key] = value
    return urllib.error.HTTPError(URL, code, "error", message, io.BytesIO(body))


class Opener:
    """Replays a script of outcomes: an exception to raise or a response."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, request, timeout=None):
        self.calls.append((request, timeout))
        outcome = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _request(method="GET"):
    return urllib.request.Request(
        URL, method=method, data=b"{}" if method == "POST" else None,
        headers={"Authorization": "Bearer " + TOKEN},
    )


def _call(opener, *, read_json=True, method="GET"):
    sleeps = []
    result = http_retry.github_request(
        _request(method), read_json=read_json, sleep=sleeps.append, opener=opener,
    )
    return result, sleeps


@pytest.mark.parametrize(
    "transient",
    [
        FakeResponse(b""),
        FakeResponse(b"   \n"),
        _http_error(502),
        _http_error(503),
        TimeoutError("timed out"),
        socket.timeout("timed out"),
        urllib.error.URLError("connection refused"),
        ConnectionResetError("reset"),
        http.client.RemoteDisconnected("closed"),
    ],
    ids=["empty", "whitespace", "502", "503", "timeout", "socket-timeout", "urlerror",
         "reset", "remote-disconnected"],
)
def test_transient_read_failure_retries_then_returns_the_payload(transient):
    opener = Opener(transient, FakeResponse(b'{"ok": true}'))
    (payload, _headers), sleeps = _call(opener)
    assert payload == {"ok": True}
    assert len(opener.calls) == 2
    assert len(sleeps) == 1 and 2.0 <= sleeps[0] <= 3.0


def test_incomplete_read_during_body_read_is_retried_inside_the_attempt():
    opener = Opener(
        FakeResponse(read_error=http.client.IncompleteRead(b'{"a"', 10)),
        FakeResponse(b"[1, 2]"),
    )
    (payload, _headers), sleeps = _call(opener)
    assert payload == [1, 2]
    assert len(opener.calls) == 2 and len(sleeps) == 1


def test_truncated_body_that_reaches_json_decoding_is_retried():
    opener = Opener(FakeResponse(b'{"a":'), FakeResponse(b'{"a": 1}'))
    (payload, _headers), _sleeps = _call(opener)
    assert payload == {"a": 1}
    assert len(opener.calls) == 2


def test_persistent_incomplete_read_is_unavailable_after_exactly_four_attempts():
    opener = Opener(FakeResponse(read_error=http.client.IncompleteRead(b"", 5)))
    with pytest.raises(http_retry.GitHubApiUnavailable) as caught:
        _call(opener)
    assert len(opener.calls) == 4
    assert caught.value.attempts == 4
    assert "IncompleteRead" in str(caught.value)


def test_persistent_502_is_unavailable_with_url_and_status_and_no_token():
    opener = Opener(_http_error(502))
    sleeps = []
    with pytest.raises(http_retry.GitHubApiUnavailable) as caught:
        http_retry.github_request(_request(), read_json=True, sleep=sleeps.append, opener=opener)
    message = str(caught.value)
    assert len(opener.calls) == 4 and len(sleeps) == 3
    assert message == "GET " + URL + " failed: HTTP 502 after 4 attempts"
    assert caught.value.status == 502
    assert TOKEN not in message
    assert not isinstance(caught.value, http_retry.GitHubApiRejected)
    # Exponential backoff with jitter: 2, 4, 8 seconds plus up to one second.
    for sleep, base in zip(sleeps, (2.0, 4.0, 8.0)):
        assert base <= sleep <= base + 1.0


def test_exhausted_empty_body_is_unavailable():
    opener = Opener(FakeResponse(b""))
    with pytest.raises(http_retry.GitHubApiUnavailable, match="empty response body after 4 attempts"):
        _call(opener)
    assert len(opener.calls) == 4


@pytest.mark.parametrize("code", [400, 401, 404, 409, 422])
def test_definitive_4xx_is_rejected_after_exactly_one_call(code):
    opener = Opener(_http_error(code, b'{"message": "Bad credentials"}'))
    sleeps = []
    with pytest.raises(http_retry.GitHubApiRejected) as caught:
        http_retry.github_request(_request(), read_json=True, sleep=sleeps.append, opener=opener)
    assert len(opener.calls) == 1 and sleeps == []
    assert not isinstance(caught.value, http_retry.GitHubApiUnavailable)
    assert str(caught.value) == f"GET {URL} failed: HTTP {code} after 1 attempt"
    assert caught.value.status == code
    assert TOKEN not in str(caught.value)


def test_plain_403_is_a_rejection_not_a_rate_limit():
    opener = Opener(_http_error(403, b'{"message": "Resource not accessible by integration"}'))
    with pytest.raises(http_retry.GitHubApiRejected, match="HTTP 403 after 1 attempt"):
        _call(opener)
    assert len(opener.calls) == 1


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("5", 5.0), ("17", 17.0), ("600", 60.0)],
)
def test_429_honours_retry_after_up_to_the_cap(retry_after, expected):
    opener = Opener(_http_error(429, headers={"Retry-After": retry_after}), FakeResponse(b"{}"))
    (payload, _), sleeps = _call(opener)
    assert payload == {}
    assert len(opener.calls) == 2
    # The delay is at least the honoured Retry-After (capped at 60s) and never
    # shorter than the first backoff step.
    assert sleeps[0] >= max(expected, 2.0)
    assert sleeps[0] <= max(expected, 3.0)


@pytest.mark.parametrize(
    ("body", "headers"),
    [
        (b'{"message": "You have exceeded a secondary rate limit."}', {}),
        (b'{"message": "API rate limit exceeded"}', {}),
        (b"", {"x-ratelimit-remaining": "0"}),
        (b"", {"Retry-After": "3"}),
    ],
    ids=["secondary-body", "primary-body", "remaining-zero", "retry-after"],
)
def test_rate_limited_403_is_retried(body, headers):
    opener = Opener(_http_error(403, body, headers), FakeResponse(b"{}"))
    (payload, _), sleeps = _call(opener)
    assert payload == {} and len(opener.calls) == 2 and len(sleeps) == 1


def test_post_mode_does_not_parse_the_body_and_retries_transient_failures():
    opener = Opener(_http_error(502), FakeResponse(b"not json at all"))
    (payload, _headers), sleeps = _call(opener, read_json=False, method="POST")
    assert payload is None
    assert len(opener.calls) == 2 and len(sleeps) == 1


def test_post_mode_rejection_names_the_method():
    opener = Opener(_http_error(422))
    with pytest.raises(http_retry.GitHubApiRejected, match=r"^POST .* failed: HTTP 422 after 1 attempt$"):
        _call(opener, read_json=False, method="POST")


def test_successful_read_returns_headers_for_pagination():
    headers = email.message.Message()
    headers["Link"] = '<https://api.github.com/next>; rel="next"'
    opener = Opener(FakeResponse(b"[]", headers=headers))
    (payload, returned), sleeps = _call(opener)
    assert payload == [] and returned.get("Link") == headers["Link"] and sleeps == []
    assert opener.calls[0][1] == 30


def test_default_opener_and_sleep_resolve_at_call_time(monkeypatch):
    calls = []
    sleeps = []

    def fake_urlopen(request, timeout=None):
        calls.append(request)
        if len(calls) == 1:
            raise _http_error(500)
        return FakeResponse(b'{"x": 1}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(http_retry.time, "sleep", sleeps.append)
    payload, _ = http_retry.github_request(_request(), read_json=True)
    assert payload == {"x": 1} and len(calls) == 2 and len(sleeps) == 1


def test_both_failure_classes_remain_value_errors():
    assert issubclass(http_retry.GitHubApiUnavailable, ValueError)
    assert issubclass(http_retry.GitHubApiRejected, ValueError)


def test_annotation_text_escapes_workflow_command_metacharacters():
    assert http_retry.annotation_text("a%b\r\n::set-output") == "a%25b%0D%0A::set-output"
