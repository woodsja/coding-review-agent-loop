"""Extraction-bounded copy of the managed-CI GitHub API retry helper.

Source repository: wwind123/coding-review-agent-loop
Source block SHA-256: 30014fec185eb3249a867f195c0bdb71be0723b2431ff8e5a8ee531c5e1e8fad
Source path: .github/workflows/managed-ci.yml
Extraction boundary: ``GitHubApiUnavailable`` through ``github_request``.
The same block is inlined, byte-identical, in five workflow locations: the
validate and publish steps of managed-ci.yml, the validate step of
managed-ci-validate.yml, and the verify jobs listing and publish steps of
managed-ci-publish.yml. The contract tests compare every copy to this fixture
and drive the helper with fake openers without making a GitHub request.
"""


# BEGIN MANAGED_CI_HTTP_RETRY
# Source repository: wwind123/coding-review-agent-loop
# Source path: .github/workflows/managed-ci.yml
# Extraction boundary: GitHubApiUnavailable through github_request().
# Bounded GitHub API retry for every managed-CI read and status POST. Each
# attempt opens the request, reads the body and parses it inside one try
# boundary. Only exhausted transient failures (network, timeout, transport,
# 5xx, 429, a rate-limited 403, an empty or truncated JSON body) raise
# GitHubApiUnavailable. A definitive answer (401, 404, 422, a 403 that is not a
# rate limit, any other 4xx) raises GitHubApiRejected after exactly one call.
# Messages name the method, URL and last status; they never carry the token.
import http.client
import json
import random
import socket
import time
import urllib.error
import urllib.request

class GitHubApiFailure(ValueError):
    def __init__(self, method, url, detail, attempts, status=None):
        self.method = method
        self.url = url
        self.detail = detail
        self.attempts = attempts
        self.status = status
        super().__init__(
            method + ' ' + url + ' failed: ' + detail + ' after ' + str(attempts)
            + (' attempt' if attempts == 1 else ' attempts')
        )

class GitHubApiUnavailable(GitHubApiFailure):
    """Retryable failures were exhausted: infrastructure, not a verdict."""

class GitHubApiRejected(GitHubApiFailure):
    """GitHub gave a definitive non-retryable answer."""

class _GitHubApiEmptyBody(Exception):
    pass

def annotation_text(value):
    text = str(value)
    return text.replace('%', '%25').replace('\r', '%0D').replace('\n', '%0A')

def _retry_after_seconds(headers):
    value = headers.get('Retry-After') if headers is not None else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None

def _is_rate_limited(error, retry_after):
    headers = error.headers
    if retry_after is not None:
        return True
    if headers is not None and str(headers.get('x-ratelimit-remaining', '')).strip() == '0':
        return True
    try:
        body = error.read() or b''
    except Exception:
        body = b''
    if isinstance(body, bytes):
        body = body.decode('utf-8', 'replace')
    return 'rate limit' in str(body).lower()

def github_request(request, *, read_json, attempts=4, timeout=30, sleep=None, opener=None):
    method = request.get_method()
    url = request.full_url
    detail = 'no attempt made'
    status = None
    for attempt in range(1, attempts + 1):
        retry_after = None
        try:
            open_url = opener if opener is not None else urllib.request.urlopen
            with open_url(request, timeout=timeout) as response:
                body = response.read()
                headers = getattr(response, 'headers', None)
            if not read_json:
                return None, headers
            if not body or not body.strip():
                raise _GitHubApiEmptyBody('empty response body')
            return json.loads(body), headers
        except urllib.error.HTTPError as exc:
            status = exc.code
            detail = 'HTTP ' + str(exc.code)
            retry_after = _retry_after_seconds(exc.headers)
            retryable = exc.code >= 500 or exc.code == 429 or (
                exc.code == 403 and _is_rate_limited(exc, retry_after)
            )
            if not retryable:
                raise GitHubApiRejected(method, url, detail, attempt, status) from exc
        except _GitHubApiEmptyBody:
            status = None
            detail = 'empty response body'
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            status = None
            detail = 'truncated or invalid JSON body (' + type(exc).__name__ + ')'
        except (
            urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError,
            http.client.HTTPException, OSError,
        ) as exc:
            status = None
            detail = type(exc).__name__
        if attempt < attempts:
            delay = min(30.0, 2.0 * 2 ** (attempt - 1)) + random.uniform(0.0, 1.0)
            if retry_after is not None:
                delay = max(delay, min(60.0, retry_after))
            (sleep if sleep is not None else time.sleep)(delay)
    raise GitHubApiUnavailable(method, url, detail, attempts, status)
# END MANAGED_CI_HTTP_RETRY
