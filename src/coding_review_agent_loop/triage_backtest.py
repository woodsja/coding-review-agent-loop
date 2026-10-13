"""Offline triage backtest (#1435, stage 3 of #1430).

``agent-loop triage backtest`` reads stage-1 ``history export`` JSON files and
nothing else.  It derives versioned outcome labels from measured counts only
(never from the route that was chosen), builds one leak-free triage request per
labeled issue at cutoff ``T`` = the row's ``run_started_at``, sends it to an
operator-supplied provider command on stdin, and strictly validates the answer
on stdout.  The report covers per-route precision/recall over issue-partitioned
calibration and holdout splits, coverage, a majority baseline and a
hybrid-routing confidence-threshold finding.  The finding is a measurement
only: nothing here changes routing, posts anything or calls GitHub.
"""

from __future__ import annotations

import datetime
import errno as errno_module
import hashlib
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace as dataclasses_replace
from pathlib import Path

from .errors import AgentLoopError
from .history_export import COVERAGE_COMPLETE, EVIDENCE_VERIFIED, SCHEMA_VERSION as HISTORY_SCHEMA_VERSION
from .response_validation import TRIAGE_ROUTES, parse_triage_response
from .triage import (
    TRIAGE_REASON_SIGNALS,
    USABLE_SNAPSHOT_STATUSES,
    extract_signals,
    load_triage_history,
    retrieve_similar_issues,
)

REQUEST_KIND = "triage_backtest_request"
REPORT_KIND = "triage_backtest_report"
REQUEST_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
LABEL_SPEC_VERSION = 1
ROUTE_ANSWER_KIND = "issue_triage"
PROBABILITY_ANSWER_KIND = "issue_triage_probability"

ROUTE_DISCUSS = "discuss-first"
ROUTE_PLAN_FIRST = "plan-first"
ROUTE_DIRECT = "direct"
ROUTES = (ROUTE_DISCUSS, ROUTE_PLAN_FIRST, ROUTE_DIRECT)
assert set(ROUTES) == set(TRIAGE_ROUTES)
ABSTAIN = "abstain"
# Majority-baseline tie order.
BASELINE_ORDER = (ROUTE_PLAN_FIRST, ROUTE_DISCUSS, ROUTE_DIRECT)

SPLIT_CALIBRATION = "calibration"
SPLIT_HOLDOUT = "holdout"
SPLITS = (SPLIT_CALIBRATION, SPLIT_HOLDOUT)
VIEW_ABSTAIN_EXCLUDED = "abstain_excluded"
VIEW_AS_DEPLOYED = "as_deployed"

DEFAULT_PROVIDER_TIMEOUT = 60
MIN_PROVIDER_TIMEOUT = 1
MAX_PROVIDER_TIMEOUT = 600
MAX_STDOUT_BYTES = 64 * 1024
MAX_STDERR_BYTES = 16 * 1024
_READER_GRACE_SECONDS = 1.0
_GROUP_TERM_GRACE_SECONDS = 1.0
_LOG_STDERR_CHARS = 200
_REASON_CHARS = 200
TOKEN_VARIABLES = ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN")

DEFAULT_HYBRID_TARGET_ACCURACY = 0.9
DEFAULT_HYBRID_MIN_SUPPORT = 10
HYBRID_THRESHOLDS = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
HYBRID_NOTE = (
    "This finding is a measurement only; agent-loop routing is unchanged and "
    "triage stays advisory."
)

# Label reasons (no label).
LABEL_OUTCOME_INCOMPLETE = "outcome-incomplete"
LABEL_COVERAGE_PARTIAL = "infrastructure-coverage-partial"
LABEL_COUNT_UNAVAILABLE = "count-unavailable"
LABEL_NO_DELIVERY = "no-delivery-outcome"
# Reasons a labeled row is not sent as a target.
EXCLUDED_NO_SNAPSHOT = "no-usable-snapshot"
EXCLUDED_NO_RUN_START = "no-run-start"
EXCLUDED_SNAPSHOT_UNDATED = "snapshot-undated"
EXCLUDED_SNAPSHOT_AFTER_RUN_START = "snapshot-after-run-start"

# Answer shapes.
SHAPE_ROUTE = "route"
SHAPE_PROBABILITY = "probability"
SHAPE_INVALID = "invalid"

# Abstain reasons of a well-formed answer that commits to no route.
REASON_UNSURE = "unsure"
REASON_PROVIDER_ABSTAINED = "provider-abstained"
REASON_PROBABILITY_TIE = "probability-tie"
VALID_ABSTAIN_REASONS = (REASON_UNSURE, REASON_PROVIDER_ABSTAINED, REASON_PROBABILITY_TIE)
# Invalid or failed-call reasons; prefixed forms carry a detail suffix.
REASON_TIMEOUT = "timeout"
REASON_STDOUT_TOO_LARGE = "stdout-too-large"
REASON_NOT_UTF8 = "not-utf8"
REASON_LAUNCH_FAILED_PREFIX = "launch-failed:"
REASON_EXIT_PREFIX = "exit-"
REASON_INVALID_ANSWER_PREFIX = "invalid-answer: "
INVALID_REASONS = (REASON_TIMEOUT, REASON_STDOUT_TOO_LARGE, REASON_NOT_UTF8)
INVALID_REASON_PREFIXES = (REASON_LAUNCH_FAILED_PREFIX, REASON_EXIT_PREFIX, REASON_INVALID_ANSWER_PREFIX)

# Answer classes and their total as_deployed mapping.  ``None`` keeps the
# answer's own prediction.  A well-formed answer that commits to no route runs
# as plan-first, mirroring the live unsure mapping; an invalid or failed call
# stays abstain, mirroring live triage posting nothing.
CLASS_VALID_ROUTE = "valid-route"
CLASS_VALID_ABSTAIN = "valid-abstain"
CLASS_INVALID = "invalid"
AS_DEPLOYED_BY_CLASS: Mapping[str, str | None] = {
    CLASS_VALID_ROUTE: None,
    CLASS_VALID_ABSTAIN: ROUTE_PLAN_FIRST,
    CLASS_INVALID: ABSTAIN,
}

_LABEL_COUNT_COLUMNS = ("human_decision_stops", "step_backs", "plan_rounds", "pr_rounds")
_LABEL_COLUMNS = (*_LABEL_COUNT_COLUMNS, "merged")


def classify_reason(reason: str | None) -> str:
    """The answer class of an abstain reason; ``None`` is a committed answer."""
    if reason is None:
        return CLASS_VALID_ROUTE
    if reason in VALID_ABSTAIN_REASONS:
        return CLASS_VALID_ABSTAIN
    if reason in INVALID_REASONS or reason.startswith(INVALID_REASON_PREFIXES):
        return CLASS_INVALID
    raise AgentLoopError(f"Unclassified triage backtest reason {reason!r}.")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


def _canonical(row: Mapping[str, object]) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False)


def _row_key(row: Mapping[str, object]) -> tuple[str, int] | None:
    repo, number = row.get("repo"), row.get("issue_number")
    if not isinstance(repo, str) or not isinstance(number, int) or isinstance(number, bool):
        return None
    return repo.lower(), number


@dataclass(frozen=True)
class Corpus:
    files: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]


def load_backtest_corpus(paths: Sequence[Path]) -> Corpus:
    """Stage-1 rows keyed by (repo, issue); conflicting duplicates are refused."""
    if not paths:
        raise AgentLoopError("Triage backtest needs at least one --history file.")
    by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for path in paths:
        for row in load_triage_history(Path(path)):
            if row.get("schema_version") != HISTORY_SCHEMA_VERSION:
                continue
            key = _row_key(row)
            if key is None:
                continue
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = row
            elif _canonical(existing) != _canonical(row):
                raise AgentLoopError(
                    f"Triage backtest history has conflicting rows for {row['repo']}#{key[1]}."
                )
    rows = tuple(by_key[key] for key in sorted(by_key))
    return Corpus(files=tuple(Path(path).name for path in paths), rows=rows)


# ---------------------------------------------------------------------------
# Outcome labels
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LabelSpec:
    discuss_min_plan_rounds: int = 4
    direct_max_pr_rounds: int = 2
    version: int = LABEL_SPEC_VERSION

    def validate(self) -> "LabelSpec":
        for name in ("discuss_min_plan_rounds", "direct_max_pr_rounds"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise AgentLoopError(f"--{name.replace('_', '-')} must be a positive integer.")
        if self.discuss_min_plan_rounds < 2:
            raise AgentLoopError("--discuss-min-plan-rounds must be at least 2.")
        return self


@dataclass(frozen=True)
class LabelResult:
    label: str | None
    reason: str


def _verified(row: Mapping[str, object], column: str) -> tuple[bool, object]:
    pair = row.get(column)
    if not isinstance(pair, Mapping) or pair.get("evidence") != EVIDENCE_VERIFIED:
        return False, None
    value = pair.get("value")
    if column == "merged":
        return isinstance(value, bool), value
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return False, None
    return True, value


def derive_outcome_label(row: Mapping[str, object], spec: LabelSpec) -> LabelResult:
    """Label from measured counts only; route, route_taken and terminal_states are never read."""
    if row.get("outcome_complete") is not True:
        return LabelResult(None, LABEL_OUTCOME_INCOMPLETE)
    if row.get("infrastructure_coverage") != COVERAGE_COMPLETE:
        return LabelResult(None, LABEL_COVERAGE_PARTIAL)
    values: dict[str, object] = {}
    for column in _LABEL_COLUMNS:
        ok, value = _verified(row, column)
        if not ok:
            return LabelResult(None, f"{LABEL_COUNT_UNAVAILABLE}:{column}")
        values[column] = value
    if (
        values["human_decision_stops"] >= 1  # type: ignore[operator]
        or values["step_backs"] >= 1  # type: ignore[operator]
        or values["plan_rounds"] >= spec.discuss_min_plan_rounds  # type: ignore[operator]
    ):
        return LabelResult(ROUTE_DISCUSS, "stop, step-back or many plan rounds")
    if values["merged"] is not True:
        return LabelResult(None, LABEL_NO_DELIVERY)
    if values["plan_rounds"] <= 1 and values["pr_rounds"] <= spec.direct_max_pr_rounds:  # type: ignore[operator]
        return LabelResult(ROUTE_DIRECT, "merged with at most one plan round and few PR rounds")
    return LabelResult(ROUTE_PLAN_FIRST, "merged after more planning or PR rounds")


# ---------------------------------------------------------------------------
# Targets, leak-free requests and splits
# ---------------------------------------------------------------------------


def _parse_time(value: object) -> datetime.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=datetime.timezone.utc)


@dataclass(frozen=True)
class Target:
    repo: str
    issue_number: int
    cutoff: datetime.datetime
    label: str
    request: Mapping[str, object]

    @property
    def key(self) -> str:
        return f"{self.repo}#{self.issue_number}"


@dataclass
class CorpusSelection:
    targets: list[Target] = field(default_factory=list)
    labeled: int = 0
    considered: int = 0
    unlabeled_by_reason: Counter = field(default_factory=Counter)
    excluded_targets_by_reason: Counter = field(default_factory=Counter)


def target_exclusion(row: Mapping[str, object]) -> str | None:
    """Why a labeled row is not sent; its own snapshot must be dated at or before T."""
    snapshot = row.get("snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("status") not in USABLE_SNAPSHOT_STATUSES
        or not isinstance(snapshot.get("title"), str)
        or not isinstance(snapshot.get("body"), str)
    ):
        return EXCLUDED_NO_SNAPSHOT
    cutoff = _parse_time(row.get("run_started_at"))
    if cutoff is None:
        return EXCLUDED_NO_RUN_START
    as_of = _parse_time(snapshot.get("as_of"))
    if as_of is None:
        return EXCLUDED_SNAPSHOT_UNDATED
    if as_of > cutoff:
        return EXCLUDED_SNAPSHOT_AFTER_RUN_START
    return None


def build_request(row: Mapping[str, object], rows: Sequence[Mapping[str, object]]) -> dict:
    """The provider request: run-start text, signals and similar issues before T only."""
    snapshot = row["snapshot"]
    assert isinstance(snapshot, Mapping)
    cutoff = _parse_time(row["run_started_at"])
    assert cutoff is not None
    title, body = str(snapshot["title"]), str(snapshot["body"])
    similar = retrieve_similar_issues(
        rows, repo=str(row["repo"]), issue_number=int(row["issue_number"]),  # type: ignore[arg-type]
        cutoff=cutoff, query=f"{title}\n{body}",
    )
    signals = extract_signals(title, body, similar=similar)
    similar_payload = []
    for item in similar:
        item_snapshot = item.row.get("snapshot")
        assert isinstance(item_snapshot, Mapping)
        similar_payload.append({
            "issue_number": item.issue_number,
            "score": item.score,
            "title": item_snapshot.get("title"),
            "body": item_snapshot.get("body"),
            "snapshot_status": item_snapshot.get("status"),
            "snapshot_as_of": item_snapshot.get("as_of"),
            "outcome": {key: value for key, value in item.row.items() if key != "snapshot"},
        })
    return {
        "schema_version": REQUEST_SCHEMA_VERSION,
        "kind": REQUEST_KIND,
        "repo": row["repo"],
        "issue_number": row["issue_number"],
        "cutoff": cutoff.isoformat(),
        "title": title,
        "body": body,
        "signals": [
            {"name": signal_.name, "route_hint": signal_.route_hint, "evidence": list(signal_.evidence)}
            for signal_ in signals
        ],
        "similar_issues": similar_payload,
        "routes": list(ROUTES),
        "reason_signals": list(TRIAGE_REASON_SIGNALS),
        "answer_shapes": [ROUTE_ANSWER_KIND, PROBABILITY_ANSWER_KIND],
    }


def select_targets(corpus: Corpus, spec: LabelSpec, *, repo: str | None = None) -> CorpusSelection:
    """Label rows and build requests; ``repo`` restricts the targets, not retrieval."""
    selection = CorpusSelection()
    for row in corpus.rows:
        if repo is not None and str(row["repo"]).lower() != repo.lower():
            continue
        selection.considered += 1
        result = derive_outcome_label(row, spec)
        if result.label is None:
            selection.unlabeled_by_reason[result.reason] += 1
            continue
        selection.labeled += 1
        excluded = target_exclusion(row)
        if excluded is not None:
            selection.excluded_targets_by_reason[excluded] += 1
            continue
        cutoff = _parse_time(row["run_started_at"])
        assert cutoff is not None
        selection.targets.append(Target(
            repo=str(row["repo"]), issue_number=int(row["issue_number"]),  # type: ignore[arg-type]
            cutoff=cutoff, label=result.label, request=build_request(row, corpus.rows),
        ))
    selection.targets.sort(key=lambda target: (target.cutoff, target.repo.lower(), target.issue_number))
    return selection


def split_targets(targets: Sequence[Target]) -> dict[str, tuple[Target, ...]]:
    """Chronological issue partition: the first ceil(n/2) calibrate, the rest hold out."""
    ordered = sorted(targets, key=lambda target: (target.cutoff, target.repo.lower(), target.issue_number))
    boundary = math.ceil(len(ordered) / 2)
    return {SPLIT_CALIBRATION: tuple(ordered[:boundary]), SPLIT_HOLDOUT: tuple(ordered[boundary:])}


# ---------------------------------------------------------------------------
# Provider hook
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderResult:
    stdout: bytes
    stderr: bytes
    returncode: int | None
    failure: str | None
    stderr_truncated: bool


def parse_provider_command(command: str) -> tuple[str, ...]:
    """argv for ``command`` without a shell; a relative path is resolved here."""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise AgentLoopError(f"--provider-command is not a valid command line: {exc}") from exc
    if not argv:
        raise AgentLoopError("--provider-command must name a command.")
    if os.sep in argv[0] and not os.path.isabs(argv[0]):
        # The provider runs in a fresh temp directory, so resolve against ours.
        argv[0] = os.path.abspath(argv[0])
    return tuple(argv)


def _provider_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in TOKEN_VARIABLES}


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def run_provider(
    argv: Sequence[str],
    request_bytes: bytes,
    *,
    timeout: float,
    max_stdout: int = MAX_STDOUT_BYTES,
    max_stderr: int = MAX_STDERR_BYTES,
) -> ProviderResult:
    """Run one provider call with bounded output and whole-group cleanup on every path."""
    with tempfile.TemporaryDirectory(prefix="agent-loop-triage-provider-") as workdir:
        try:
            proc = subprocess.Popen(
                list(argv),
                cwd=workdir,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_provider_env(),
                start_new_session=True,
            )
        except OSError as exc:
            name = errno_module.errorcode.get(exc.errno or 0, "UNKNOWN")
            return ProviderResult(b"", b"", None, f"{REASON_LAUNCH_FAILED_PREFIX}{name}", False)
        pgid = proc.pid
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        stdout_overflow = threading.Event()
        stderr_state = {"retained": 0, "truncated": False}

        def write_stdin() -> None:
            try:
                proc.stdin.write(request_bytes)  # type: ignore[union-attr]
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    proc.stdin.close()  # type: ignore[union-attr]
                except OSError:
                    pass

        def read_stdout() -> None:
            total = 0
            while True:
                chunk = proc.stdout.read1(65536)  # type: ignore[union-attr]
                if not chunk:
                    return
                total += len(chunk)
                if total > max_stdout:
                    stdout_overflow.set()
                    return
                stdout_chunks.append(chunk)

        def read_stderr() -> None:
            while True:
                chunk = proc.stderr.read1(65536)  # type: ignore[union-attr]
                if not chunk:
                    return
                room = max_stderr - stderr_state["retained"]
                if room > 0:
                    stderr_chunks.append(chunk[:room])
                    stderr_state["retained"] += min(room, len(chunk))
                if len(chunk) > room:
                    stderr_state["truncated"] = True

        threads = [
            threading.Thread(target=write_stdin, daemon=True),
            threading.Thread(target=read_stdout, daemon=True),
            threading.Thread(target=read_stderr, daemon=True),
        ]
        for thread in threads:
            thread.start()
        failure: str | None = None
        deadline = time.monotonic() + timeout
        try:
            while True:
                if stdout_overflow.is_set():
                    failure = REASON_STDOUT_TOO_LARGE
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = REASON_TIMEOUT
                    break
                try:
                    proc.wait(timeout=min(0.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            # (1) The direct child is gone or is killed now.
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            # (2) Readers drain what was already written.
            grace_end = time.monotonic() + _READER_GRACE_SECONDS
            for thread in threads[1:]:
                thread.join(timeout=max(0.0, grace_end - time.monotonic()))
            # (3) Every group member is terminated, then killed; the kill also
            # closes any pipe a grandchild still holds.
            if _group_alive(pgid):
                _signal_group(pgid, signal.SIGTERM)
                term_end = time.monotonic() + _GROUP_TERM_GRACE_SECONDS
                while _group_alive(pgid) and time.monotonic() < term_end:
                    time.sleep(0.02)
                _signal_group(pgid, signal.SIGKILL)
            # (4) Join every helper before the temp directory is removed.
            for thread in threads:
                thread.join(timeout=5)
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
        if failure is None and stdout_overflow.is_set():
            failure = REASON_STDOUT_TOO_LARGE
        return ProviderResult(
            stdout=b"".join(stdout_chunks),
            stderr=b"".join(stderr_chunks),
            returncode=proc.returncode,
            failure=failure,
            stderr_truncated=bool(stderr_state["truncated"]),
        )


# ---------------------------------------------------------------------------
# Answer validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    answer_shape: str
    prediction: str
    reason: str | None
    confidence: float | None = None
    max_probability: float | None = None
    stderr_truncated: bool = False

    @property
    def as_deployed(self) -> str:
        mapped = AS_DEPLOYED_BY_CLASS[classify_reason(self.reason)]
        return self.prediction if mapped is None else mapped


def _invalid(message: str) -> Prediction:
    return Prediction(SHAPE_INVALID, ABSTAIN, (REASON_INVALID_ANSWER_PREFIX + message)[:_REASON_CHARS])


def _is_probability(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        # Range first: an exact int/float comparison never converts a huge
        # integer to float, so it cannot overflow.
        and 0 <= value <= 1 and math.isfinite(value)
    )


def _validate_probability(payload: Mapping[str, object], suffix: str) -> Prediction:
    if suffix:
        return _invalid("probability answer must be one JSON object with nothing after it.")
    required = {"schema_version", "kind", "probabilities", "abstain"}
    if set(payload) != required:
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        return _invalid(f"probability answer fields differ (missing {missing}, unknown {unknown}).")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        return _invalid("probability answer schema_version must be 1.")
    probabilities = payload["probabilities"]
    if not isinstance(probabilities, dict) or set(probabilities) != set(ROUTES):
        return _invalid("probabilities must have exactly discuss-first, plan-first and direct.")
    if not all(_is_probability(value) for value in probabilities.values()):
        return _invalid("each probability must be a finite number in [0, 1].")
    if abs(sum(probabilities.values()) - 1.0) > 1e-3:
        return _invalid("probabilities must sum to 1 within 0.001.")
    if not isinstance(payload["abstain"], bool):
        return _invalid("abstain must be a boolean.")
    top = max(probabilities.values())
    if payload["abstain"]:
        return Prediction(SHAPE_PROBABILITY, ABSTAIN, REASON_PROVIDER_ABSTAINED, max_probability=float(top))
    leaders = [route for route in ROUTES if probabilities[route] == top]
    if len(leaders) > 1:
        return Prediction(SHAPE_PROBABILITY, ABSTAIN, REASON_PROBABILITY_TIE, max_probability=float(top))
    return Prediction(
        SHAPE_PROBABILITY, leaders[0], None, confidence=float(top), max_probability=float(top)
    )


def validate_answer(text: str) -> Prediction:
    """A route answer, a probability-with-abstain answer, or an invalid abstain.

    Any failure while decoding or checking the bounded provider output,
    including decoder resource limits (deep nesting, oversized numbers), is
    an invalid answer for this target and never aborts the backtest.
    """
    try:
        return _validate_answer(text)
    except (ValueError, RecursionError, OverflowError, MemoryError, TypeError) as exc:
        return _invalid(f"answer could not be decoded ({type(exc).__name__}).")


def _validate_answer(text: str) -> Prediction:
    stripped = text.strip()
    try:
        payload, end = json.JSONDecoder().raw_decode(stripped)
    except json.JSONDecodeError as exc:
        return _invalid(f"not valid JSON: {exc.msg}.")
    if not isinstance(payload, dict):
        return _invalid("answer must be a JSON object.")
    kind = payload.get("kind")
    if kind == ROUTE_ANSWER_KIND:
        try:
            answer = parse_triage_response(text, allowed_signals=TRIAGE_REASON_SIGNALS)
        except AgentLoopError as exc:
            return _invalid(str(exc))
        if answer.unsure:
            return Prediction(SHAPE_ROUTE, ABSTAIN, REASON_UNSURE)
        return Prediction(SHAPE_ROUTE, answer.route, None)
    if kind == PROBABILITY_ANSWER_KIND:
        return _validate_probability(payload, stripped[end:].strip())
    return _invalid(f"unrecognized answer kind {str(kind)[:40]!r}.")


def prediction_from_result(result: ProviderResult) -> Prediction:
    def failed(reason: str) -> Prediction:
        return Prediction(SHAPE_INVALID, ABSTAIN, reason, stderr_truncated=result.stderr_truncated)

    if result.failure is not None:
        return failed(result.failure)
    if result.returncode != 0:
        return failed(f"{REASON_EXIT_PREFIX}{result.returncode}")
    try:
        text = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return failed(REASON_NOT_UTF8)
    prediction = validate_answer(text)
    if result.stderr_truncated:
        prediction = dataclasses_replace(prediction, stderr_truncated=True)
    return prediction


ProviderCall = Callable[[Target], Prediction]


def command_provider(argv: Sequence[str], *, timeout: float) -> ProviderCall:
    def call(target: Target) -> Prediction:
        request_bytes = json.dumps(target.request, sort_keys=True, ensure_ascii=False).encode("utf-8")
        result = run_provider(argv, request_bytes, timeout=timeout)
        prediction = prediction_from_result(result)
        if prediction.reason is not None and classify_reason(prediction.reason) == CLASS_INVALID:
            stderr = result.stderr.decode("utf-8", "replace")[:_LOG_STDERR_CHARS]
            print(
                f"agent-loop: triage backtest {target.key}: provider answer counted as abstain "
                f"({prediction.reason}); stderr: {' '.join(stderr.split()) or '(empty)'}",
                file=sys.stderr,
            )
        return prediction

    return call


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scored:
    target: Target
    split: str
    prediction: Prediction


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 4)


def _route_metrics(pairs: Sequence[tuple[str, str]]) -> dict:
    """``pairs`` are (label, predicted) with ``abstain`` for no prediction."""
    total = len(pairs)
    covered = [(label, predicted) for label, predicted in pairs if predicted != ABSTAIN]
    correct = sum(1 for label, predicted in covered if label == predicted)
    per_route = {}
    for route in ROUTES:
        support = sum(1 for label, _ in pairs if label == route)
        predicted = sum(1 for _, value in covered if value == route)
        tp = sum(1 for label, value in covered if value == route and label == route)
        per_route[route] = {
            "support": support, "predicted": predicted, "tp": tp,
            "precision": _ratio(tp, predicted), "recall": _ratio(tp, support),
        }
    return {
        "total": total,
        "covered": len(covered),
        "coverage": _ratio(len(covered), total),
        "accuracy_on_covered": _ratio(correct, len(covered)),
        "per_route": per_route,
    }


def compute_metrics(scored: Sequence[Scored]) -> dict:
    """Per-route precision/recall, coverage and accuracy for both views and every split."""
    groups = {
        SPLIT_CALIBRATION: [item for item in scored if item.split == SPLIT_CALIBRATION],
        SPLIT_HOLDOUT: [item for item in scored if item.split == SPLIT_HOLDOUT],
        "all": list(scored),
    }
    metrics: dict[str, dict] = {}
    for view in (VIEW_ABSTAIN_EXCLUDED, VIEW_AS_DEPLOYED):
        metrics[view] = {}
        for name, items in groups.items():
            pairs = [
                (
                    item.target.label,
                    item.prediction.prediction if view == VIEW_ABSTAIN_EXCLUDED else item.prediction.as_deployed,
                )
                for item in items
            ]
            result = _route_metrics(pairs)
            result["invalid_count"] = sum(1 for item in items if item.prediction.answer_shape == SHAPE_INVALID)
            metrics[view][name] = result
    return metrics


def majority_baseline(scored: Sequence[Scored]) -> dict:
    calibration = [item.target.label for item in scored if item.split == SPLIT_CALIBRATION]
    if not calibration:
        return {"route": None, "reason": "insufficient-data", "splits": None}
    counts = Counter(calibration)
    top = max(counts.values())
    route = next(candidate for candidate in BASELINE_ORDER if counts.get(candidate) == top)
    splits = {}
    for name in (*SPLITS, "all"):
        labels = [item.target.label for item in scored if name == "all" or item.split == name]
        result = _route_metrics([(label, route) for label in labels])
        splits[name] = {
            "accuracy": result["accuracy_on_covered"], "per_route": result["per_route"],
        }
    return {"route": route, "reason": None, "splits": splits}


def hybrid_finding(scored: Sequence[Scored], *, target_accuracy: float, min_support: int) -> dict:
    """Report-only confidence-threshold finding over committed probability answers."""
    base = {
        "target_accuracy": target_accuracy, "min_support": min_support,
        "thresholds": list(HYBRID_THRESHOLDS), "routing_changed": False, "note": HYBRID_NOTE,
    }
    probability_answers = [item for item in scored if item.prediction.answer_shape == SHAPE_PROBABILITY]
    if not probability_answers:
        return {
            **base, "status": "not-applicable", "decided_by": "rule-1-no-probability-answers",
            "reason": "the provider returned no probabilities", "threshold": None,
            "committed_answers": None, "sweep": [],
        }
    totals = {name: sum(1 for item in scored if item.split == name) for name in SPLITS}
    committed = {
        name: [
            item for item in probability_answers
            if item.split == name and item.prediction.prediction != ABSTAIN
            and item.prediction.confidence is not None
        ]
        for name in SPLITS
    }

    def counts(name: str, threshold: float) -> tuple[int, int]:
        confident = [item for item in committed[name] if item.prediction.confidence >= threshold]  # type: ignore[operator]
        correct = sum(1 for item in confident if item.prediction.prediction == item.target.label)
        return len(confident), correct

    def at(name: str, threshold: float) -> dict:
        confident, correct = counts(name, threshold)
        return {
            "confident": confident,
            "accuracy": _ratio(correct, confident),
            "coverage": _ratio(confident, totals[name]),
        }

    sweep = [
        {"threshold": threshold, **{name: at(name, threshold) for name in SPLITS}}
        for threshold in HYBRID_THRESHOLDS
    ]
    result = {
        **base, "committed_answers": {name: len(committed[name]) for name in SPLITS}, "sweep": sweep,
    }

    def qualifies(name: str, threshold: float) -> bool:
        # Decide on the unrounded accuracy; rounding is for the report only.
        confident, correct = counts(name, threshold)
        return confident >= min_support and confident > 0 and correct / confident >= target_accuracy

    if any(len(committed[name]) < min_support for name in SPLITS):
        return {
            **result, "status": "insufficient-data", "decided_by": "rule-2-min-support",
            "reason": f"a split has fewer than {min_support} committed probability answers",
            "threshold": None,
        }
    chosen = next((entry for entry in sweep if qualifies(SPLIT_CALIBRATION, entry["threshold"])), None)
    if chosen is None:
        return {
            **result, "status": "not-supported", "decided_by": "rule-3-calibration-search",
            "reason": "no threshold meets the target accuracy and support on calibration",
            "threshold": None,
        }
    if qualifies(SPLIT_HOLDOUT, chosen["threshold"]):
        return {
            **result, "status": "supported", "decided_by": "rule-4-holdout-check",
            "reason": "the calibration threshold also meets the target on holdout",
            "threshold": chosen["threshold"],
        }
    return {
        **result, "status": "not-supported", "decided_by": "rule-4-holdout-check",
        "reason": "the calibration threshold misses the target on holdout",
        "threshold": chosen["threshold"],
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def build_report(
    corpus: Corpus,
    *,
    provider: ProviderCall,
    provider_info: Mapping[str, object],
    spec: LabelSpec,
    repo: str | None = None,
    hybrid_target_accuracy: float = DEFAULT_HYBRID_TARGET_ACCURACY,
    hybrid_min_support: int = DEFAULT_HYBRID_MIN_SUPPORT,
) -> dict:
    selection = select_targets(corpus, spec, repo=repo)
    splits = split_targets(selection.targets)
    scored: list[Scored] = []
    for name in SPLITS:
        for target in splits[name]:
            scored.append(Scored(target, name, provider(target)))
    targets_payload = [
        {
            "repo": item.target.repo,
            "issue_number": item.target.issue_number,
            "cutoff": item.target.cutoff.isoformat(),
            "split": item.split,
            "label": item.target.label,
            "similar_issue_numbers": [
                similar["issue_number"] for similar in item.target.request["similar_issues"]  # type: ignore[union-attr]
            ],
            "answer_shape": item.prediction.answer_shape,
            "prediction": item.prediction.prediction,
            "confidence": _round(item.prediction.confidence),
            "max_probability": _round(item.prediction.max_probability),
            "reason": item.prediction.reason,
            "answer_class": classify_reason(item.prediction.reason),
            "as_deployed": item.prediction.as_deployed,
            "stderr_truncated": item.prediction.stderr_truncated,
        }
        for item in scored
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": REPORT_KIND,
        "label_spec": {
            "version": spec.version,
            "discuss_min_plan_rounds": spec.discuss_min_plan_rounds,
            "direct_max_pr_rounds": spec.direct_max_pr_rounds,
        },
        "provider": dict(provider_info),
        "corpus": {
            "files": list(corpus.files),
            "repo_filter": repo,
            "rows": len(corpus.rows),
            "considered": selection.considered,
            "labeled": selection.labeled,
            "unlabeled_by_reason": dict(sorted(selection.unlabeled_by_reason.items())),
            "excluded_targets_by_reason": dict(sorted(selection.excluded_targets_by_reason.items())),
            "targets": len(selection.targets),
        },
        "splits": {name: [target.key for target in splits[name]] for name in SPLITS},
        "targets": targets_payload,
        "metrics": compute_metrics(scored),
        "baseline": majority_baseline(scored),
        "hybrid": hybrid_finding(
            scored, target_accuracy=hybrid_target_accuracy, min_support=hybrid_min_support
        ),
    }


def render_report_json(report: Mapping[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _cell(value: object) -> str:
    if value is None:
        return "n/a"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_report_markdown(report: Mapping[str, object]) -> str:
    corpus = report["corpus"]
    spec = report["label_spec"]
    provider = report["provider"]
    hybrid = report["hybrid"]
    baseline = report["baseline"]
    assert isinstance(corpus, Mapping) and isinstance(spec, Mapping) and isinstance(provider, Mapping)
    assert isinstance(hybrid, Mapping) and isinstance(baseline, Mapping)
    lines = [
        "# Triage backtest report",
        "",
        f"- Provider: `{provider['executable']}` (argv sha256 `{provider['argv_sha256']}`, "
        f"timeout {provider['timeout_seconds']}s)",
        f"- Label spec v{spec['version']}: discuss_min_plan_rounds={spec['discuss_min_plan_rounds']}, "
        f"direct_max_pr_rounds={spec['direct_max_pr_rounds']}",
        f"- Corpus: {corpus['rows']} rows from {', '.join(corpus['files'])}; "  # type: ignore[arg-type]
        f"{corpus['labeled']} labeled, {corpus['targets']} targets",
    ]
    for title, key in (("Unlabeled", "unlabeled_by_reason"), ("Excluded targets", "excluded_targets_by_reason")):
        counts = corpus[key]
        assert isinstance(counts, Mapping)
        rendered = ", ".join(f"{reason}: {count}" for reason, count in counts.items()) or "none"
        lines.append(f"- {title}: {rendered}")
    metrics = report["metrics"]
    assert isinstance(metrics, Mapping)
    for view, view_metrics in metrics.items():
        lines += ["", f"## Metrics ({view})", "",
                  "| Split | Total | Coverage | Accuracy on covered | Invalid | Route | Support | Predicted | Precision | Recall |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for split, values in view_metrics.items():
            for route, per in values["per_route"].items():
                lines.append(
                    f"| {split} | {values['total']} | {_cell(values['coverage'])} | "
                    f"{_cell(values['accuracy_on_covered'])} | {values['invalid_count']} | {route} | "
                    f"{per['support']} | {per['predicted']} | {_cell(per['precision'])} | {_cell(per['recall'])} |"
                )
    lines += ["", "## Majority baseline", ""]
    if baseline["route"] is None:
        lines.append(f"- n/a ({baseline['reason']})")
    else:
        lines += [f"- Route: {baseline['route']}", "",
                  "| Split | Accuracy | Route | Support | Predicted | Precision | Recall |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for split, values in baseline["splits"].items():  # type: ignore[union-attr]
            for route, per in values["per_route"].items():
                lines.append(
                    f"| {split} | {_cell(values['accuracy'])} | {route} | {per['support']} | "
                    f"{per['predicted']} | {_cell(per['precision'])} | {_cell(per['recall'])} |"
                )
    lines += [
        "", "## Hybrid routing finding", "",
        f"- Status: **{hybrid['status']}** (decided by {hybrid['decided_by']}; {hybrid['reason']})",
        f"- Threshold: {_cell(hybrid['threshold'])}; target accuracy {hybrid['target_accuracy']}, "
        f"min support {hybrid['min_support']}",
        f"- {hybrid['note']}",
    ]
    sweep = hybrid["sweep"]
    if sweep:
        lines += ["", "| Threshold | Calibration confident | Calibration accuracy | Calibration coverage "
                  "| Holdout confident | Holdout accuracy | Holdout coverage |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for entry in sweep:  # type: ignore[union-attr]
            cal, hold = entry[SPLIT_CALIBRATION], entry[SPLIT_HOLDOUT]
            lines.append(
                f"| {entry['threshold']} | {cal['confident']} | {_cell(cal['accuracy'])} | "
                f"{_cell(cal['coverage'])} | {hold['confident']} | {_cell(hold['accuracy'])} | "
                f"{_cell(hold['coverage'])} |"
            )
    lines += ["", "## Targets", "",
              "| Issue | Split | Label | Shape | Prediction | Confidence | Reason | As deployed |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for target in report["targets"]:  # type: ignore[union-attr]
        lines.append(
            f"| {_cell(target['repo'])}#{target['issue_number']} | {target['split']} | {target['label']} | "
            f"{target['answer_shape']} | {target['prediction']} | {_cell(target['confidence'])} | "
            f"{_cell(target['reason'])} | {target['as_deployed']} |"
        )
    return "\n".join(lines) + "\n"


def write_atomically(path: Path, text: str) -> None:
    """Write ``text`` through a sibling temp file and ``os.replace``."""
    directory = path.parent if str(path.parent) else Path(".")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def provider_info(command_argv: Sequence[str], *, timeout: float) -> dict:
    return {
        "kind": "command",
        "executable": os.path.basename(command_argv[0]),
        "argv_sha256": hashlib.sha256(json.dumps(list(command_argv)).encode("utf-8")).hexdigest(),
        "timeout_seconds": timeout,
        "max_stdout_bytes": MAX_STDOUT_BYTES,
        "max_stderr_bytes": MAX_STDERR_BYTES,
    }


def validate_backtest_options(
    *, provider_timeout: float, hybrid_target_accuracy: float, hybrid_min_support: int
) -> None:
    if not MIN_PROVIDER_TIMEOUT <= provider_timeout <= MAX_PROVIDER_TIMEOUT:
        raise AgentLoopError(
            f"--provider-timeout must be between {MIN_PROVIDER_TIMEOUT} and {MAX_PROVIDER_TIMEOUT} seconds."
        )
    if not 0.0 < hybrid_target_accuracy <= 1.0:
        raise AgentLoopError("--hybrid-target-accuracy must be in (0, 1].")
    if isinstance(hybrid_min_support, bool) or hybrid_min_support < 1:
        raise AgentLoopError("--hybrid-min-support must be a positive integer.")


def run_backtest(
    history_paths: Sequence[Path],
    *,
    provider_command: str,
    output: Path,
    output_format: str = "json",
    repo: str | None = None,
    provider_timeout: float = DEFAULT_PROVIDER_TIMEOUT,
    spec: LabelSpec = LabelSpec(),
    hybrid_target_accuracy: float = DEFAULT_HYBRID_TARGET_ACCURACY,
    hybrid_min_support: int = DEFAULT_HYBRID_MIN_SUPPORT,
) -> dict:
    """Validate every option, load the corpus, run the provider and write the report."""
    if output_format not in {"json", "markdown"}:
        raise AgentLoopError("--format must be json or markdown.")
    spec.validate()
    validate_backtest_options(
        provider_timeout=provider_timeout,
        hybrid_target_accuracy=hybrid_target_accuracy,
        hybrid_min_support=hybrid_min_support,
    )
    argv = parse_provider_command(provider_command)
    if not output.parent.is_dir() and str(output.parent):
        raise AgentLoopError(f"--output directory {output.parent} does not exist.")
    corpus = load_backtest_corpus(history_paths)
    report = build_report(
        corpus,
        provider=command_provider(argv, timeout=provider_timeout),
        provider_info=provider_info(shlex.split(provider_command), timeout=provider_timeout),
        spec=spec,
        repo=repo,
        hybrid_target_accuracy=hybrid_target_accuracy,
        hybrid_min_support=hybrid_min_support,
    )
    rendered = render_report_markdown(report) if output_format == "markdown" else render_report_json(report)
    write_atomically(output, rendered)
    return report


def summary_line(report: Mapping[str, object]) -> str:
    corpus, hybrid = report["corpus"], report["hybrid"]
    assert isinstance(corpus, Mapping) and isinstance(hybrid, Mapping)
    metrics = report["metrics"]
    assert isinstance(metrics, Mapping)
    overall = metrics[VIEW_ABSTAIN_EXCLUDED]["all"]
    return (
        f"triage backtest: {corpus['targets']} targets, coverage {_cell(overall['coverage'])}, "
        f"accuracy on covered {_cell(overall['accuracy_on_covered'])}, hybrid {hybrid['status']} "
        "(routing unchanged)"
    )
