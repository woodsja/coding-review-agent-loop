"""Offline triage backtest (#1435, stage 3 of #1430).

Every test uses fixture corpora and a local fixture provider script; no live
model, GitHub or external service is called.
"""

from __future__ import annotations

import datetime
import json
import os
import shlex
import stat
import sys
import time
from pathlib import Path

import pytest

import coding_review_agent_loop.triage_backtest as backtest
from coding_review_agent_loop.cli import main
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.history_export import (
    COVERAGE_COMPLETE,
    COVERAGE_PARTIAL,
    EVIDENCE_UNAVAILABLE,
    EVIDENCE_VERIFIED,
    SCHEMA_VERSION,
    SNAPSHOT_NEVER_EDITED,
    SNAPSHOT_UNAVAILABLE,
)
from coding_review_agent_loop.triage_backtest import (
    ABSTAIN,
    LabelSpec,
    Prediction,
    Scored,
    Target,
    build_report,
    classify_reason,
    command_provider,
    compute_metrics,
    derive_outcome_label,
    hybrid_finding,
    load_backtest_corpus,
    run_backtest,
    run_provider,
    select_targets,
    validate_answer,
)

REPO = "OWNER/REPO"
UTC = datetime.timezone.utc
TITLE = "normalize degraded payloads in the export"


def _pair(value, evidence=EVIDENCE_VERIFIED):
    return {"value": value, "evidence": evidence}


def _row(number, *, repo=REPO, run_started="2026-09-01T00:00:00+00:00", as_of="2026-08-31T00:00:00+00:00",
         status=SNAPSHOT_NEVER_EDITED, title=TITLE, body="degraded payload normalization",
         plan_rounds=1, pr_rounds=1, step_backs=0, stops=0, merged=True, complete=True,
         coverage=COVERAGE_COMPLETE, available="2026-09-02T00:00:00+00:00", **overrides):
    row = {
        "schema_version": SCHEMA_VERSION, "repo": repo, "issue_number": number,
        "route": "direct", "route_taken": ["direct"], "terminal_states": ["pr-merged"],
        "plan_rounds": _pair(plan_rounds), "step_backs": _pair(step_backs),
        "human_decision_stops": _pair(stops), "pr_rounds": _pair(pr_rounds),
        "merged": _pair(merged), "final_plan_size": _pair(None, EVIDENCE_UNAVAILABLE),
        "infrastructure_coverage": coverage, "infrastructure_failures": [],
        "outcome_complete": complete, "outcome_available_at": available,
        "run_started_at": run_started,
        "snapshot": {"status": status, "title": title, "body": body, "as_of": as_of},
    }
    row.update(overrides)
    return row


def _write_history(path: Path, rows) -> Path:
    path.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "repo": REPO, "rows": rows}), encoding="utf-8")
    return path


PROVIDER_SCRIPT = r'''
import json, os, subprocess, sys, time
text = sys.stdin.read()
request = json.loads(text)
number = request["issue_number"]
echo = os.environ.get("BT_ECHO_DIR")
if echo:
    with open(os.path.join(echo, f"{number}.json"), "w") as handle:
        handle.write(text)
    with open(os.path.join(echo, f"{number}.cwd"), "w") as handle:
        handle.write(os.getcwd())
    tokens = [name for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN")
              if name in os.environ]
    with open(os.path.join(echo, f"{number}.env"), "w") as handle:
        json.dump(tokens, handle)
with open(os.environ["BT_BEHAVIORS"]) as handle:
    behavior = json.load(handle).get(str(number), "route:plan-first")
kind, _, arg = behavior.partition(":")

def route(name, unsure=False):
    return json.dumps({"schema_version": 1, "kind": "issue_triage", "route": name, "unsure": unsure,
                       "reasons": [{"signal": "no-signal", "explanation": "fixture"}]})

def prob(values, abstain=False, **extra):
    a, b, c = (float(v) for v in values.split(","))
    payload = {"schema_version": 1, "kind": "issue_triage_probability",
               "probabilities": {"discuss-first": a, "plan-first": b, "direct": c}, "abstain": abstain}
    payload.update(extra)
    return json.dumps(payload)

def spawn(redirect):
    streams = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if redirect else {}
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], **streams)
    with open(os.path.join(os.environ["BT_PID_DIR"], f"{number}.pid"), "w") as handle:
        handle.write(str(child.pid))

if kind == "route":
    print(route(arg))
elif kind == "unsure":
    print(route("plan-first", unsure=True))
elif kind == "prob":
    print(prob(arg))
elif kind == "prob-abstain":
    print(prob(arg, abstain=True))
elif kind == "extra-key":
    print(prob("0.2,0.2,0.6", note="x"))
elif kind == "malformed":
    print("{not json")
elif kind == "exit":
    print(route("direct"))
    sys.exit(int(arg))
elif kind == "timeout":
    spawn(redirect=False)
    time.sleep(60)
elif kind == "big-stdout":
    sys.stdout.write("x" * (70 * 1024))
elif kind == "noisy-stderr":
    sys.stderr.write("e" * (40 * 1024))
    sys.stderr.flush()
    print(route("direct"))
elif kind == "grandchild":
    spawn(redirect=True)
    print(route("direct"))
    sys.stdout.flush()
    sys.exit(int(arg))
'''


@pytest.fixture
def provider(tmp_path, monkeypatch):
    """A fixture provider whose answer per issue comes from a behavior file."""
    root = tmp_path / "provider"
    root.mkdir()
    script = root / "provider.py"
    script.write_text(PROVIDER_SCRIPT, encoding="utf-8")
    behaviors = root / "behaviors.json"
    echo, pids = root / "echo", root / "pids"
    echo.mkdir()
    pids.mkdir()
    monkeypatch.setenv("BT_BEHAVIORS", str(behaviors))
    monkeypatch.setenv("BT_PID_DIR", str(pids))

    class Provider:
        command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"
        argv = (sys.executable, str(script))
        echo_dir = echo
        pid_dir = pids

        @staticmethod
        def behave(mapping, *, echo_requests=True):
            behaviors.write_text(json.dumps({str(key): value for key, value in mapping.items()}), encoding="utf-8")
            if echo_requests:
                monkeypatch.setenv("BT_ECHO_DIR", str(echo))
            else:
                monkeypatch.delenv("BT_ECHO_DIR", raising=False)

    return Provider


def _pid_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except (OSError, IndexError):
            return True
        if state in {"Z", "X"}:
            return True
        time.sleep(0.05)
    return False


def _report(tmp_path, rows, provider, behaviors, *, echo_requests=True, name="report.json", **kwargs):
    provider.behave(behaviors, echo_requests=echo_requests)
    history = _write_history(tmp_path / f"history-{name}.json", rows)
    out_dir = tmp_path / f"out-{name}"
    out_dir.mkdir(exist_ok=True)
    output = out_dir / name
    kwargs.setdefault("provider_timeout", 5)
    report = run_backtest([history], provider_command=provider.command, output=output, **kwargs)
    return report, output


def _by_issue(report):
    return {target["issue_number"]: target for target in report["targets"]}


# ---------------------------------------------------------------------------
# label-measured-only
# ---------------------------------------------------------------------------

LABEL_TABLE = [
    (_row(1, stops=1, merged=False), "discuss-first", None),
    (_row(2, step_backs=1), "discuss-first", None),
    (_row(3, plan_rounds=4), "discuss-first", None),
    (_row(4, plan_rounds=0, pr_rounds=1), "direct", None),
    (_row(5, plan_rounds=1, pr_rounds=2), "direct", None),
    (_row(6, plan_rounds=2, pr_rounds=1), "plan-first", None),
    (_row(7, plan_rounds=1, pr_rounds=3), "plan-first", None),
    (_row(8, plan_rounds=2, merged=False), None, "no-delivery-outcome"),
    (_row(9, stops=1, coverage=COVERAGE_PARTIAL), None, "infrastructure-coverage-partial"),
    (_row(10, pr_rounds=None) | {"pr_rounds": _pair(None, EVIDENCE_UNAVAILABLE)}, None,
     "count-unavailable:pr_rounds"),
    (_row(11, stops=1, complete=False), None, "outcome-incomplete"),
    (_row(12) | {"plan_rounds": _pair(True)}, None, "count-unavailable:plan_rounds"),
    (_row(13) | {"human_decision_stops": _pair(0, EVIDENCE_UNAVAILABLE)}, None,
     "count-unavailable:human_decision_stops"),
]


def test_labels_follow_the_table_and_ignore_route_and_terminal_states():
    spec = LabelSpec()
    for row, label, reason in LABEL_TABLE:
        result = derive_outcome_label(row, spec)
        assert result.label == label, row["issue_number"]
        if reason is not None:
            assert result.reason == reason
        mutated = {
            **row, "route": "discuss", "route_taken": ["discuss", "plan-first"],
            "terminal_states": ["human-decision-stop", "plan-approved"],
            "triage": {"route": "discuss-first"},
        }
        assert derive_outcome_label(mutated, spec) == result
        stripped = {key: value for key, value in row.items() if key not in {"route", "route_taken", "terminal_states"}}
        assert derive_outcome_label(stripped, spec) == result


def test_label_thresholds_are_overridable_and_validated():
    spec = LabelSpec(discuss_min_plan_rounds=2, direct_max_pr_rounds=3)
    assert derive_outcome_label(_row(6, plan_rounds=2), spec).label == "discuss-first"
    assert derive_outcome_label(_row(7, plan_rounds=1, pr_rounds=3), spec).label == "direct"
    with pytest.raises(AgentLoopError, match="at least 2"):
        LabelSpec(discuss_min_plan_rounds=1).validate()
    with pytest.raises(AgentLoopError, match="positive integer"):
        LabelSpec(direct_max_pr_rounds=0).validate()


def test_report_echoes_the_spec_and_counts_unlabeled_rows(tmp_path, provider):
    rows = [row for row, _, _ in LABEL_TABLE]
    report, _ = _report(tmp_path, rows, provider, {})
    assert report["label_spec"] == {"version": 1, "discuss_min_plan_rounds": 4, "direct_max_pr_rounds": 2}
    assert report["corpus"]["labeled"] == 7
    assert report["corpus"]["unlabeled_by_reason"] == {
        "count-unavailable:human_decision_stops": 1, "count-unavailable:plan_rounds": 1,
        "count-unavailable:pr_rounds": 1, "infrastructure-coverage-partial": 1,
        "no-delivery-outcome": 1, "outcome-incomplete": 1,
    }


# ---------------------------------------------------------------------------
# split-by-issue
# ---------------------------------------------------------------------------


def test_identical_duplicates_collapse_and_conflicts_are_refused_before_any_call(tmp_path, provider):
    first = _write_history(tmp_path / "a.json", [_row(1), _row(2)])
    second = _write_history(tmp_path / "b.json", [_row(2), _row(3)])
    corpus = load_backtest_corpus([first, second])
    assert [row["issue_number"] for row in corpus.rows] == [1, 2, 3]
    assert corpus.files == ("a.json", "b.json")

    conflict = _write_history(tmp_path / "c.json", [_row(2, plan_rounds=3)])
    with pytest.raises(AgentLoopError, match="conflicting rows for OWNER/REPO#2"):
        load_backtest_corpus([first, conflict])
    provider.behave({})
    output = tmp_path / "out.json"
    with pytest.raises(AgentLoopError, match="conflicting"):
        run_backtest([first, conflict], provider_command=provider.command, output=output)
    assert list(provider.echo_dir.iterdir()) == []
    assert not output.exists()


def test_every_issue_lands_in_exactly_one_chronological_split(tmp_path, provider):
    rows = [
        _row(number, run_started=f"2026-09-{day:02d}T00:00:00+00:00", as_of=f"2026-09-{day:02d}T00:00:00+00:00")
        for number, day in ((5, 9), (1, 3), (4, 7), (2, 1), (3, 5))
    ]
    report, _ = _report(tmp_path, rows, provider, {})
    splits = report["splits"]
    assert splits == {
        "calibration": ["OWNER/REPO#2", "OWNER/REPO#1", "OWNER/REPO#3"],
        "holdout": ["OWNER/REPO#4", "OWNER/REPO#5"],
    }
    assert not set(splits["calibration"]) & set(splits["holdout"])
    assert sorted(target["issue_number"] for target in report["targets"]) == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# request-leak-free
# ---------------------------------------------------------------------------


def test_request_is_leak_free_at_the_run_start_cutoff(tmp_path, provider):
    cutoff = "2026-10-01T12:00:00+00:00"
    rows = [
        _row(50, run_started=cutoff, as_of="2026-09-30T00:00:00+00:00",
             terminal_states=["leak-marker-50"], route="leak-route-50"),
        _row(10, available="2026-09-15T00:00:00+00:00", run_started="2026-09-10T00:00:00+00:00",
             as_of="2026-09-10T00:00:00+00:00"),
        _row(11, available=cutoff, run_started="2026-09-11T00:00:00+00:00", as_of="2026-09-11T00:00:00+00:00"),
        _row(12, available="2026-10-03T00:00:00+00:00", run_started="2026-09-12T00:00:00+00:00",
             as_of="2026-09-12T00:00:00+00:00"),
        _row(13, repo="OTHER/REPO", available="2026-09-15T00:00:00+00:00"),
        _row(14, status=SNAPSHOT_UNAVAILABLE, available="2026-09-15T00:00:00+00:00"),
        # A discussion record precedes an issue edit that the later planner's digest matched.
        _row(51, run_started="2026-09-20T00:00:00+00:00", as_of="2026-09-22T00:00:00+00:00",
             available="2026-10-05T00:00:00+00:00"),
        _row(52, as_of=None),
    ]
    report, _ = _report(tmp_path, rows, provider, {})
    sent = json.loads((provider.echo_dir / "50.json").read_text())
    assert set(sent) == {
        "schema_version", "kind", "repo", "issue_number", "cutoff", "title", "body", "signals",
        "similar_issues", "routes", "reason_signals", "answer_shapes",
    }
    assert sent["kind"] == "triage_backtest_request"
    assert sent["cutoff"] == cutoff
    assert sent["title"] == TITLE
    assert [item["issue_number"] for item in sent["similar_issues"]] == [10]
    assert sent["similar_issues"][0]["outcome"]["issue_number"] == 10
    assert "snapshot" not in sent["similar_issues"][0]["outcome"]
    raw = (provider.echo_dir / "50.json").read_text()
    assert "leak-marker-50" not in raw and "leak-route-50" not in raw
    assert '"label"' not in raw
    assert sent["answer_shapes"] == ["issue_triage", "issue_triage_probability"]
    for excluded in (51, 52, 14):
        assert not (provider.echo_dir / f"{excluded}.json").exists()
    assert report["corpus"]["excluded_targets_by_reason"] == {
        "no-usable-snapshot": 1, "snapshot-after-run-start": 1, "snapshot-undated": 1,
    }
    assert 50 in _by_issue(report)


def test_repo_filter_restricts_targets_but_not_same_repo_retrieval(tmp_path, provider):
    rows = [_row(1, repo="OTHER/REPO"), _row(2, run_started="2026-09-20T00:00:00+00:00",
                                               as_of="2026-09-20T00:00:00+00:00"), _row(3)]
    report, _ = _report(tmp_path, rows, provider, {}, repo="owner/repo")
    assert sorted(_by_issue(report)) == [2, 3]
    sent = json.loads((provider.echo_dir / "2.json").read_text())
    assert [item["issue_number"] for item in sent["similar_issues"]] == [3]


# ---------------------------------------------------------------------------
# provider-failure-modes
# ---------------------------------------------------------------------------


def test_provider_failures_become_recorded_abstains_and_no_process_survives(tmp_path, provider, monkeypatch):
    for name in backtest.TOKEN_VARIABLES:
        monkeypatch.setenv(name, "secret-token")
    behaviors = {
        1: "timeout", 2: "exit:3", 3: "big-stdout", 4: "noisy-stderr", 5: "grandchild:0",
        6: "grandchild:4", 7: "malformed", 8: "extra-key", 9: "prob:0.5,0.4,0.4",
        10: "prob:0.5,0.5,0", 11: "route:direct",
    }
    rows = [
        _row(number, run_started=f"2026-09-{number:02d}T00:00:00+00:00", as_of=f"2026-09-{number:02d}T00:00:00+00:00")
        for number in behaviors
    ]
    report, output = _report(tmp_path, rows, provider, behaviors, provider_timeout=2)
    targets = _by_issue(report)
    assert targets[1]["reason"] == "timeout"
    assert targets[2]["reason"] == "exit-3"
    assert targets[3]["reason"] == "stdout-too-large"
    assert targets[4]["reason"] is None and targets[4]["prediction"] == "direct"
    assert targets[4]["stderr_truncated"] is True
    assert targets[5]["reason"] is None and targets[5]["prediction"] == "direct"
    assert targets[6]["reason"] == "exit-4"
    assert targets[7]["reason"].startswith("invalid-answer: not valid JSON")
    assert targets[8]["reason"].startswith("invalid-answer: probability answer fields differ")
    assert targets[9]["reason"] == "invalid-answer: probabilities must sum to 1 within 0.001."
    assert targets[10]["reason"] == "probability-tie"
    assert targets[11]["prediction"] == "direct"
    for number in (1, 2, 3, 6, 7, 8, 9):
        assert targets[number]["prediction"] == ABSTAIN
        assert targets[number]["answer_shape"] == "invalid"
    for number in (1, 5, 6):
        pid = int((provider.pid_dir / f"{number}.pid").read_text())
        assert _pid_gone(pid), f"provider grandchild of issue {number} survived"
    for number in behaviors:
        assert json.loads((provider.echo_dir / f"{number}.env").read_text()) == []
        assert not Path((provider.echo_dir / f"{number}.cwd").read_text()).exists()
    assert output.exists()


def test_noisy_stderr_is_drained_with_bounded_retention(provider):
    provider.behave({7: "noisy-stderr"}, echo_requests=False)
    request = json.dumps({"issue_number": 7}).encode()
    result = run_provider(provider.argv, request, timeout=5)
    assert result.failure is None and result.returncode == 0
    assert result.stderr_truncated is True
    assert len(result.stderr) == backtest.MAX_STDERR_BYTES
    assert validate_answer(result.stdout.decode()).prediction == "direct"


def test_launch_failures_are_errno_named_and_the_rest_still_run(tmp_path, provider):
    provider.behave({}, echo_requests=False)
    missing = str(tmp_path / "secret-dir" / "missing-provider")
    not_executable = tmp_path / "not-executable"
    not_executable.write_text("#!/bin/sh\necho hi\n")
    not_executable.chmod(stat.S_IRUSR | stat.S_IWUSR)
    calls = {
        1: command_provider((missing,), timeout=5),
        2: command_provider((str(not_executable),), timeout=5),
        3: command_provider(provider.argv, timeout=5),
    }
    rows = [
        _row(number, run_started=f"2026-09-{number:02d}T00:00:00+00:00", as_of=f"2026-09-{number:02d}T00:00:00+00:00")
        for number in calls
    ]
    corpus = load_backtest_corpus([_write_history(tmp_path / "h.json", rows)])
    report = build_report(
        corpus, provider=lambda target: calls[target.issue_number](target),
        provider_info={"kind": "command"}, spec=LabelSpec(),
    )
    targets = _by_issue(report)
    assert targets[1]["reason"] == "launch-failed:ENOENT"
    assert targets[2]["reason"] == "launch-failed:EACCES"
    assert targets[3]["prediction"] == "plan-first"
    assert "secret-dir" not in json.dumps(report)


# ---------------------------------------------------------------------------
# Answer validation and as_deployed
# ---------------------------------------------------------------------------


def _route_text(route, unsure=False):
    return json.dumps({"schema_version": 1, "kind": "issue_triage", "route": route, "unsure": unsure,
                       "reasons": [{"signal": "no-signal", "explanation": "fixture"}]})


def _prob_text(a, b, c, abstain=False):
    return json.dumps({"schema_version": 1, "kind": "issue_triage_probability",
                       "probabilities": {"discuss-first": a, "plan-first": b, "direct": c}, "abstain": abstain})


@pytest.mark.parametrize("text, prediction, reason, confidence", [
    (_route_text("direct"), "direct", None, None),
    (_route_text("plan-first", unsure=True), ABSTAIN, "unsure", None),
    (_route_text("somewhere"), ABSTAIN, "unsure", None),
    (_prob_text(0.1, 0.2, 0.7), "direct", None, 0.7),
    (_prob_text(0.05, 0.9, 0.05, abstain=True), ABSTAIN, "provider-abstained", None),
    (_prob_text(0.5, 0.5, 0.0), ABSTAIN, "probability-tie", None),
])
def test_valid_answers_and_null_confidence_for_every_abstain(text, prediction, reason, confidence):
    result = validate_answer(text)
    assert (result.prediction, result.reason, result.confidence) == (prediction, reason, confidence)


@pytest.mark.parametrize("text", [
    _prob_text(0.1, 0.2, 0.7) + "\n-- sig",
    _prob_text(0.1, 0.2, 0.6),
    _prob_text(True, 0.0, 0.0),
    _prob_text(1.2, -0.1, -0.1),
    json.dumps({"schema_version": 1, "kind": "issue_triage_probability",
                "probabilities": {"discuss-first": float("nan"), "plan-first": 0.5, "direct": 0.5}, "abstain": False}),
    json.dumps({"schema_version": 1, "kind": "something-else"}),
    "[1, 2]",
    "",
])
def test_invalid_answers_abstain_with_a_recorded_reason(text):
    result = validate_answer(text)
    assert result.prediction == ABSTAIN and result.answer_shape == "invalid" and result.confidence is None
    assert classify_reason(result.reason) == "invalid"
    assert len(result.reason) <= 200


def test_every_reason_the_module_can_emit_is_classified():
    exact = {getattr(backtest, name) for name in dir(backtest)
             if name.startswith("REASON_") and not name.endswith("_PREFIX")}
    prefixes = {getattr(backtest, name) for name in dir(backtest)
                if name.startswith("REASON_") and name.endswith("_PREFIX")}
    assert exact == set(backtest.VALID_ABSTAIN_REASONS) | set(backtest.INVALID_REASONS)
    assert prefixes == set(backtest.INVALID_REASON_PREFIXES)
    for reason in exact:
        assert classify_reason(reason) in backtest.AS_DEPLOYED_BY_CLASS
    for prefix in prefixes:
        assert classify_reason(prefix + "X") == "invalid"
    assert {classify_reason(reason) for reason in backtest.VALID_ABSTAIN_REASONS} == {"valid-abstain"}
    assert classify_reason(None) == "valid-route"
    with pytest.raises(AgentLoopError):
        classify_reason("never-emitted")


def _target(number, label, *, day=1):
    cutoff = datetime.datetime(2026, 9, day, tzinfo=UTC)
    return Target(REPO, number, cutoff, label, {"similar_issues": []})


def _scored(number, label, prediction, split="calibration"):
    return Scored(_target(number, label), split, prediction)


def _committed(route, confidence):
    return Prediction("probability", route, None, confidence=confidence, max_probability=confidence)


def test_as_deployed_and_abstain_excluded_metrics_match_hand_computation():
    scored = [
        _scored(1, "direct", Prediction("route", "direct", None)),
        _scored(2, "plan-first", Prediction("route", ABSTAIN, "unsure")),
        _scored(3, "discuss-first", Prediction("probability", ABSTAIN, "provider-abstained", max_probability=0.9)),
        _scored(4, "plan-first", Prediction("probability", ABSTAIN, "probability-tie", max_probability=0.5)),
        _scored(5, "discuss-first", Prediction("invalid", ABSTAIN, "invalid-answer: not valid JSON."), "holdout"),
        _scored(6, "direct", Prediction("invalid", ABSTAIN, "timeout"), "holdout"),
        _scored(7, "plan-first", Prediction("invalid", ABSTAIN, "launch-failed:ENOENT"), "holdout"),
        _scored(8, "discuss-first", _committed("discuss-first", 0.8), "holdout"),
    ]
    assert [item.prediction.as_deployed for item in scored] == [
        "direct", "plan-first", "plan-first", "plan-first", ABSTAIN, ABSTAIN, ABSTAIN, "discuss-first",
    ]
    metrics = compute_metrics(scored)
    excluded = metrics["abstain_excluded"]["all"]
    assert (excluded["coverage"], excluded["accuracy_on_covered"], excluded["invalid_count"]) == (0.25, 1.0, 3)
    assert excluded["per_route"] == {
        "discuss-first": {"support": 3, "predicted": 1, "tp": 1, "precision": 1.0, "recall": 0.3333},
        "plan-first": {"support": 3, "predicted": 0, "tp": 0, "precision": None, "recall": 0.0},
        "direct": {"support": 2, "predicted": 1, "tp": 1, "precision": 1.0, "recall": 0.5},
    }
    deployed = metrics["as_deployed"]["all"]
    assert (deployed["coverage"], deployed["accuracy_on_covered"]) == (0.625, 0.8)
    assert deployed["per_route"]["plan-first"] == {
        "support": 3, "predicted": 3, "tp": 2, "precision": 0.6667, "recall": 0.6667,
    }
    assert metrics["as_deployed"]["calibration"]["coverage"] == 1.0
    assert metrics["abstain_excluded"]["holdout"]["coverage"] == 0.25
    assert compute_metrics([])["abstain_excluded"]["all"]["coverage"] is None


def test_majority_baseline_uses_calibration_with_fixed_tie_order():
    scored = [
        _scored(1, "direct", Prediction("route", "direct", None)),
        _scored(2, "discuss-first", Prediction("route", "direct", None)),
        _scored(3, "direct", Prediction("route", "direct", None), "holdout"),
    ]
    baseline = backtest.majority_baseline(scored)
    assert baseline["route"] == "discuss-first"
    assert baseline["splits"]["holdout"]["accuracy"] == 0.0
    assert baseline["splits"]["calibration"]["accuracy"] == 0.5
    assert backtest.majority_baseline([])["reason"] == "insufficient-data"


# ---------------------------------------------------------------------------
# hybrid-finding-report-only
# ---------------------------------------------------------------------------


def _hybrid_fixture(holdout_second_label="direct"):
    return [
        _scored(1, "direct", _committed("direct", 0.9)),
        _scored(2, "plan-first", _committed("plan-first", 0.6)),
        _scored(3, "plan-first", Prediction("probability", ABSTAIN, "probability-tie", max_probability=0.5)),
        _scored(4, "direct", Prediction("probability", ABSTAIN, "provider-abstained", max_probability=0.97)),
        _scored(5, "discuss-first", _committed("discuss-first", 0.95), "holdout"),
        _scored(6, holdout_second_label, _committed("plan-first", 0.7), "holdout"),
        _scored(7, "direct", Prediction("probability", ABSTAIN, "provider-abstained", max_probability=0.99), "holdout"),
    ]


def test_hybrid_sweep_counts_only_committed_probability_answers():
    finding = hybrid_finding(_hybrid_fixture(), target_accuracy=0.9, min_support=2)
    assert finding["committed_answers"] == {"calibration": 2, "holdout": 2}
    sweep = {entry["threshold"]: entry for entry in finding["sweep"]}
    assert [entry["threshold"] for entry in finding["sweep"]] == [
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    ]
    assert sweep[0.5]["calibration"] == {"confident": 2, "accuracy": 1.0, "coverage": 0.5}
    assert sweep[0.5]["holdout"] == {"confident": 2, "accuracy": 0.5, "coverage": 0.6667}
    assert sweep[0.65]["calibration"] == {"confident": 1, "accuracy": 1.0, "coverage": 0.25}
    assert sweep[0.95]["calibration"] == {"confident": 0, "accuracy": None, "coverage": 0.0}
    assert sweep[0.95]["holdout"] == {"confident": 1, "accuracy": 1.0, "coverage": 0.3333}
    # Holdout fails at the calibration threshold.
    assert (finding["status"], finding["decided_by"], finding["threshold"]) == (
        "not-supported", "rule-4-holdout-check", 0.5,
    )
    assert finding["routing_changed"] is False
    assert "routing is unchanged" in finding["note"]


def test_hybrid_finding_rules_in_order():
    supported = hybrid_finding(_hybrid_fixture("plan-first"), target_accuracy=0.9, min_support=2)
    assert (supported["status"], supported["decided_by"], supported["threshold"]) == (
        "supported", "rule-4-holdout-check", 0.5,
    )
    wrong = [_scored(n, "direct", _committed("plan-first", 0.9)) for n in (1, 2)] + [
        _scored(n, "direct", _committed("direct", 0.9), "holdout") for n in (3, 4)
    ]
    calibration_fails = hybrid_finding(wrong, target_accuracy=0.9, min_support=2)
    assert (calibration_fails["status"], calibration_fails["decided_by"]) == (
        "not-supported", "rule-3-calibration-search",
    )
    overlap = [_scored(n, "direct", _committed("plan-first", 0.9)) for n in range(1, 11)] + [
        _scored(n, "direct", _committed("direct", 0.9), "holdout") for n in range(11, 14)
    ]
    insufficient = hybrid_finding(overlap, target_accuracy=0.9, min_support=10)
    assert (insufficient["status"], insufficient["decided_by"]) == ("insufficient-data", "rule-2-min-support")
    route_only = [_scored(1, "direct", Prediction("route", "direct", None))]
    not_applicable = hybrid_finding(route_only, target_accuracy=0.9, min_support=2)
    assert (not_applicable["status"], not_applicable["decided_by"]) == (
        "not-applicable", "rule-1-no-probability-answers",
    )
    for finding in (supported, calibration_fails, insufficient, not_applicable):
        assert finding["routing_changed"] is False and "routing is unchanged" in finding["note"]
    abstain_only = [
        _scored(1, "direct", Prediction("probability", ABSTAIN, "provider-abstained", max_probability=0.99)),
        _scored(2, "direct", Prediction("probability", ABSTAIN, "probability-tie", max_probability=0.5), "holdout"),
    ]
    finding = hybrid_finding(abstain_only, target_accuracy=0.9, min_support=1)
    assert finding["status"] == "insufficient-data"
    assert finding["committed_answers"] == {"calibration": 0, "holdout": 0}
    assert all(entry[split]["confident"] == 0 for entry in finding["sweep"] for split in ("calibration", "holdout"))


# ---------------------------------------------------------------------------
# backtest-offline-provider
# ---------------------------------------------------------------------------


def _offline_rows():
    return [
        _row(1, run_started="2026-09-01T00:00:00+00:00", as_of="2026-09-01T00:00:00+00:00"),
        _row(2, plan_rounds=2, run_started="2026-09-02T00:00:00+00:00", as_of="2026-09-02T00:00:00+00:00"),
        _row(3, stops=1, run_started="2026-09-03T00:00:00+00:00", as_of="2026-09-03T00:00:00+00:00"),
        _row(4, run_started="2026-09-04T00:00:00+00:00", as_of="2026-09-04T00:00:00+00:00"),
        _row(5, plan_rounds=2, run_started="2026-09-05T00:00:00+00:00", as_of="2026-09-05T00:00:00+00:00"),
        _row(6, step_backs=1, run_started="2026-09-06T00:00:00+00:00", as_of="2026-09-06T00:00:00+00:00"),
    ]


OFFLINE_BEHAVIORS = {
    1: "route:direct", 2: "unsure", 3: "prob-abstain:0.9,0.05,0.05",
    4: "malformed", 5: "prob:0.1,0.8,0.1", 6: "route:discuss-first",
}


@pytest.mark.parametrize("output_format", ["json", "markdown"])
def test_offline_backtest_report_is_deterministic_and_leaves_only_the_report(
    tmp_path, provider, output_format
):
    rows = _offline_rows()
    provider.behave(OFFLINE_BEHAVIORS, echo_requests=False)
    history = _write_history(tmp_path / "history.json", rows)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output = out_dir / f"report.{output_format}"
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    first = run_backtest([history], provider_command=provider.command, output=output,
                         output_format=output_format, provider_timeout=5)
    first_bytes = output.read_bytes()
    second = run_backtest([history], provider_command=provider.command, output=output,
                          output_format=output_format, provider_timeout=5)
    assert output.read_bytes() == first_bytes
    assert first == second
    assert sorted(path.name for path in out_dir.iterdir()) == [output.name]
    targets = _by_issue(first)
    assert targets[2]["reason"] == "unsure"
    assert targets[3]["reason"] == "provider-abstained"
    assert targets[4]["reason"].startswith("invalid-answer:")
    assert targets[5]["confidence"] == 0.8
    assert first["hybrid"]["status"] == "insufficient-data"
    assert first["hybrid"]["routing_changed"] is False
    assert first["provider"]["executable"] == os.path.basename(sys.executable)
    assert str(tmp_path) not in json.dumps(first["provider"])
    text = first_bytes.decode()
    if output_format == "json":
        assert json.loads(text) == first
        assert text.endswith("}\n")
    else:
        assert text.startswith("# Triage backtest report")
        assert "routing is unchanged" in text
        assert "provider-abstained" in text and "unsure" in text


def test_provider_temp_directories_are_removed(tmp_path, provider):
    report, output = _report(tmp_path, _offline_rows(), provider, OFFLINE_BEHAVIORS)
    for number in OFFLINE_BEHAVIORS:
        workdir = Path((provider.echo_dir / f"{number}.cwd").read_text())
        assert workdir.name.startswith("agent-loop-triage-provider-")
        assert not workdir.exists()
    assert sorted(path.name for path in output.parent.iterdir()) == [output.name]


def test_atomic_write_removes_the_temp_file_on_failure(tmp_path, monkeypatch):
    target = tmp_path / "report.json"

    def boom(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(backtest.os, "replace", boom)
    with pytest.raises(OSError):
        backtest.write_atomically(target, "x")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# cli-offline-only
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_gh(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "gh-invoked"
    gh = bin_dir / "gh"
    gh.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(marker))}\nexit 97\n")
    gh.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    def refuse(*args, **kwargs):
        raise AssertionError("the backtest must not use the agent runner")

    monkeypatch.setattr("coding_review_agent_loop.runner.Runner.run", refuse)
    return marker


def test_cli_backtest_is_offline_and_writes_only_the_report(tmp_path, provider, fake_gh, capsys):
    provider.behave(OFFLINE_BEHAVIORS, echo_requests=False)
    history = _write_history(tmp_path / "history.json", _offline_rows())
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    output = out_dir / "report.json"
    code = main([
        "triage", "backtest", "--history", str(history), "--provider-command", provider.command,
        "--output", str(output), "--provider-timeout", "5", "--hybrid-min-support", "1",
        "--discuss-min-plan-rounds", "3",
    ])
    assert code == 0
    assert not fake_gh.exists()
    assert sorted(path.name for path in out_dir.iterdir()) == ["report.json"]
    report = json.loads(output.read_text())
    assert report["kind"] == "triage_backtest_report"
    assert report["label_spec"]["discuss_min_plan_rounds"] == 3
    assert report["hybrid"]["min_support"] == 1
    assert "routing unchanged" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [
    ["--provider-timeout", "0"],
    ["--provider-timeout", "601"],
    ["--discuss-min-plan-rounds", "1"],
    ["--direct-max-pr-rounds", "0"],
    ["--hybrid-target-accuracy", "0"],
    ["--hybrid-target-accuracy", "1.5"],
    ["--hybrid-min-support", "0"],
    ["--provider-command", ""],
])
def test_cli_invalid_flags_fail_before_any_provider_call(tmp_path, provider, fake_gh, flags):
    provider.behave(OFFLINE_BEHAVIORS)
    history = _write_history(tmp_path / "history.json", _offline_rows())
    output = tmp_path / "out" / "report.json"
    output.parent.mkdir()
    argv = ["triage", "backtest", "--history", str(history), "--provider-command", provider.command,
            "--output", str(output)]
    assert main(argv + flags) == 1
    assert list(provider.echo_dir.iterdir()) == []
    assert not output.exists()
    assert not fake_gh.exists()


def test_selection_counts_considered_rows():
    corpus = backtest.Corpus(files=("x.json",), rows=(_row(1), _row(2, complete=False)))
    selection = select_targets(corpus, LabelSpec())
    assert (selection.considered, selection.labeled, len(selection.targets)) == (2, 1, 1)
