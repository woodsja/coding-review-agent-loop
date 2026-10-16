"""Opt-in live check of the default Antigravity semantic dedupe turn (#1472).

It makes one real ``agy`` call through ``default_semantic_transport`` with a
directly constructed default configuration, so CI never runs it.  Run with:

    AGENT_LOOP_LIVE_AGY=1 python -m pytest tests/test_semantic_dedupe_live.py -q -s -p no:cacheprovider

The test prints a ``LIVE-EVIDENCE`` line with the subprocess log path and the
parsed match; set ``AGENT_LOOP_LIVE_LOG_DIR`` to keep the log outside pytest's
temporary directory.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop.agents import antigravity as agy_mod
from coding_review_agent_loop.config import effective_semantic_followup_timeout_seconds
from coding_review_agent_loop.runner import Runner
from coding_review_agent_loop.semantic_dedupe import (
    SemanticCandidate,
    SemanticDedupeMatcher,
    default_semantic_transport,
)
from coding_review_agent_loop.usage import RunUsageContext

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENT_LOOP_LIVE_AGY") != "1" or shutil.which("agy") is None,
    reason="live agy semantic dedupe check; set AGENT_LOOP_LIVE_AGY=1 with agy on PATH",
)


def test_live_default_antigravity_semantic_dedupe_returns_parsed_match(tmp_path):
    """`live-agy-dedupe` (#1472)."""
    log_dir = Path(os.environ.get("AGENT_LOOP_LIVE_LOG_DIR") or tmp_path / "logs")
    config = make_config(tmp_path, quiet=False, log_dir=log_dir)
    assert config.semantic_followup_backend == "antigravity"
    assert effective_semantic_followup_timeout_seconds(config) == 120
    settings_path = agy_mod._antigravity_settings_path()
    settings_before = settings_path.read_text(encoding="utf-8") if settings_path.exists() else None
    usage = RunUsageContext(run_id="semantic-live", summary_path=tmp_path / "usage.json")
    matcher = SemanticDedupeMatcher(runner=Runner(), config=config, usage_context=usage)
    assert matcher.transport is default_semantic_transport
    assert matcher.preflight() is True

    match = matcher.match(
        proposed=(
            "Add a retry with exponential backoff when the GitHub search API "
            "returns HTTP 502 while looking up existing follow-up trackers."
        ),
        candidates=(
            SemanticCandidate(
                identity=101,
                title="Retry transient 502s from GitHub issue search",
                body=(
                    "Follow-up tracker lookup fails on a transient HTTP 502 from the "
                    "search API. Retry it with exponential backoff."
                ),
            ),
            SemanticCandidate(
                identity=202,
                title="Document the README installation steps for Windows",
                body="Explain how to install the CLI on Windows in the README.",
            ),
        ),
        source_context="Live semantic dedupe check for issue #1472.",
    )

    assert match.duplicate_of in {101, 202, None}
    assert match.confidence in {"high", "medium", "low"}
    assert match.reason.strip()
    record = usage.records[-1]
    assert record.turn_role == "semantic-dedupe"
    assert record.validation_status == "validated"
    settings_after = settings_path.read_text(encoding="utf-8") if settings_path.exists() else None
    assert settings_after == settings_before
    print(
        "LIVE-EVIDENCE "
        + json.dumps(
            {
                "log_path": record.log_path,
                "model": record.model,
                "duplicate_of": match.duplicate_of,
                "confidence": match.confidence,
                "reason": match.reason,
            },
            sort_keys=True,
        )
    )
