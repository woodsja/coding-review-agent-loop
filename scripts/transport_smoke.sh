#!/usr/bin/env bash
# Pre-merge transport smoke check (#1383).
#
# Builds this checkout (the PR head) into a temporary venv and runs that
# build's `agent-loop transport-check` once against the target repository,
# using the operator's real remote URL and credentials.  Required before
# merging any change to git_transport.py, secure_git.py or the checkout sync
# path; see docs/local_agent_loop.md "Pre-merge transport smoke check".
#
# Usage: scripts/transport_smoke.sh --repo OWNER/REPO [--base BRANCH]
#          [--trusted-origin-protocol https|ssh] [--git-ssh-identity PATH] [--gh-cmd GH]
#
# Environment: AGENT_LOOP_SMOKE_SOURCE (checkout to build; default: this
# script's repository root), AGENT_LOOP_SMOKE_PYTHON (default: python3).
#
# Exit status: transport-check's own status (0 pass, 1 fail), 2 for a
# missing --repo, 3 when the venv or the install fails.
set -euo pipefail

has_repo=0
for arg in "$@"; do
  if [ "$arg" = "--repo" ]; then
    has_repo=1
  fi
done
if [ "$has_repo" -ne 1 ]; then
  echo "transport-smoke: --repo OWNER/REPO is required" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
src=${AGENT_LOOP_SMOKE_SOURCE:-$(dirname "$script_dir")}
py=${AGENT_LOOP_SMOKE_PYTHON:-python3}

venv=$(mktemp -d)
trap 'rm -rf "$venv"' EXIT

"$py" -m venv "$venv" >/dev/null 2>&1 || { echo "transport-smoke: build failed"; exit 3; }
"$venv/bin/pip" install --quiet "$src" >/dev/null 2>&1 || { echo "transport-smoke: build failed"; exit 3; }

status=0
"$venv/bin/agent-loop" transport-check "$@" || status=$?
exit "$status"
