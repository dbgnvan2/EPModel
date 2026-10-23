#!/bin/bash
# SessionStart hook for Claude Code cloud sessions: make the suite runnable on the fresh clone.
# CI (.github/workflows/tests.yml) does the same two things: install requirements.txt and check out with the
# tag `version-1` present (tests/test_version_1_status.py reads it, and a shallow clone leaves it out).
# Idempotent; does nothing outside a cloud session.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt

# The clone is shallow and without tags; the status test needs the tag and the commits behind it.
if [ "$(git rev-parse --is-shallow-repository)" = "true" ]; then
  git fetch --quiet --unshallow --tags origin
elif ! git rev-parse --quiet --verify 'version-1^{commit}' > /dev/null; then
  git fetch --quiet --tags origin
fi
