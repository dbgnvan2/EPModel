"""Shared setup for the v2 agent-model tests.

The M11.D.7 guard (no test may write outside its temporary path) lands at
step 13 of docs/implementation_plan_phase_b.md; until then each test that
writes must take ``tmp_path`` explicitly.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
