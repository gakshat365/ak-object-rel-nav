#!/usr/bin/env python3
"""
DEPRECATED: rerun_missing_episodes.py has been consolidated into rerun_all_failure_episodes.py.
rerun_all_failure_episodes.py automatically detects existing video files and only reruns missing episodes.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from visualisations.rerun_all_failure_episodes import main

if __name__ == "__main__":
    print("[NOTICE] rerun_missing_episodes.py is deprecated. Delegating to rerun_all_failure_episodes.py...")
    main()
