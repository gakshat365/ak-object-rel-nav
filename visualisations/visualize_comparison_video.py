#!/usr/bin/env python3
"""
Synchronized 4-Way Comparison Video Generator.
Compares: Baseline Behavioral Cloning (BC) vs. DAgger Iteration 0, 1, 2 on failure episodes.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from dagger.visualize_comparison_video import main

if __name__ == "__main__":
    main()
