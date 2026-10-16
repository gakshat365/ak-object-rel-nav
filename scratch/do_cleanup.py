import os
from pathlib import Path

repo_root = Path("/scratch2/akshat.gupta/ak-object-rel-nav")

# 1. Remove 0-byte experiment_1
exp1 = repo_root / "dagger" / "experiment_1"
if exp1.exists() and exp1.stat().st_size == 0:
    exp1.unlink()
    print("Deleted empty file dagger/experiment_1")

# 2. Remove obsolete partial compilation (458 MB)
comp_partial = repo_root / "visualisations" / "failure_episodes_compilation.mp4"
comp_all = repo_root / "visualisations" / "failure_episodes_compilation_all338.mp4"
if comp_partial.exists() and comp_all.exists() and comp_all.stat().st_size > 500_000_000:
    comp_partial.unlink()
    print(f"Deleted obsolete partial compilation {comp_partial.name} (reclaimed 458 MB)")

# 3. Create dagger/legacy_tests directory and move validate_expert.py & validate_gate2.py
legacy_dir = repo_root / "dagger" / "legacy_tests"
legacy_dir.mkdir(parents=True, exist_ok=True)
for fname in ["validate_expert.py", "validate_gate2.py"]:
    src = repo_root / "dagger" / fname
    dst = legacy_dir / fname
    if src.exists() and not dst.exists():
        src.rename(dst)
        print(f"Moved {fname} -> dagger/legacy_tests/{fname}")

print("Cleanup script execution complete!")
