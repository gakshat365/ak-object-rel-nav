"""
Standalone Milestone Validation for Gate 2:
Interleaved Rollout Collection and Dataset Aggregation Verification.

Validates that:
1. RolloutCollector successfully spawns Habitat-Sim, queries student policy and expert oracle,
   samples executed action via Bernoulli(beta), and writes sample bundles.
2. DataAggregator writes valid iteration manifests and aggregates without duplicates.
3. DAggerRecoveryDataset loads recovery samples with correct ObjectReact shapes and normalization.
4. GNM forward pass, action loss calculation, and backward gradient computation succeed without NaNs.
"""

import logging
from pathlib import Path
import sys
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[ValidateGate2]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "libs" / "control" / "object_react" / "train"))

from dagger.core.failure_loader import FailureLoader
from dagger.core.rollout_collector import RolloutCollector
from dagger.core.data_aggregator import DataAggregator, DAggerRecoveryDataset, verify_gate2


def run_pilot_and_validate():
    cfg_path = REPO_ROOT / "dagger" / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    checkpoint = cfg["paths"]["base_checkpoint"]

    logger.info("Step 1: Loading failure cases...")
    loader = FailureLoader(str(cfg_path))
    all_cases = loader.load_from_failed_cases_dir()
    train_c, val_c, test_c = loader.split_cases(all_cases)
    logger.info(f"Loaded {len(all_cases)} total cases: Train={len(train_c)}, Val={len(val_c)}, Test={len(test_c)}")

    pilot_case = train_c[0]
    logger.info(f"Selected pilot case: {pilot_case.episode_name} ({pilot_case.scene_id})")

    logger.info("\nStep 2: Executing pilot rollout (5 steps, beta=0.7)...")
    collector = RolloutCollector(str(cfg_path))
    pilot_out_dir = REPO_ROOT / "dagger" / "data" / "pilot" / "samples"
    pilot_out_dir.mkdir(parents=True, exist_ok=True)

    pilot_records = collector.collect_iteration(
        cases=[pilot_case],
        checkpoint_path=checkpoint,
        beta=0.7,
        iteration_idx=0,
        output_dir=pilot_out_dir,
        max_episodes=1,
        max_steps_per_episode=5,
    )
    assert len(pilot_records) > 0, "Pilot rollout failed to collect any records!"
    logger.info(f"Collected {len(pilot_records)} pilot records.")

    logger.info("\nStep 3: Emitting iteration manifest and aggregating...")
    aggregator = DataAggregator(str(cfg_path))
    manifest_path = aggregator.save_iteration_manifest(pilot_records, iteration_idx=0)
    agg_manifest, total_samples = aggregator.aggregate_manifests(current_iteration_idx=0)
    assert agg_manifest.exists(), f"Aggregate manifest {agg_manifest} does not exist!"
    logger.info(f"Aggregate manifest ready with {total_samples} samples.")

    logger.info("\nStep 4: Running Gate 2 DataLoader and Model Verification...")
    passed = verify_gate2(agg_manifest, checkpoint_path=checkpoint, device="cpu", batch_size=2)
    assert passed, "Gate 2 dataset and model verification failed!"

    logger.info("\n========================================================")
    logger.info(">>> GO / NO-GO GATE 2 PASSED: Rollout and Aggregation Verified!")
    logger.info("========================================================\n")
    return True


if __name__ == "__main__":
    success = run_pilot_and_validate()
    sys.exit(0 if success else 1)
