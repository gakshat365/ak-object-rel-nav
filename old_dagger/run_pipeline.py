"""
DAgger Full Pipeline Runner.

Executes the complete DAgger experiment for failure recovery:

  Baseline A: Evaluate the base checkpoint (no fine-tuning) → pi_0
  Baseline B: Fine-tune on expert-only demonstrations (beta=1.0, single pass)
  DAgger Iters: 3 iterations with beta = [0.7, 0.3, 0.0]
    - Iter 0: beta=0.7 → pi_1
    - Iter 1: beta=0.3 → pi_2
    - Iter 2: beta=0.0 → pi_3

After all runs, generates a Markdown report summarising:
  - Recovery Success Rate (R-SR)
  - Recovery SPL (R-SPL)
  - Recovery Soft-SPL
  - Mean Distance Closed

Gate 3: R-SR of best DAgger policy must strictly exceed both baselines.

Usage (from repo root):
  pixi run python dagger/run_pipeline.py [--config dagger/config.yaml]

Flags to resume a partial run:
  --start-from  {baseline_a | baseline_b | dagger_0 | dagger_1 | dagger_2 | report}
  --results-json  <path>  # provide partial results JSON to resume from
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("[RunPipeline]")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from old_dagger.core.failure_loader import FailureLoader
from old_dagger.core.data_aggregator import DataAggregator
from old_dagger.core.trainer import DAggerTrainer
from old_dagger.evaluate_recovery import evaluate_checkpoint_on_cases
from old_dagger.run_iteration import run_iteration


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_cfg(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_test_cases(config_path: str, cfg: dict):
    loader = FailureLoader(config_path)
    all_cases = loader.load_cases()
    _, _, test_cases = loader.split_cases(all_cases)
    return test_cases


def _save_results(results: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved → {out_path}")


def _metric_row(entry: dict) -> dict:
    es = entry.get("eval_summary", {})
    return {
        "label": entry.get("label", ""),
        "R-SR (%)": round(es.get("recovery_success_rate", 0.0), 2),
        "R-SPL (%)": round(es.get("recovery_spl", 0.0), 2),
        "Soft-SPL (%)": round(es.get("recovery_soft_spl", 0.0), 2),
        "Dist Closed (m)": round(es.get("mean_distance_closed", 0.0), 3),
        "checkpoint": entry.get("checkpoint_path", "N/A"),
    }


def _write_report(results: dict, report_path: Path, gate3_passed: bool):
    """Generate a Markdown report comparing all runs."""
    lines = [
        "# DAgger Recovery Experiment — Summary Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Recovery Metrics",
        "",
        "| Run | R-SR (%) | R-SPL (%) | Soft-SPL (%) | Dist Closed (m) |",
        "|-----|----------|-----------|--------------|-----------------|",
    ]

    for entry in results.get("runs", []):
        m = _metric_row(entry)
        lines.append(
            f"| **{m['label']}** "
            f"| {m['R-SR (%)']:.2f} "
            f"| {m['R-SPL (%)']:.2f} "
            f"| {m['Soft-SPL (%)']:.2f} "
            f"| {m['Dist Closed (m)']:.3f} |"
        )

    gate_emoji = "✅ PASSED" if gate3_passed else "❌ FAILED"
    lines += [
        "",
        "---",
        "",
        f"## Gate 3 Verdict: {gate_emoji}",
        "",
    ]

    if gate3_passed:
        best_run = results.get("best_dagger_run", {})
        best_label = best_run.get("label", "unknown")
        best_sr = best_run.get("eval_summary", {}).get("recovery_success_rate", 0.0)
        baseline_a_sr = results.get("baseline_a", {}).get("eval_summary", {}).get("recovery_success_rate", 0.0)
        baseline_b_sr = results.get("baseline_b", {}).get("eval_summary", {}).get("recovery_success_rate", 0.0)
        lines += [
            f"Best DAgger policy **{best_label}** achieved R-SR={best_sr:.2f}% "
            f"vs Baseline A={baseline_a_sr:.2f}% and Baseline B={baseline_b_sr:.2f}%.",
            "",
            "> **DAgger improves failure recovery over both baselines.**",
        ]
    else:
        lines += [
            "> DAgger did not achieve statistically significant improvement over both baselines.",
            "> Consider: more iterations, more failure cases, or hyperparameter tuning.",
        ]

    lines += [
        "",
        "## Checkpoint Paths",
        "",
    ]
    for entry in results.get("runs", []):
        lines.append(f"- **{entry['label']}**: `{entry.get('checkpoint_path', 'N/A')}`")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Report written → {report_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Gate 3
# ──────────────────────────────────────────────────────────────────────────────

def check_gate3(results: dict) -> tuple[bool, dict | None]:
    """
    Gate 3: Best DAgger iteration R-SR must strictly exceed both baselines.
    Returns (passed, best_dagger_run).
    """
    baseline_a_sr = results.get("baseline_a", {}).get("eval_summary", {}).get("recovery_success_rate", 0.0)
    baseline_b_sr = results.get("baseline_b", {}).get("eval_summary", {}).get("recovery_success_rate", 0.0)

    dagger_runs = results.get("dagger_iterations", [])
    if not dagger_runs:
        logger.warning("No DAgger iterations found for Gate 3 check.")
        return False, None

    best_dagger = max(
        dagger_runs,
        key=lambda r: r.get("eval_summary", {}).get("recovery_success_rate", -1.0),
    )
    best_sr = best_dagger.get("eval_summary", {}).get("recovery_success_rate", 0.0)

    passed = (best_sr > baseline_a_sr) and (best_sr > baseline_b_sr)
    logger.info(f"\nGate 3 Check:")
    logger.info(f"  Best DAgger R-SR : {best_sr:.2f}%")
    logger.info(f"  Baseline A R-SR  : {baseline_a_sr:.2f}%")
    logger.info(f"  Baseline B R-SR  : {baseline_b_sr:.2f}%")
    logger.info(f"  → {'PASSED ✅' if passed else 'FAILED ❌'}")

    return passed, best_dagger


# ──────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    config_path: str,
    start_from: str = "baseline_a",
    prior_results: dict | None = None,
) -> dict:
    """
    Orchestrates the full DAgger pipeline. Returns a consolidated results dict.

    start_from options: baseline_a | baseline_b | dagger_0 | dagger_1 | dagger_2 | report
    """
    cfg = _load_cfg(config_path)
    logs_dir = REPO_ROOT / cfg["paths"]["logs_dir"]
    manifests_dir = REPO_ROOT / cfg["paths"]["manifests_dir"]
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    base_ckpt = cfg["paths"]["base_checkpoint"]
    beta_schedule = cfg["dagger"]["beta_schedule"]
    test_cases = _load_test_cases(config_path, cfg)

    results = prior_results or {
        "baseline_a": {},
        "baseline_b": {},
        "dagger_iterations": [],
        "runs": [],
    }

    STAGES = ["baseline_a", "baseline_b", "dagger_0", "dagger_1", "dagger_2", "dagger_3", "report"]
    start_idx = STAGES.index(start_from) if start_from in STAGES else 0

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline A: Raw base checkpoint (no fine-tuning)
    # ─────────────────────────────────────────────────────────────────────────
    if start_idx <= STAGES.index("baseline_a"):
        if results.get("baseline_a", {}).get("eval_summary"):
            logger.info("\n[SKIP] Baseline A already evaluated in prior results.")
        else:
            logger.info("\n" + "="*70)
            logger.info("  BASELINE A — Base ObjectReact checkpoint (no fine-tuning)")
            logger.info("="*70)

            eval_summary_a = evaluate_checkpoint_on_cases(
                checkpoint_path=base_ckpt,
                cases=test_cases,
                cfg=cfg,
                split_name="test",
            )
            baseline_a = {
                "label": "Baseline A (π₀ base)",
                "checkpoint_path": base_ckpt,
                "eval_summary": eval_summary_a,
            }
            results["baseline_a"] = baseline_a
            results["runs"] = [r for r in results["runs"] if r.get("label") != baseline_a["label"]]
            results["runs"].append(baseline_a)
            _save_results(results, logs_dir / "pipeline_results.json")

    # ─────────────────────────────────────────────────────────────────────────
    # Baseline B: Fine-tune on expert-only demonstrations (beta=1.0)
    # ─────────────────────────────────────────────────────────────────────────
    if start_idx <= STAGES.index("baseline_b"):
        if results.get("baseline_b", {}).get("eval_summary"):
            logger.info("\n[SKIP] Baseline B already completed in prior results.")
        else:
            logger.info("\n" + "="*70)
            logger.info("  BASELINE B — Expert-only fine-tuning (beta=1.0)")
            logger.info("="*70)

            # Re-use existing rollouts if already collected to save 35 minutes
            manifest_b = manifests_dir / "iter_-1_manifest.csv"
            skip_b_rollout = manifest_b.exists() and manifest_b.stat().st_size > 1000

            baseline_b = run_iteration(
                iteration_idx=-1,         # -1 → special label for expert-only baseline
                beta=1.0,
                checkpoint_path=base_ckpt,
                cfg=cfg,
                config_path=config_path,
                prior_manifest=None,      # no aggregation with prior data
                skip_rollout=skip_b_rollout,
            )
            baseline_b["label"] = "Baseline B (expert-only β=1.0)"
            results["baseline_b"] = baseline_b
            results["runs"] = [r for r in results["runs"] if r.get("label") != baseline_b["label"]]
            results["runs"].append(baseline_b)
            _save_results(results, logs_dir / "pipeline_results.json")

    # ─────────────────────────────────────────────────────────────────────────
    # DAgger Iterations 0, 1, 2
    # ─────────────────────────────────────────────────────────────────────────
    prev_agg_manifest = None
    prev_ckpt = base_ckpt

    # Recover manifests/checkpoints from prior dagger iterations if resuming
    for prior_iter in results.get("dagger_iterations", []):
        prior_idx = prior_iter.get("iteration", -2)
        if prior_idx >= 0:
            prior_agg = manifests_dir / f"aggregated_iter_{prior_idx}.csv"
            if prior_agg.exists():
                prev_agg_manifest = str(prior_agg)
            if prior_iter.get("checkpoint_path"):
                prev_ckpt = prior_iter["checkpoint_path"]

    for iter_idx, beta in enumerate(beta_schedule):
        stage_key = f"dagger_{iter_idx}"
        if start_idx > STAGES.index(stage_key):
            logger.info(f"[SKIP] DAgger iteration {iter_idx} (start_from={start_from})")
            continue

        logger.info("\n" + "="*70)
        logger.info(f"  DAGGER ITERATION {iter_idx}  [beta={beta}]")
        logger.info("="*70)

        iter_result = run_iteration(
            iteration_idx=iter_idx,
            beta=beta,
            checkpoint_path=prev_ckpt,
            cfg=cfg,
            config_path=config_path,
            prior_manifest=prev_agg_manifest,
        )
        iter_result["label"] = f"DAgger Iter {iter_idx} (β={beta})"

        # Update chain
        prev_agg_manifest = iter_result["manifest_path"]
        prev_ckpt = iter_result["checkpoint_path"]

        # Upsert into results
        results["dagger_iterations"] = [
            r for r in results["dagger_iterations"] if r.get("iteration") != iter_idx
        ]
        results["dagger_iterations"].append(iter_result)
        results["runs"] = [r for r in results["runs"] if r.get("label") != iter_result["label"]]
        results["runs"].append(iter_result)
        _save_results(results, logs_dir / "pipeline_results.json")

    # ─────────────────────────────────────────────────────────────────────────
    # Gate 3 + Report
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("  GATE 3 CHECK & FINAL REPORT")
    logger.info("="*70)

    gate3_passed, best_dagger = check_gate3(results)
    results["gate3_passed"] = gate3_passed
    results["best_dagger_run"] = best_dagger or {}

    report_path = logs_dir / "dagger_experiment_report.md"
    _write_report(results, report_path, gate3_passed)
    _save_results(results, logs_dir / "pipeline_results.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="DAgger Full Pipeline Runner")
    parser.add_argument("--config", type=str, default="old_dagger/config.yaml")
    parser.add_argument(
        "--start-from",
        type=str,
        default="baseline_a",
        choices=["baseline_a", "baseline_b", "dagger_0", "dagger_1", "dagger_2", "dagger_3", "report"],
        help="Resume the pipeline from a specific stage.",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default=None,
        help="Path to a prior pipeline_results.json to resume from (optional).",
    )
    args = parser.parse_args()

    config_path = str(REPO_ROOT / args.config)

    cfg = _load_cfg(config_path)
    prior_results = None
    default_results_path = REPO_ROOT / cfg["paths"]["logs_dir"] / "pipeline_results.json"
    if args.results_json:
        with open(args.results_json, "r") as f:
            prior_results = json.load(f)
        logger.info(f"Resuming from prior results: {args.results_json}")
    elif default_results_path.exists():
        try:
            with open(default_results_path, "r") as f:
                prior_results = json.load(f)
            logger.info(f"Resuming from existing pipeline_results.json: {default_results_path}")
        except Exception as e:
            logger.warning(f"Could not load prior pipeline_results.json: {e}")

    results = run_pipeline(
        config_path=config_path,
        start_from=args.start_from,
        prior_results=prior_results,
    )

    # Final summary
    gate3 = results.get("gate3_passed", False)
    logger.info("\n" + "="*70)
    logger.info(f"  PIPELINE COMPLETE — Gate 3: {'PASSED ✅' if gate3 else 'FAILED ❌'}")
    logger.info("="*70)
    for run in results.get("runs", []):
        es = run.get("eval_summary", {})
        sr = es.get("recovery_success_rate", "N/A")
        spl = es.get("recovery_spl", "N/A")
        label = run.get("label", "?")
        logger.info(f"  {label:45s}  R-SR={sr if isinstance(sr, str) else f'{sr:.2f}%':>10}  R-SPL={spl if isinstance(spl, str) else f'{spl:.2f}%':>10}")


if __name__ == "__main__":
    main()
