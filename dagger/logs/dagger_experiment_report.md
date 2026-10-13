# DAgger Recovery Experiment — Summary Report

Generated: 2026-10-14 03:02:07

## Recovery Metrics

| Run | R-SR (%) | R-SPL (%) | Soft-SPL (%) | Dist Closed (m) |
|-----|----------|-----------|--------------|-----------------|
| **Baseline A (π₀ base)** | 50.00 | 50.00 | 16.99 | 0.297 |
| **Baseline B (expert-only β=1.0)** | 0.00 | 0.00 | 0.35 | -0.001 |
| **DAgger Iter 0 (β=0.7)** | 0.00 | 0.00 | 0.56 | 0.013 |
| **DAgger Iter 1 (β=0.3)** | 0.00 | 0.00 | 0.53 | 0.013 |
| **DAgger Iter 2 (β=0.0)** | 0.00 | 0.00 | 0.51 | 0.012 |

---

## Gate 3 Verdict: ❌ FAILED

> DAgger did not achieve statistically significant improvement over both baselines.
> Consider: more iterations, more failure cases, or hyperparameter tuning.

## Checkpoint Paths

- **Baseline A (π₀ base)**: `./libs/control/object_react/train/logs/object_rel_nav/object_react_hm3d_2026_10_07_20_54_34_/latest.pth`
- **Baseline B (expert-only β=1.0)**: `/scratch2/akshat.gupta/ak-object-rel-nav/dagger/checkpoints/policy_iter_-1.pth`
- **DAgger Iter 0 (β=0.7)**: `/scratch2/akshat.gupta/ak-object-rel-nav/dagger/checkpoints/policy_iter_0.pth`
- **DAgger Iter 1 (β=0.3)**: `/scratch2/akshat.gupta/ak-object-rel-nav/dagger/checkpoints/policy_iter_1.pth`
- **DAgger Iter 2 (β=0.0)**: `/scratch2/akshat.gupta/ak-object-rel-nav/dagger/checkpoints/policy_iter_2.pth`
