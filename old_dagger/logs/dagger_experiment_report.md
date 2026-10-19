# DAgger Recovery Experiment — Summary Report

Generated: 2026-10-15 01:32:09

## Recovery Metrics

| Run | R-SR (%) | R-SPL (%) | Soft-SPL (%) | Dist Closed (m) |
|-----|----------|-----------|--------------|-----------------|
| **Baseline B (expert-only β=1.0)** | 0.00 | 0.00 | 0.73 | 0.016 |
| **DAgger Iter 0 (β=0.5)** | 0.00 | 0.00 | 16.14 | 1.049 |
| **DAgger Iter 1 (β=0.0)** | 0.00 | 0.00 | 16.20 | 1.050 |

---

## Gate 3 Verdict: ❌ FAILED

> DAgger did not achieve statistically significant improvement over both baselines.
> Consider: more iterations, more failure cases, or hyperparameter tuning.

## Checkpoint Paths

- **Baseline B (expert-only β=1.0)**: `/scratch2/akshat.gupta/ak-object-rel-nav/dagger/checkpoints/policy_iter_-1.pth`
- **DAgger Iter 0 (β=0.5)**: `/scratch2/akshat.gupta/ak-object-rel-nav/dagger/checkpoints/policy_iter_0.pth`
- **DAgger Iter 1 (β=0.0)**: `/scratch2/akshat.gupta/ak-object-rel-nav/dagger/checkpoints/policy_iter_1.pth`
