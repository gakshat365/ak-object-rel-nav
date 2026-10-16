"""
Backward compatibility stub for rollout_collector_optimised.py.
All logic has been consolidated into dagger.core.rollout_collector.
"""

from dagger.core.rollout_collector import RolloutCollector, OptimizedRolloutCollector

__all__ = ["RolloutCollector", "OptimizedRolloutCollector"]
