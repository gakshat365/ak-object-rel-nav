"""
Core modules for Iterative DAgger failure recovery.
"""

from dagger.core.failure_loader import FailureLoader, FailureCase
from dagger.core.expert_oracle import HabitatExpertOracle, OracleOutput
from dagger.core.rollout_collector import RolloutCollector
from dagger.core.data_aggregator import DataAggregator, DAggerRecoveryDataset

__all__ = [
    "FailureLoader",
    "FailureCase",
    "HabitatExpertOracle",
    "OracleOutput",
    "RolloutCollector",
    "DataAggregator",
    "DAggerRecoveryDataset",
]
