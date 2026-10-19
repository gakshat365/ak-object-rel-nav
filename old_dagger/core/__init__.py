"""
Core modules for Iterative DAgger failure recovery.
"""

from old_dagger.core.failure_loader import FailureLoader, FailureCase
from old_dagger.core.expert_oracle import HabitatExpertOracle, OracleOutput
from old_dagger.core.rollout_collector import RolloutCollector
from old_dagger.core.data_aggregator import DataAggregator, DAggerRecoveryDataset
from old_dagger.core.trainer import DAggerTrainer

__all__ = [
    "FailureLoader",
    "FailureCase",
    "HabitatExpertOracle",
    "OracleOutput",
    "RolloutCollector",
    "DataAggregator",
    "DAggerRecoveryDataset",
    "DAggerTrainer",
]
