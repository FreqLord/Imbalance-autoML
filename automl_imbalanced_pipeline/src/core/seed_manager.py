"""Deterministic hierarchical seed derivation using numpy.random.SeedSequence.

Ensures independent random streams for:
- Data partitioners and cross-validation splitters
- Optuna samplers and trial orchestrators
- Preprocessing and resampling transforms (SMOTE, SMOTENC)
- Downstream machine learning estimators
"""

import warnings
from typing import Dict, List, Optional, Union
import numpy as np


class SeedManager:
    """Manages independent, non-overlapping entropy streams derived from a single master seed.

    Uses NumPy's SeedSequence spawn mechanism to generate independent child PRNG states.
    """

    def __init__(self, master_seed: Optional[int] = 42):
        self.master_seed = master_seed
        self.root_sequence = np.random.SeedSequence(master_seed)
        self._spawned_registry: Dict[str, np.random.SeedSequence] = {}

    def get_child_sequence(self, domain: str) -> np.random.SeedSequence:
        """Derives a dedicated child SeedSequence for a subsystem domain."""
        if domain not in self._spawned_registry:
            self._spawned_registry[domain] = self.root_sequence.spawn(1)[0]
        return self._spawned_registry[domain]

    def derive_int_seed(self, domain: str, sub_index: int = 0) -> int:
        """Derives a 32-bit integer seed for libraries requiring integer seeds (scikit-learn, Optuna, etc.)."""
        child_seq = self.get_child_sequence(domain)
        # Spawn an index-specific child to maintain isolation
        sub_child = child_seq.spawn(sub_index + 1)[sub_index]
        return int(sub_child.generate_state(1, dtype=np.uint32)[0])

    def derive_rng(self, domain: str, sub_index: int = 0) -> np.random.Generator:
        """Derives an independent NumPy Generator for array operations and bootstrapping."""
        child_seq = self.get_child_sequence(domain)
        sub_child = child_seq.spawn(sub_index + 1)[sub_index]
        return np.random.default_rng(sub_child)

    def spawn_fold_seeds(self, n_folds: int, domain: str = "cv_split") -> List[int]:
        """Generates deterministic seeds for each fold in cross-validation."""
        return [self.derive_int_seed(domain, sub_index=i) for i in range(n_folds)]

    @staticmethod
    def audit_multithreading_determinism(n_jobs: int, model_name: str) -> None:
        """Emits an advisory warning if multi-threading is enabled for algorithms with non-deterministic reduction."""
        tree_models = ["lightgbm", "catboost", "xgboost", "random_forest"]
        if n_jobs != 1 and any(m in model_name.lower() for m in tree_models):
            warnings.warn(
                f"Model '{model_name}' configured with n_jobs={n_jobs}. "
                "Floating-point summation order across multiple worker threads may introduce minor run-to-run discrepancies "
                "despite deterministic seeding.",
                UserWarning,
                stacklevel=2,
            )
