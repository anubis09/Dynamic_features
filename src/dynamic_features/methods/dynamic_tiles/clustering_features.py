import hashlib
from typing import List, cast

import numpy as np
from joblib import Parallel, delayed
from mushroom_rl.core import Environment

from .kmeans_tiling import KMeansTilingFitter


class ClusteringFeatures:
    """Encodes RL states using multiple independent KMeans-based tilings.

    Each tiling partitions the state space into clusters and encodes a state
    as a one-hot vector. Multiple tilings with different random seeds produce
    overlapping partitions, yielding a richer feature representation when
    concatenated.

    The final encoding for a state is a binary vector of shape
    (n_tilings * n_clusters,) where exactly n_tilings bits are active.

    Attributes:
        _n_tilings: Number of independent tilings.
        _n_clusters: Number of clusters per tiling.
        _seed: Base random seed used to derive per-tiling seeds.
        _tilings: List of fitted KMeansTilingFitter instances.

    """

    __slots__ = ["_n_tilings", "_n_clusters", "_tilings", "_seed"]

    def __init__(self, n_tilings: int, n_clusters: int, seed: int) -> None:
        """Initializes the ClusterFeatures.

        Args:
            n_tilings: Number of overlapping tilings.
            n_clusters: Number of clusters (tiles) per tiling.
            seed: Base random seed. Each tiling derives a unique seed
                via hashing to avoid collisions across runs.
        """
        self._n_tilings: int = n_tilings
        self._n_clusters: int = n_clusters
        self._seed: int = seed
        self._tilings = [
            KMeansTilingFitter(n_clusters=n_clusters, seed=self._derive_seed(seed=seed, tiling_index=tiling_index))
            for tiling_index in range(n_tilings)
        ]

    def _derive_seed(self, seed: int, tiling_index: int) -> int:
        """Derives a unique seed for a tiling by hashing the base seed and index.

        Args:
            seed: Base random seed for the run.
            tiling_index: Index of the tiling.

        Returns:
            A deterministic seed in the range [0, 2^32 - 1].
        """
        h = hashlib.md5(f"{seed}_{tiling_index}".encode()).hexdigest()
        return int(h, 16) % (2**32)

    def train(self, mdp: Environment, n_states_for_train: int) -> None:
        """Trains all tilings on randomly sampled states from the environment.

        Args:
            mdp: The environment to sample states from.
            n_states_for_train: Number of random states to sample per tiling.
        """
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(tiling.train)(mdp=mdp, n_states_for_train=n_states_for_train) for tiling in self._tilings
        )

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encodes states as concatenated one-hot vectors across all tilings.

        Args:
            X: State array of shape (n_samples, n_features).

        Returns:
            One-hot encoded array of shape (n_samples, n_tilings * n_clusters).
        """
        out = []
        for tiling in self._tilings:
            out.append(tiling.encode(X=X))
        return np.concatenate(out, axis=-1)

    def _p_encode(self, X: np.ndarray) -> np.ndarray:
        """Encodes states as concatenated one-hot vectors across all tilings.

        Args:
            X: State array of shape (n_samples, n_features).

        Returns:
            One-hot encoded array of shape (n_samples, n_tilings * n_clusters).
        """
        out = cast(
            List[np.ndarray],
            Parallel(n_jobs=-1, prefer="threads")(delayed(tiling.encode)(X=X) for tiling in self._tilings),
        )
        return np.concatenate(out, axis=-1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Calls encode. See `encode` for details."""
        return self.encode(X=X)

    @property
    def size(self):
        """int: Feature vector dimensionality for 1 sample."""
        return self._n_tilings * self._n_clusters
