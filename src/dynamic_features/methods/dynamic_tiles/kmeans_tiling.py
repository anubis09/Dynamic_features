from logging import DEBUG, Logger, getLogger

import numpy as np
from mushroom_rl.core import Environment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .utils import get_states_from_mdp


class KMeansTilingFitter:
    __slots__ = ["_n_clusters", "_scaler", "_seed", "_logger", "_kmeans"]

    def __init__(self, n_clusters: int, seed: int) -> None:
        """Initializes the TilingFitter.

        Args:
            n_clusters: Number of clusters (tiles) to partition the state space.
            seed: Random seed for reproducibility.

        Note:
            Calls `_setup_clustering_algorithm` at the end of init. Subclass
            constructors must set any attributes required by that method before
            calling `super().__init__`.
        """
        self._n_clusters: int = n_clusters
        self._scaler: StandardScaler = StandardScaler()
        self._seed: int = seed
        self._kmeans = KMeans(n_clusters=self._n_clusters, random_state=self._seed, n_init="auto")
        self._logger: Logger = getLogger(self.__class__.__name__)
        self._logger.setLevel(DEBUG)

    def train(self, mdp: Environment, n_states_for_train: int) -> None:
        """Fit the scaler and clustering algorithm on environment samples.

        Args:
            mdp: The environment to sample states from.
            n_states_for_train: Number of states used for fitting.
        """
        self._logger.debug(f"Training tiling with {n_states_for_train} states, {self._n_clusters} clusters")
        X = get_states_from_mdp(mdp=mdp, n_states=n_states_for_train)
        X_normalized = self._scaler.fit_transform(X)
        self._kmeans.fit(X_normalized)
        self._logger.debug("Training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign each state to its tile index.

        Args:
            X: State array of shape (n_samples, n_features) or
                (n_features,) for a single state.

        Returns:
            One-hot encoded array of shape (n_clusters,) for single state input,
            or (n_samples, n_clusters) for batch input.
        """
        is_1d = X.ndim == 1
        X = np.atleast_2d(X)
        X_normalized = self._scaler.transform(X)
        indexes = self._kmeans.predict(X_normalized)
        out = np.zeros((len(indexes), self._n_clusters))
        # out[:, indexes]=1 would put 1 in the indexes place for EACH row and not 1 for row.
        out[np.arange(len(indexes)), indexes] = 1
        if is_1d:
            out = out.squeeze(0)
        return out

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Calls predict. See `predict` for details."""
        return self.predict(X)
