from typing import cast

import numpy as np
from sklearn.cluster import KMeans


class KMeansTilingFitter:
    """Partitions a pre-normalized state space into clusters and encodes states as one-hot vectors.

    Normalization is expected to be applied by the caller before passing data to
    ``train``, ``predict``, or ``encode``. Typically ``ClusteringFeatures`` owns
    the single shared ``StandardScaler`` and pre-normalizes all inputs.

    Typical usage::

        fitter = KMeansTilingFitter(n_clusters=16, seed=42)
        fitter.train(X_normalized)          # fit KMeans on pre-normalized data
        encoding = fitter.encode(state_normalized)  # one-hot vector of length n_clusters
    """

    __slots__ = ["_n_clusters", "_kmeans", "_seed"]

    def __init__(self, n_clusters: int, seed: int) -> None:
        """Initializes the KMeansTilingFitter.

        Args:
            n_clusters: Number of clusters (tiles) to partition the state space.
            seed: Random seed passed to KMeans for reproducibility.
        """
        self._n_clusters: int = n_clusters
        self._seed: int = seed
        self._kmeans = KMeans(n_clusters=self._n_clusters, random_state=self._seed, n_init="auto")

    def train(self, X: np.ndarray) -> None:
        """Fits the KMeans model on a set of pre-normalized environment state samples.

        Must be called before ``predict`` or ``encode``.

        Args:
            X: Pre-normalized state array of shape (n_samples, n_features).
        """
        self._kmeans.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns the nearest cluster index for each state.

        A 1-D input is promoted to 2-D internally via ``np.atleast_2d``, so
        the output is always at least 1-D.

        Args:
            X: Pre-normalized state array of shape (n_samples, n_features) or
                (n_features,) for a single state.

        Returns:
            Integer cluster indices of shape (n_samples,) for batch input,
            or shape (1,) for a single state.
        """
        X = np.atleast_2d(X)
        indexes = self._kmeans.predict(X)
        return cast(np.ndarray, indexes)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encodes states as one-hot vectors over the cluster vocabulary.

        Calls ``predict`` to obtain cluster indices, then builds a binary
        matrix with a single active bit per row at the predicted cluster
        position. The batch dimension is squeezed away for 1-D input so
        that a single state returns a flat vector.

        Args:
            X: State array of shape (n_samples, n_features) or
                (n_features,) for a single state.

        Returns:
            One-hot encoded array of shape (n_clusters,) for a single state,
            or (n_samples, n_clusters) for batch input.
        """
        is_1d = X.ndim == 1
        indexes = self.predict(X)
        out = np.zeros((len(indexes), self._n_clusters))
        # out[:, indexes]=1 would put 1 in the indexes place for EACH row and not 1 for row.
        out[np.arange(len(indexes)), indexes] = 1
        if is_1d:
            out = out.squeeze(0)
        return out
