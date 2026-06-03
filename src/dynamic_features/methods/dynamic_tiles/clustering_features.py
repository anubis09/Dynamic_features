import hashlib
from logging import DEBUG, Logger, getLogger
from typing import List, Tuple, cast

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from dynamic_features.backbones import MultiHeadMLP, Trainer
from dynamic_features.configs import ApproximatorConfig

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

    __slots__ = [
        "_n_tilings",
        "_n_clusters",
        "_tilings",
        "_approximator",
        "_approximator_config",
        "_seed",
        "_scaler",
        "_logger",
    ]

    def __init__(
        self, input_shape: int, n_tilings: int, n_clusters: int, seed: int, approximator_config: ApproximatorConfig
    ) -> None:
        """Initializes ClusteringFeatures with a set of KMeans tilings and a neural approximator.

        Args:
            input_shape: Number of features in each state observation.
            n_tilings: Number of independent overlapping tilings to build.
            n_clusters: Number of clusters (tiles) per tiling.
            seed: Base random seed. A unique seed for each tiling is
                deterministically derived via MD5 hashing so that tiling
                seeds never collide across runs or across tiling indices.
            approximator_config: Configuration for the MultiHeadMLP
                approximator, including architecture, optimizer, and
                training hyperparameters.
        """
        self._n_tilings: int = n_tilings
        self._n_clusters: int = n_clusters
        self._seed: int = seed
        self._tilings = [
            KMeansTilingFitter(n_clusters=n_clusters, seed=self._derive_seed(seed=seed, tiling_index=tiling_index))
            for tiling_index in range(n_tilings)
        ]
        self._scaler: StandardScaler = StandardScaler()
        self._approximator_config = approximator_config
        self._approximator = MultiHeadMLP(
            n_input=input_shape,
            n_hidden_layers_backbone=self._approximator_config.n_hidden_layers,
            hidden_dim=self._approximator_config.hidden_dim,
            n_heads=self._n_tilings,
            n_output_per_head=self._n_clusters,
            dropout=self._approximator_config.dropout,
        )
        self._logger: Logger = getLogger(self.__class__.__name__)
        self._logger.setLevel(DEBUG)

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

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Applies the fitted StandardScaler, preserving the input dimensionality."""
        is_1d = X.ndim == 1
        X_scaled = cast(np.ndarray, self._scaler.transform(np.atleast_2d(X)))
        return X_scaled.squeeze(0) if is_1d else X_scaled

    def train(self, X_per_tilings: List[np.ndarray]) -> None:
        """Fits all KMeans tilings in parallel on their respective state samples.

        The internal ``StandardScaler`` is refit on the concatenation of all
        per-tiling arrays on every call, so the scaler, the KMeans centroids and
        any subsequent approximator training/inference all live in the same
        normalized space. When called to adapt to a shifted state distribution,
        the approximator must be retrained afterwards, since its previous fit
        assumes the old scale and cluster assignments.

        Args:
            X_per_tilings: List of state arrays, one per tiling, each of shape
                (n_samples, n_features). The i-th array is passed to the i-th
                tiling's ``train`` method. Must have exactly ``n_tilings`` elements.
        """
        assert len(X_per_tilings) == self._n_tilings
        self._logger.info("Starting training of each tiling.")
        self._scaler.fit(np.concatenate(X_per_tilings, axis=0))
        X_normalized_per_tilings = [self._scaler.transform(X) for X in X_per_tilings]
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(tiling.train)(X=X) for X, tiling in zip(X_normalized_per_tilings, self._tilings)
        )
        self._logger.info("Train over.")

    def train_approximator(self, X: np.ndarray, X_test: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Trains all MultiHeadMLP parameters (backbone + heads) on the current KMeans assignments.

        Head weights are always reset before training because cluster label assignments are
        arbitrary after each KMeans refit, making old head weights meaningless. The backbone
        is kept as a warm start and fine-tuned together with the heads.

        Args:
            X: Training state array of shape (n_samples, n_features). Cast to
                float32 internally before tensor conversion.
            X_test: Hold-out state array of shape (n_test_samples, n_features)
                used only for post-training accuracy evaluation; it does not
                influence training or early stopping.

        Returns:
            A tuple ``(per_head_acc, overall_acc)`` evaluated on ``X_test``, where:
                - ``per_head_acc``: Tensor of shape ``(n_tilings,)`` with per-head accuracy.
                - ``overall_acc``: Macro-average as a Python float.
        """
        self._approximator.reset_heads()
        optimizer = self._approximator_config.optimizer.get_class()
        optimizer = optimizer(params=self._approximator.parameters(), lr=self._approximator_config.lr)
        return self._fit_approximator(X=X, X_test=X_test, optimizer=optimizer)

    def _fit_approximator(
        self, X: np.ndarray, X_test: np.ndarray, optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, float]:
        """Core training procedure: fits pseudo-labels, normalizes, and runs the Trainer with early stopping.

        Generates pseudo-labels from the currently fitted KMeans tilings via
        ``predict``, normalizes ``X`` with the existing scaler, splits into
        train/validation sets, and runs the ``Trainer`` with early stopping.
        The caller is responsible for configuring which parameters the optimizer covers.

        Args:
            X: Training state array of shape (n_samples, n_features). Cast to
                float32 internally before tensor conversion.
            X_test: Hold-out state array of shape (n_test_samples, n_features)
                used only for post-training accuracy evaluation; does not
                influence training or early stopping.
            optimizer: A fully configured PyTorch optimizer. Its parameter groups
                determine which parts of the model are updated.

        Returns:
            A tuple ``(per_head_acc, overall_acc)`` evaluated on ``X_test``, where:
                - ``per_head_acc``: Tensor of shape ``(n_tilings,)`` with per-head accuracy.
                - ``overall_acc``: Macro-average as a Python float.
        """
        loss_fn = self._approximator_config.loss.get_function()

        def loss(preds: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
            # preds: (batch, n_heads, n_out_per_head); ys: (batch, n_heads) class indices
            n_heads = preds.shape[1]
            head_losses = [loss_fn(preds[:, n_head, :], ys[:, n_head]) for n_head in range(n_heads)]
            return torch.stack(head_losses).mean()

        y = self.predict(X).astype(np.int64)

        X_scaled = self._normalize(X).astype(np.float32)
        X_train, X_eval, y_train, y_eval = train_test_split(
            X_scaled, y, test_size=self._approximator_config.eval_size, random_state=42
        )

        train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_dataloader = DataLoader(dataset=train_ds, batch_size=self._approximator_config.batch_size)

        eval_ds = TensorDataset(torch.from_numpy(X_eval), torch.from_numpy(y_eval))
        eval_dataloader = DataLoader(dataset=eval_ds, batch_size=self._approximator_config.batch_size)

        trainer = Trainer(model=self._approximator, optimizer=optimizer, loss=loss)
        trainer.train(
            train_dataloader=train_dataloader,
            n_epochs=self._approximator_config.n_epochs,
            eval_dataloader=eval_dataloader,
            patience=self._approximator_config.patience,
            min_delta=self._approximator_config.min_delta,
        )

        per_head_acc, accuracy = self._evaluate_approximator(X=X_test)
        self._logger.debug(f"Per head accuracy score: {per_head_acc}")
        self._logger.debug(f"Accuracy score: {accuracy}")
        return per_head_acc, accuracy

    def _evaluate_approximator(self, X: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Evaluates the approximator's cluster-prediction accuracy on a state array.

        Generates ground-truth cluster labels from the fitted KMeans tilings via
        ``predict``, runs the approximator in inference mode, and computes
        per-head classification accuracy.

        Args:
            X: State array of shape (n_samples, n_features). Cast to float32
                internally before tensor conversion.

        Returns:
            A tuple ``(per_head_acc, overall_acc)`` where:
                - ``per_head_acc``: Tensor of shape ``(n_tilings,)`` with the
                  fraction of correctly predicted clusters for each tiling head.
                - ``overall_acc``: Macro-average of ``per_head_acc`` as a Python float.
        """
        y_test = self.predict(X=X)
        test_ds = TensorDataset(torch.from_numpy(self._normalize(X).astype(np.float32)), torch.from_numpy(y_test))
        test_dataloader = DataLoader(dataset=test_ds, batch_size=128)

        all_preds = []
        with torch.inference_mode():
            for X_batch, _ in test_dataloader:
                logits = self._approximator(X_batch)  # (batch, n_heads, n_clusters)
                all_preds.append(logits.argmax(dim=-1).cpu())  # (batch, n_heads)

        preds = torch.cat(all_preds)  # (n_test, n_heads)
        targets = torch.from_numpy(y_test)  # (n_test, n_heads)
        per_head_acc = (preds == targets).float().mean(dim=0)  # (n_heads,)
        overall_acc = per_head_acc.mean().item()
        return per_head_acc, overall_acc

    def encode_approx(self, X: np.ndarray) -> np.ndarray:
        """Encodes states using the trained neural approximator instead of exact KMeans lookup.

        Runs the MultiHeadMLP to predict a cluster index per tiling head and
        returns the result as a concatenated one-hot encoding. Semantically
        equivalent to ``encode`` but avoids the KMeans distance computation at
        inference time. Requires ``train_approximator`` to have been called first.

        Args:
            X: State array of shape (n_samples, n_features) or (n_features,)
                for a single state. Cast to float32 internally before tensor conversion.

        Returns:
            One-hot encoded array of shape (n_samples, n_tilings * n_clusters)
            for batch input, or (n_tilings * n_clusters,) for a single state.
        """
        X_tensor = torch.from_numpy(self._normalize(X).astype(np.float32))

        with torch.inference_mode():
            # logits: (n_samples, n_heads, n_clusters) or (n_heads, n_clusters)
            logits = self._approximator(X_tensor)
            classes = logits.argmax(dim=-1)
        preds = torch.nn.functional.one_hot(classes, num_classes=self._n_clusters)
        # flatten last two dims to produce a single concatenated one-hot vector
        preds = preds.flatten(start_dim=-2)
        return preds.numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster indices for each state across all tilings.

        Args:
            X: State array of shape (n_samples, n_features) or of shape (n_features,).

        Returns:
            Cluster indices of shape (n_samples, n_tilings) or of shape (n_tilings,), where each column
            contains the cluster assignment from one tiling.
        """
        X_norm = self._normalize(X)
        out = []
        for tiling in self._tilings:
            out.append(tiling.predict(X=X_norm))
        return np.stack(out, axis=-1)

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encodes states as concatenated one-hot vectors across all tilings.

        Args:
            X: State array of shape (n_samples, n_features), or of shape (n_features,).

        Returns:
            One-hot encoded array of shape (n_samples, n_tilings * n_clusters) or of shape (n_tilings * n_clusters,).
        """
        X_norm = self._normalize(X)
        out = []
        for tiling in self._tilings:
            out.append(tiling.encode(X=X_norm))
        return np.concatenate(out, axis=-1)

    def _p_encode(self, X: np.ndarray) -> np.ndarray:
        """Parallelised variant of ``encode`` that runs all tilings concurrently.

        Uses joblib threads to call each tiling's ``encode`` method in parallel,
        then concatenates the results. Produces output identical to ``encode``;
        prefer this for large batches where tiling inference is the bottleneck.

        Args:
            X: State array of shape (n_samples, n_features), or of shape (n_features,).

        Returns:
            One-hot encoded array of shape (n_samples, n_tilings * n_clusters)
            for batch input, or (n_tilings * n_clusters,) for a single state.
        """
        X_norm = self._normalize(X)
        out = cast(
            List[np.ndarray],
            Parallel(n_jobs=-1, prefer="threads")(delayed(tiling.encode)(X=X_norm) for tiling in self._tilings),
        )
        return np.concatenate(out, axis=-1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Calls encode. See `encode` for details."""
        return self.encode(X=X)

    @property
    def size(self):
        """int: Feature vector dimensionality for 1 sample."""
        return self._n_tilings * self._n_clusters
