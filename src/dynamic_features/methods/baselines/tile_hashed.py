"""Tile coding with hashing for SARSA(λ)."""

import numpy as np

from .tiles3_sutton import IHT, tiles


class HashingRecTilesFeatures:
    """Rectangular Tiles with IHT hashing.

    Maps continuous state (+ optional discrete action) to sparse active tile
    indices or dense binary feature vector. Uses Sutton's IHT for
    collision-handled index assignment.

    Attributes:
        size: Hash table size (feature vector dimensionality).
        num_active_features: Number of active tiles per encoding (= tilings).
    """

    def __init__(self, hash_size, tilings, state_low, state_high, tiles_per_dim, action_values=None, epsilon=1e-10):
        """Initializes the hashing tile coder.

        Args:
            hash_size: Size of the IHT. Determines feature vector length.
                Should be >> tilings * prod(tiles_per_dim) * num_actions
                to minimize collisions.
            tilings: Number of overlapping tilings.
            state_low: Lower bounds for each state dimension. Shape (state_dim,).
            state_high: Upper bounds for each state dimension. Shape (state_dim,).
            tiles_per_dim: Number of tiles per dimension. Either a single int
                (same for all dims) or a list/array of length state_dim.
            action_values: Discrete action values for state-action tiling.
                E.g., [0, 1, 2, 3] for LunarLander. None for state-only tiling.
            epsilon: Small offset to prevent state_high from landing on the
                next tile boundary.
        """
        self._iht = IHT(hash_size)
        self._hash_size = hash_size
        self._tilings = tilings
        self._epsilon = epsilon

        self._state_low = np.asarray(state_low, dtype=np.float64)
        self._state_high = np.asarray(state_high, dtype=np.float64)
        self._state_dim = len(self._state_low)

        if isinstance(tiles_per_dim, (int, float)):
            tiles_per_dim = np.full(self._state_dim, tiles_per_dim, dtype=np.float64)
        else:
            tiles_per_dim = np.asarray(tiles_per_dim, dtype=np.float64)

        # Scale factor: converts state coordinates to tile coordinates.
        # Sutton's tiles() grids at unit intervals, so we scale each dimension
        # such that the full range maps to tiles_per_dim units.
        self._scale = tiles_per_dim / (self._state_high - self._state_low)

        if action_values is not None:
            self._action_to_idx = {a: i for i, a in enumerate(action_values)}
        else:
            self._action_to_idx = None

    def get_active_tiles(self, state, action=None):
        """Returns the indices of active tiles for a given state(-action) pair.

        This is the efficient path for SARSA(λ): operate directly on sparse
        indices (e.g., w[active].sum() for Q-value, e[active] += 1 for traces)
        instead of building a dense feature vector.

        Args:
            state: State vector. Shape (state_dim,).
            action: Optional discrete action value. Must be in action_values
                if provided at init.

        Returns:
            List of int tile indices, length tilings. Each index is in
            [0, hash_size).

        Raises:
            ValueError: If action is provided but action_values was not set
                at init.
            KeyError: If action is not in action_values.
        """
        state = np.asarray(state, dtype=np.float64)
        clipped = np.clip(state, self._state_low, self._state_high - self._epsilon)
        scaled = (clipped - self._state_low) * self._scale

        ints = []
        if action is not None:
            if self._action_to_idx is None:
                raise ValueError("action_values must be provided at init to use actions")
            key = action.item() if isinstance(action, np.generic) else action
            ints = [self._action_to_idx[key]]

        return tiles(self._iht, self._tilings, scaled.tolist(), ints)

    def __call__(self, state, action=None):
        """Returns a dense binary feature vector for a given state(-action) pair.

        Args:
            state: State vector. Shape (state_dim,).
            action: Optional discrete action value.

        Returns:
            Binary feature vector of shape (hash_size,) with exactly
            tilings ones (fewer if hash collisions occur).
        """
        active = self.get_active_tiles(state, action)
        features = np.zeros(self._hash_size, dtype=np.float32)
        features[active] = 1.0
        return features

    def reset_iht(self):
        """Resets the index hash table, clearing all learned tile mappings."""
        self._iht = IHT(self._hash_size)

    @property
    def size(self):
        """int: Feature vector dimensionality (= hash_size)."""
        return self._hash_size

    @property
    def num_active_features(self):
        """int: Number of active features per encoding (= tilings)."""
        return self._tilings
