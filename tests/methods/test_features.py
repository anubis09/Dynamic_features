from typing import Callable

import numpy as np
from conftest import N_CLUSTERS, N_TILINGS, SEED

# ==================== Helpers ====================


def assert_2d_shape(fn: Callable, states: np.ndarray, expected_width: int) -> None:
    """Output shape is (n_samples, expected_width)."""
    out = fn(states)
    assert out.shape == (len(states), expected_width)


def assert_single_state_shape(fn: Callable, states: np.ndarray, expected_width: int) -> None:
    """A single state still returns a 2D array with correct width."""
    out = fn(states[:1])
    assert out.shape == (1, expected_width)


def assert_binary_output(fn: Callable, states: np.ndarray) -> None:
    """Every value in the output is either 0 or 1."""
    out = fn(states)
    assert np.all((out == 0) | (out == 1))


def assert_1d_state_shape(fn: Callable, state: np.ndarray, expected_width: int) -> None:
    """A single 1D state returns a 1D array with correct width."""
    assert state.ndim == 1
    out = fn(state)
    assert out.shape == (expected_width,)


# ==================== KMeansTilingFitter ====================


class TestKMeansTilingFitter:
    def test_encode_shape(self, trained_tiling, test_states):
        assert_2d_shape(trained_tiling.encode, test_states, N_CLUSTERS)

    def test_single_state_shape(self, trained_tiling, test_states):
        assert_single_state_shape(trained_tiling.encode, test_states, N_CLUSTERS)

    def test_binary_output(self, trained_tiling, test_states):
        assert_binary_output(trained_tiling.encode, test_states)

    def test_one_active_tile_per_state(self, trained_tiling, test_states):
        """Each state activates exactly one tile."""
        out = trained_tiling.encode(test_states)
        assert np.all(out.sum(axis=1) == 1)

    def test_1d_single_state_shape(self, trained_tiling, test_states):
        assert_1d_state_shape(trained_tiling.encode, test_states[0], N_CLUSTERS)

    def test_predict_shape(self, trained_tiling, test_states):
        out = trained_tiling.predict(test_states)
        assert out.shape == (len(test_states),)


# ==================== ClusteringFeatures ====================


class TestClusteringFeatures:
    def test_encode_shape(self, trained_tilings, test_states):
        assert_2d_shape(trained_tilings.encode, test_states, N_TILINGS * N_CLUSTERS)

    def test_single_state_shape(self, trained_tilings, test_states):
        assert_single_state_shape(trained_tilings.encode, test_states, N_TILINGS * N_CLUSTERS)

    def test_binary_output(self, trained_tilings, test_states):
        assert_binary_output(trained_tilings.encode, test_states)

    def test_total_active_tiles(self, trained_tilings, test_states):
        """Each state activates exactly n_tilings tiles in total."""
        out = trained_tilings.encode(test_states)
        assert np.all(out.sum(axis=-1) == N_TILINGS)

    def test_one_active_per_tiling_block(self, trained_tilings, test_states):
        """Within each tiling block, exactly one tile is active per state."""
        out = trained_tilings.encode(test_states)
        for i in range(N_TILINGS):
            block = out[:, i * N_CLUSTERS : (i + 1) * N_CLUSTERS]
            assert np.all(block.sum(axis=1) == 1)

    def test_1d_single_state_shape(self, trained_tilings, test_states):
        assert_1d_state_shape(trained_tilings.encode, test_states[0], N_TILINGS * N_CLUSTERS)

    def test_1d_total_active_tiles(self, trained_tilings, test_states):
        """1D input: exactly n_tilings tiles active in the flat output."""
        out = trained_tilings.encode(test_states[0])
        assert out.sum() == N_TILINGS

    def test_tilings_produce_different_partitions(self, trained_tilings, test_states):
        """At least two tiling blocks assign states differently."""
        out = trained_tilings.encode(test_states)
        blocks = [out[:, i * N_CLUSTERS : (i + 1) * N_CLUSTERS] for i in range(N_TILINGS)]
        any_differ = any(
            not np.array_equal(blocks[i], blocks[j]) for i in range(N_TILINGS) for j in range(i + 1, N_TILINGS)
        )
        assert any_differ

    def test_predict_shape(self, trained_tilings, test_states):
        assert_2d_shape(trained_tilings.predict, test_states, N_TILINGS)

    def test_predict_values_in_range(self, trained_tilings, test_states):
        """All cluster indices are valid cluster labels."""
        out = trained_tilings.predict(test_states)
        assert np.all((out >= 0) & (out < N_CLUSTERS))


# ==================== Seed Derivation ====================


class TestDeriveSeed:
    def test_within_numpy_seed_range(self, trained_tilings):
        for i in range(200):
            assert 0 <= trained_tilings._derive_seed(seed=SEED, tiling_index=i) < 2**32

    def test_deterministic(self, trained_tilings):
        assert trained_tilings._derive_seed(seed=SEED, tiling_index=0) == trained_tilings._derive_seed(
            seed=SEED, tiling_index=0
        )

    def test_unique_across_tiling_indices(self, trained_tilings):
        seeds = [trained_tilings._derive_seed(seed=SEED, tiling_index=i) for i in range(10)]
        assert len(set(seeds)) == 10
