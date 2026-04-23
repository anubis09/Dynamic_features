import pytest
from mushroom_rl.core import Environment

from dynamic_features.methods import ClusteringFeatures, KMeansTilingFitter, get_states_from_mdp

N_CLUSTERS = 5
N_TILINGS = 3
N_TRAIN = 200
N_TEST = 50
SEED = 42


@pytest.fixture(scope="session")
def env():
    return Environment.make("Gym.MountainCar-v0")


@pytest.fixture(scope="session")
def test_states(env):
    return get_states_from_mdp(mdp=env, n_states=N_TEST)


@pytest.fixture(scope="session")
def trained_tiling(env):
    fitter = KMeansTilingFitter(n_clusters=N_CLUSTERS, seed=SEED)
    fitter.train(mdp=env, n_states_for_train=N_TRAIN)
    return fitter


@pytest.fixture(scope="session")
def trained_tilings(env):
    fitter = ClusteringFeatures(n_tilings=N_TILINGS, n_clusters=N_CLUSTERS, seed=SEED)
    fitter.train(mdp=env, n_states_for_train=N_TRAIN)
    return fitter
