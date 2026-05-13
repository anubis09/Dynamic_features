import pytest
from mushroom_rl.core import Environment

from dynamic_features.configs import ApproximatorConfig, Losses, Optimizers
from dynamic_features.methods import ClusteringFeatures, KMeansTilingFitter, get_states_from_mdp

N_CLUSTERS = 10
N_TILINGS = 10
N_TRAIN = 200
N_TEST = 50
SEED = 42
INPUT_SHAPE = 2  # MountainCar-v0 observation dim

APPROXIMATOR_CONFIG = ApproximatorConfig(
    n_hidden_layers=1,
    hidden_dim=16,
    dropout=0.0,
    batch_size=64,
    lr=1e-3,
    eval_size=0.1,
    loss=Losses.CE,
    optimizer=Optimizers.ADAM,
    n_epochs=1,
    patience=5,
    min_delta=0.005,
)


@pytest.fixture(scope="session")
def env():
    return Environment.make("Gym.MountainCar-v0")


@pytest.fixture(scope="session")
def test_states(env):
    return get_states_from_mdp(mdp=env, n_states=N_TEST)


@pytest.fixture(scope="session")
def train_states(env):
    return get_states_from_mdp(mdp=env, n_states=N_TEST)


@pytest.fixture(scope="session")
def trained_tiling(train_states):
    fitter = KMeansTilingFitter(n_clusters=N_CLUSTERS, seed=SEED)
    fitter.train(X=train_states)
    return fitter


@pytest.fixture(scope="session")
def trained_tilings(env):
    fitter = ClusteringFeatures(
        input_shape=INPUT_SHAPE,
        n_tilings=N_TILINGS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
        approximator_config=APPROXIMATOR_CONFIG,
    )
    X = [get_states_from_mdp(mdp=env, n_states=N_TRAIN) for _ in range(N_TILINGS)]
    fitter.train(X_per_tilings=X)
    return fitter
