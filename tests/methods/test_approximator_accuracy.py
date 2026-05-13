import numpy as np
import pytest
from mushroom_rl.core import Environment

from dynamic_features.configs import SarsaLambdaDTConfig
from dynamic_features.methods import ClusteringFeatures, get_states_from_mdp

N_TILINGS = 10
N_CLUSTERS = 20
SEED = 42
ACCURACY_THRESHOLD = 0.90
AGENT_CONF = SarsaLambdaDTConfig().agent
MULTIPLIER = AGENT_CONF.clustering_training_states_multiplier

APPROXIMATOR_CONFIG = AGENT_CONF.approximator


@pytest.mark.slow
@pytest.mark.parametrize(
    "env_name",
    [
        ("Gym.MountainCar-v0"),
        ("Gym.CartPole-v1"),
        ("Gym.Acrobot-v1"),
        ("Gym.LunarLander-v2"),
        ("PuddleWorld"),
    ],
)
def test_approximator_accuracy(env_name):
    """Approximator trained with production hyperparameters must reach >90% accuracy."""
    env = Environment.make(env_name)
    n_states = MULTIPLIER * N_CLUSTERS
    input_shape = env.info.observation_space.shape[0]

    tilings = ClusteringFeatures(
        input_shape=input_shape,
        n_tilings=N_TILINGS,
        n_clusters=N_CLUSTERS,
        seed=SEED,
        approximator_config=APPROXIMATOR_CONFIG,
    )

    X_list = [get_states_from_mdp(mdp=env, n_states=n_states) for _ in range(N_TILINGS)]
    tilings.train(X_per_tilings=X_list)

    X_train = np.concatenate(X_list, axis=0)
    X_test = get_states_from_mdp(mdp=env, n_states=n_states)
    _, overall_acc = tilings.train_approximator(X=X_train, X_test=X_test)

    print(f"\n[{env_name}] approximator accuracy: {overall_acc:.2%}")

    assert overall_acc > ACCURACY_THRESHOLD, (
        f"{env_name}: expected accuracy > {ACCURACY_THRESHOLD:.0%}, got {overall_acc:.2%}"
    )
