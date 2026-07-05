import numpy as np
import pytest
from mushroom_rl.core import Environment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from dynamic_features.methods import get_states_from_mdp

SEED = 42
N_RUNS = 20

# Swept as a full grid (Cartesian product). n_states = multiplier * n_clusters.
N_CLUSTERS = [24, 48]
MULTIPLIERS = [50, 100, 200, 500, 1000]

ENV_NAMES = [
    "Gym.MountainCar-v0",
    "Gym.CartPole-v1",
    "Gym.Acrobot-v1",
    "Gym.LunarLander-v2",
    "PuddleWorld",
]


def mean_distortion(kmeans, X):
    """Mean distance from each sample in X to its assigned centroid."""
    return kmeans.transform(X).min(axis=1).mean()


@pytest.mark.slow
@pytest.mark.parametrize("env_name", ENV_NAMES)
@pytest.mark.parametrize("multiplier", MULTIPLIERS)
@pytest.mark.parametrize("n_clusters", N_CLUSTERS)
def test_kmeans_ari(request, env_name, n_clusters, multiplier):
    env = Environment.make(env_name)
    n_states = multiplier * n_clusters

    aris = []
    distortions = []
    for run in range(N_RUNS):
        # New sampling seed per run so the random-policy draws differ; averaging over runs
        # smooths out the sampling noise that a single draw would leave in the estimate.
        np.random.seed(SEED + run)

        # Two independent draws, each fit on exactly n_states states.
        train_set_1 = get_states_from_mdp(mdp=env, n_states=n_states)
        train_set_2 = get_states_from_mdp(mdp=env, n_states=n_states)
        scaler = StandardScaler().fit(np.concatenate([train_set_1, train_set_2], axis=0))

        # Same seed on both fits so KMeans init variance cannot masquerade as sampling noise.
        kmeans1 = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto").fit(
            scaler.transform(train_set_1)
        )
        kmeans2 = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto").fit(
            scaler.transform(train_set_2)
        )

        test_set = get_states_from_mdp(mdp=env, n_states=n_states)
        test_scaled = scaler.transform(test_set)
        aris.append(adjusted_rand_score(kmeans1.predict(test_scaled), kmeans2.predict(test_scaled)))
        distortions.append((mean_distortion(kmeans1, test_scaled) + mean_distortion(kmeans2, test_scaled)) / 2)

    mean_ari = float(np.mean(aris))
    std_ari = float(np.std(aris))
    mean_distortion_ = float(np.mean(distortions))
    std_distortion = float(np.std(distortions))
    print(
        f"\n[{request.node.name}] n_clusters={n_clusters} multiplier={multiplier} "
        f"mean_ari={mean_ari:.3f} std={std_ari:.3f} "
        f"mean_distortion={mean_distortion_:.3f} std_distortion={std_distortion:.3f} (n_runs={N_RUNS})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("env_name", ENV_NAMES)
@pytest.mark.parametrize("multiplier", MULTIPLIERS)
@pytest.mark.parametrize("n_clusters", N_CLUSTERS)
def test_kmeans_seed_ari(request, env_name, n_clusters, multiplier):
    env = Environment.make(env_name)
    n_states = multiplier * n_clusters

    aris = []
    distortions = []
    for run in range(N_RUNS):
        np.random.seed(SEED + run)

        train_set = get_states_from_mdp(mdp=env, n_states=n_states)
        scaler = StandardScaler().fit(train_set)
        train_scaled = scaler.transform(train_set)

        # Same scaled sample, two different KMeans seeds: isolates instability coming from
        # the existence of multiple equally-good partitions (init-dependent local optima)
        # from instability coming from sampling a finite n_states.
        kmeans1 = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED, n_init="auto").fit(train_scaled)
        kmeans2 = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED + 1, n_init="auto").fit(train_scaled)

        test_set = get_states_from_mdp(mdp=env, n_states=n_states)
        test_scaled = scaler.transform(test_set)
        aris.append(adjusted_rand_score(kmeans1.predict(test_scaled), kmeans2.predict(test_scaled)))
        distortions.append((mean_distortion(kmeans1, test_scaled) + mean_distortion(kmeans2, test_scaled)) / 2)

    mean_ari = float(np.mean(aris))
    std_ari = float(np.std(aris))
    mean_distortion_ = float(np.mean(distortions))
    std_distortion = float(np.std(distortions))
    print(
        f"\n[{request.node.name}] n_clusters={n_clusters} multiplier={multiplier} "
        f"mean_ari={mean_ari:.3f} std={std_ari:.3f} "
        f"mean_distortion={mean_distortion_:.3f} std_distortion={std_distortion:.3f} (n_runs={N_RUNS})"
    )
