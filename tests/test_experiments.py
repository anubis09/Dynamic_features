from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from dynamic_features import CONFIGS_PATH
from dynamic_features.configs import register_configs
from dynamic_features.launcher import experiment

register_configs()


@pytest.mark.slow
@pytest.mark.parametrize("environment", ["Acrobot", "MountainCar", "LunarLander", "CartPole", "PuddleWorld"])
@pytest.mark.parametrize("algorithm", ["dqn", "rainbow", "sarsa_lambda_rec", "sarsa_lambda_dt"])
def test_experiments(environment, algorithm):
    """Test the experiment function directly using Hydra context."""

    overrides = [
        f"environment={environment}",
        "environment.budget=20",
        f"algorithm={algorithm}",
        "algorithm.train.n_epochs=5",
        "algorithm.train.test_episodes=3",
        "algorithm.train.final_test_episodes=5",
        "wandb_log=False",
    ]

    if algorithm in ("dqn", "rainbow"):
        overrides.append("algorithm.agent.initial_replay_size=10")
        overrides.append("algorithm.agent.max_replay_size=20")
        overrides.append("algorithm.agent.target_update_frequency=5")
    elif algorithm == "sarsa_lambda_dt":
        overrides.append("algorithm.agent.n_tilings=2")
        overrides.append("algorithm.agent.n_clusters=2")

    with initialize_config_dir(version_base=None, config_dir=str(Path(CONFIGS_PATH))):
        cfg = compose(config_name="config", overrides=overrides)
        experiment(cfg)
