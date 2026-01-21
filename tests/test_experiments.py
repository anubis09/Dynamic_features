from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from dynamic_features import CONFIGS_PATH
from dynamic_features.configs import register_configs
from dynamic_features.launcher import experiment

register_configs()


@pytest.mark.parametrize("environment", ["Acrobot", "MountainCar", "LunarLander", "CartPole"])
@pytest.mark.parametrize("algorithm", ["dqn"])
def test_experiments(environment, algorithm):
    """Test the experiment function directly using Hydra context."""

    with initialize_config_dir(version_base=None, config_dir=str(Path(CONFIGS_PATH))):
        cfg = compose(
            config_name="config",
            overrides=[
                f"environment={environment}",
                "environment.budget=20",
                f"algorithm={algorithm}",
                "algorithm.train.n_epochs=1",
                "algorithm.train.test_episodes=2",
                "algorithm.train.final_test_episodes=10",
                "wandb_log=False",
            ],
        )

        experiment(cfg)
