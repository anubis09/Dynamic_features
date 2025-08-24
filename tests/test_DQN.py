import yaml

from dynamic_features import GAME_ENV_CONFIG_PATH
from dynamic_features.implementations.DQN import DQN_exp

with open(GAME_ENV_CONFIG_PATH, "r") as f:
    CONFIG = yaml.load(f, yaml.Loader)


def test_class():
    games = ["MountainCar-v0", "Acrobot-v1", "LunarLander-v2"]
    for name in games:
        assert DQN_exp(name, seed=42).gamma == CONFIG[name.split("-")[0]]["gamma"]
