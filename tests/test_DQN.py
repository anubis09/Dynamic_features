import os
import sys

import yaml

sys.path.append("src")

from src.implementations.DQN import DQN_exp
from src.launcher import launcher

with open(os.path.join("src", "configs", "game_env_conf.yaml"), "r") as f:
    CONFIG = yaml.load(f, yaml.Loader)


def test_class():
    games = ["MountainCar-v0", "Acrobot-v1", "LunarLander-v2"]
    for name in games:
        assert DQN_exp(name).gamma == CONFIG[name.split("-")[0]]["gamma"]


def test_train():
    launcher()
