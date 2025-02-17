
import yaml

from src.exp_class import exp

with open("src/config.yaml","r",) as f:
    CONFIG = yaml.load(f, yaml.Loader)

def test_class():
    games = ["MountainCar-v0","Acrobot-v1", "LunarLander-v2"]
    for name in games:
        assert exp(name).gamma == CONFIG[name.split('-')[0]]["gamma"]
