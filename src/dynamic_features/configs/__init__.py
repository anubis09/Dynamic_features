from hydra.core.config_store import ConfigStore

from dynamic_features.configs.config import Config
from dynamic_features.configs.env_config import AcrobotEnvConfig, LunarLanderEnvConfig, MountainCarEnvConfig
from dynamic_features.configs.experiments.dqn import DqnConfig


# config store are used to load predefined configurations at runtime.
# Usually Config is preloaded by Hydra main, but config expects
# a specified environment config and algorithm config. Config store allows us to register these configurations.
def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="MountainCar", group="environment", node=MountainCarEnvConfig)
    cs.store(name="LunarLander", group="environment", node=LunarLanderEnvConfig)
    cs.store(name="Acrobot", group="environment", node=AcrobotEnvConfig)

    cs.store(name="dqn", group="algorithm", node=DqnConfig)

    cs.store(name="config", node=Config)


__all__ = ["register_configs"]
