from hydra.core.config_store import ConfigStore

from .config import Config
from .env_config import AcrobotEnvConfig, EnvConfig, EnvCore, LunarLanderEnvConfig, MountainCarEnvConfig, RainbowParams
from .experiments import (
    AgentConfig,
    DqnAgentConfig,
    DqnConfig,
    ExperimentConfig,
    Loss,
    Losses,
    PolicyConfig,
    TAgentConfig,
    TrainingConfig,
)
from .wandb_config import WandbConfig


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


__all__ = [
    "register_configs",
    "Config",
    "EnvConfig",
    "EnvCore",
    "AcrobotEnvConfig",
    "LunarLanderEnvConfig",
    "MountainCarEnvConfig",
    "RainbowParams",
    "AgentConfig",
    "DqnAgentConfig",
    "DqnConfig",
    "ExperimentConfig",
    "Loss",
    "Losses",
    "PolicyConfig",
    "TAgentConfig",
    "TrainingConfig",
    "WandbConfig",
]
