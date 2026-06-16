from hydra.core.config_store import ConfigStore

from .config import Config
from .env_config import (
    AcrobotEnvConfig,
    CartPoleEnvConfig,
    EnvConfig,
    EnvCore,
    LunarLanderEnvConfig,
    MountainCarEnvConfig,
    PuddleWorldEnvConfig,
    RainbowParams,
)
from .experiments.baselines.dqn import DqnAgentConfig, DqnConfig
from .experiments.baselines.rainbow import RainbowAgentConfig, RainbowConfig
from .experiments.baselines.sarsa_lambda_rec import SarsaLambdaRecAgentConfig, SarsaLambdaRecConfig
from .experiments.dynamic_tiles.sarsa_lambda_dt import (
    ApproximatorConfig,
    RegularizationPrior,
    RetrainTrigger,
    SarsaLambdaDTAgentConfig,
    SarsaLambdaDTConfig,
)
from .experiments.experiment_conf import (
    AgentConfig,
    ExperimentConfig,
    Losses,
    Optimizers,
    PolicyConfig,
    TAgentConfig,
    TrainingConfig,
)
from .wandb_config import WandbConfig


# config store are used to load predefined configurations at runtime.
# Usually Config is preloaded by Hydra main, but config expects
# a specified environment config and algorithm config.
# Config store allows us to register these configurations.
def register_configs():
    cs = ConfigStore.instance()

    cs.store(name="MountainCar", group="environment", node=MountainCarEnvConfig)
    cs.store(name="LunarLander", group="environment", node=LunarLanderEnvConfig)
    cs.store(name="Acrobot", group="environment", node=AcrobotEnvConfig)
    cs.store(name="CartPole", group="environment", node=CartPoleEnvConfig)
    cs.store(name="PuddleWorld", group="environment", node=PuddleWorldEnvConfig)

    cs.store(name="dqn", group="algorithm", node=DqnConfig)
    cs.store(name="rainbow", group="algorithm", node=RainbowConfig)
    cs.store(name="sarsa_lambda_rec", group="algorithm", node=SarsaLambdaRecConfig)
    cs.store(name="sarsa_lambda_dt", group="algorithm", node=SarsaLambdaDTConfig)

    cs.store(name="config", node=Config)


__all__ = [
    "register_configs",
    "Config",
    "EnvConfig",
    "EnvCore",
    "AcrobotEnvConfig",
    "LunarLanderEnvConfig",
    "MountainCarEnvConfig",
    "CartPoleEnvConfig",
    "PuddleWorldEnvConfig",
    "RainbowParams",
    "AgentConfig",
    "DqnAgentConfig",
    "DqnConfig",
    "SarsaLambdaRecAgentConfig",
    "SarsaLambdaRecConfig",
    "SarsaLambdaDTAgentConfig",
    "SarsaLambdaDTConfig",
    "ApproximatorConfig",
    "RainbowConfig",
    "RainbowAgentConfig",
    "ExperimentConfig",
    "Losses",
    "Optimizers",
    "RetrainTrigger",
    "RegularizationPrior",
    "PolicyConfig",
    "TAgentConfig",
    "TrainingConfig",
    "WandbConfig",
]
