from dataclasses import dataclass, field
from typing import List

from omegaconf import MISSING

from dynamic_features.configs.algorithm.algorithm_conf import AlgorithmConfig
from dynamic_features.configs.wandb_config import WandbConfig

from .env_config import EnvConfig

defaults = [
    # Load the config "dqn" from the config group "algorithm"
    {"algorithm": "dqn"},
    {"environment": MISSING},
    "_self_",
]


@dataclass
class Config:
    defaults: List = field(default_factory=lambda: defaults)

    # Hydra will populate this field based on the defaults list
    algorithm: AlgorithmConfig = MISSING
    environment: EnvConfig = MISSING

    seed: int = 0
    wandb_log: bool = False
    wandb: WandbConfig = WandbConfig(
        project="Dynamic_Features",
        entity="Dynamic_Tiles_Thesis",
        job_type="",
        group="${environment.name}",
        name="seed_${seed}",
        config={},
    )
