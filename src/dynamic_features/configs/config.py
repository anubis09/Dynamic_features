from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING

from dynamic_features.configs.experiments.experiment_conf import ExperimentConfig
from dynamic_features.configs.wandb_config import WandbConfig

from .env_config import EnvConfig

defaults = [
    {"algorithm": "dqn"},
    {"environment": MISSING},
    {"override hydra/launcher": "joblib"},  # overriding the default launcher with joblib
    "_self_",
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Hydra will populate this field based on the defaults list
    algorithm: ExperimentConfig = MISSING
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
