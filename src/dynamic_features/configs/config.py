from dataclasses import dataclass, field
from typing import Any, List

from omegaconf import MISSING

from dynamic_features.configs.experiments.experiment_conf import ExperimentConfig
from dynamic_features.configs.wandb_config import WandbConfig, WandbMode

from .env_config import EnvConfig

defaults = [
    {"algorithm": MISSING},
    # Forces the user to specify an environment config; the program will fail if omitted.
    {"environment": MISSING},
    # overrides the default launcher with joblib
    {"override hydra/launcher": "joblib"},
    # The "_self_" keyword indicates that the configuration defined in the class below (Config)
    # takes precedence over the defaults loaded above regarding the root level keys.
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
    wandb: WandbConfig = field(
        default_factory=lambda: WandbConfig(
            project="Dynamic_Features",
            entity="Dynamic_Tiles_Thesis",
            job_type="",
            mode=WandbMode.OFFLINE,
            group="${environment.name}",
            name="seed_${seed}",
            config={"sweep_id": ""},
        )
    )
