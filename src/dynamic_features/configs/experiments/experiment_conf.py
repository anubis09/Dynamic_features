from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import torch
import torch.nn.functional as F


class Losses(Enum):
    MSE = "mse_loss"
    SL1 = "smooth_l1_loss"
    HUBER = "huber_loss"
    CE = "cross_entropy"

    def get_function(self):
        """Resolve to actual PyTorch function"""
        return getattr(F, self.value)


class Optimizers(Enum):
    ADAM = "Adam"
    ADAMW = "AdamW"
    RMSPROP = "RMSprop"

    def get_class(self):
        """Resolve to actual optimizer class"""
        return getattr(torch.optim, self.value)


@dataclass
class PolicyConfig:
    initial_exp_rate: float
    final_exp_rate: float
    final_exp_frame: int


@dataclass
class AgentConfig:
    learning_rate: float


@dataclass
class TrainingConfig:
    n_epochs: int
    test_episodes: int
    final_test_episodes: int  # this is provided because we want better precision for the last evaluation.
    eval_policy_param: float
    train_frequency: int
    log_frequency: int  # frequency for training logging. if 5, it will log training details every 5 epoch.


TAgentConfig = TypeVar("TAgentConfig", bound=AgentConfig)


@dataclass
class ExperimentConfig:
    _target_: str
    policy: PolicyConfig
    agent: Any  # don't like this but hydra doesn't support generic and protocol
    train: TrainingConfig
