from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar

import torch
import torch.nn.functional as F


class Losses(Enum):
    MSE = "mse_loss"
    SL1 = "smooth_l1_loss"
    HUBER = "huber_loss"

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
    batch_size: int
    n_features: int
    loss: Losses
    optimizer: Optimizers
    learning_rate: float


@dataclass
class TrainingConfig:
    n_epochs: int
    n_steps: int
    test_samples: int
    eval_policy_param: float
    train_frequency: int


TAgentConfig = TypeVar("TAgentConfig", bound=AgentConfig)


@dataclass
class ExperimentConfig:
    _target_: str
    policy: PolicyConfig
    agent: Any  # don't like this but hydra doesn't support generic and protocol
    train: TrainingConfig
