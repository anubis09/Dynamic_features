from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

import torch
import torch.nn.functional as F


@dataclass
class Loss:
    name: str
    function: Callable


class Losses(Enum):
    MSE = Loss(name="mse_loss", function=F.mse_loss)
    SL1 = Loss(name="smooth_l1_loss", function=F.smooth_l1_loss)
    HUBER = Loss(name="huber_loss", function=F.huber_loss)


@dataclass
class Optimizer:
    name: str
    cls: Callable


class Optimizers(Enum):
    ADAM = Optimizer(name="adam", cls=torch.optim.Adam)
    ADAMW = Optimizer(name="adamw", cls=torch.optim.AdamW)
    RMSPROP = Optimizer(name="rmsprop", cls=torch.optim.RMSprop)


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
    agent: AgentConfig
    train: TrainingConfig
