from dataclasses import dataclass
from typing import TypeVar


@dataclass
class PolicyConfig:
    initial_exp_rate: float
    final_exp_rate: float
    final_exp_frame: int


@dataclass
class AgentConfig:
    batch_size: int
    n_features: int
    loss_name: str
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
