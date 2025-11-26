from dataclasses import dataclass

from dynamic_features.configs.experiments.experiment_conf import (
    AgentConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainingConfig,
)


@dataclass
class DqnAgentConfig(AgentConfig):
    initial_replay_size: int
    max_replay_size: int
    target_update_frequency: int


@dataclass
class DqnConfig(ExperimentConfig):
    _target_: str = "dynamic_features.experiments.dqn.DQN_exp"
    policy: PolicyConfig = PolicyConfig(initial_exp_rate=1.0, final_exp_rate=0.01, final_exp_frame=1_000)
    agent: AgentConfig = DqnAgentConfig(
        initial_replay_size=250,
        max_replay_size=10_000,
        target_update_frequency=300,
        batch_size=32,
        n_features=100,
        loss_name="mse_loss",
        learning_rate=3e-4,
    )
    train: TrainingConfig = TrainingConfig(
        n_epochs=100, n_steps=20_000, test_samples=10_000, eval_policy_param=0, train_frequency=1
    )
