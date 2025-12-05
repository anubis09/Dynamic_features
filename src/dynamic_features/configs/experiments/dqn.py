from dataclasses import dataclass, field

from .experiment_conf import AgentConfig, ExperimentConfig, Losses, Optimizers, PolicyConfig, TrainingConfig


@dataclass
class DqnAgentConfig(AgentConfig):
    initial_replay_size: int
    max_replay_size: int
    target_update_frequency: int


@dataclass
class DqnConfig(ExperimentConfig):
    _target_: str = "dynamic_features.experiments.dqn.DqnExperiment"
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(initial_exp_rate=1.0, final_exp_rate=0.01, final_exp_frame=10_000)
    )
    agent: DqnAgentConfig = field(
        default_factory=lambda: DqnAgentConfig(
            initial_replay_size=1_000,
            max_replay_size=50_000,
            target_update_frequency=500,
            batch_size=32,
            n_features=100,
            loss=Losses.MSE,
            optimizer=Optimizers.RMSPROP,
            learning_rate=2.5e-4,
        )
    )
    train: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            n_epochs=100, n_steps=20_000, test_samples=10_000, eval_policy_param=0, train_frequency=1
        )
    )
