from dataclasses import dataclass, field

from dynamic_features.configs.experiments.experiment_conf import (
    AgentConfig,
    ExperimentConfig,
    Optimizers,
    PolicyConfig,
    TrainingConfig,
)


@dataclass
class RainbowAgentConfig(AgentConfig):
    initial_replay_size: int
    max_replay_size: int
    target_update_frequency: int
    n_atoms: int
    n_steps_returns: int
    alpha_coeff: float
    sigma_coeff: float
    beta_starting: float
    beta_final: float
    batch_size: int
    n_features: int
    optimizer: Optimizers


@dataclass
class RainbowConfig(ExperimentConfig):
    _target_: str = "dynamic_features.experiments.RainbowExperiment"
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(initial_exp_rate=1.0, final_exp_rate=0.01, final_exp_frame=30_000)
    )
    agent: RainbowAgentConfig = field(
        default_factory=lambda: RainbowAgentConfig(
            initial_replay_size=10_000,
            max_replay_size=50_000,
            target_update_frequency=500,
            n_atoms=51,
            n_steps_returns=3,
            alpha_coeff=0.6,
            sigma_coeff=0.5,
            beta_starting=0.4,
            beta_final=1.0,
            # beta frame is usually max_steps // train frequency
            batch_size=128,
            n_features=128,
            optimizer=Optimizers.ADAM,
            learning_rate=1e-4,
        )
    )
    train: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            n_epochs=100,
            # the number of steps used for training in each epoch is give by the env budget / n_epochs.
            test_episodes=15,  # episode tested after every epoch of training
            final_test_episodes=100,
            eval_policy_param=0,
            train_frequency=4,
            log_frequency=10,
        )
    )
