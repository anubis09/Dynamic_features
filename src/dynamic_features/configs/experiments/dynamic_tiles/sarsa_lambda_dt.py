from dataclasses import dataclass, field

from dynamic_features.configs.experiments.experiment_conf import (
    AgentConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainingConfig,
)


@dataclass
class SarsaLambdaDTAgentConfig(AgentConfig):
    n_tilings: int
    n_clusters: int
    lambda_coeff: float
    clustering_training_states_multiplier: int


@dataclass
class SarsaLambdaDTConfig(ExperimentConfig):
    _target_: str = "dynamic_features.experiments.SarsaLambdaDTExperiment"
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(initial_exp_rate=1.0, final_exp_rate=0.01, final_exp_frame=10_000)
    )
    agent: SarsaLambdaDTAgentConfig = field(
        default_factory=lambda: SarsaLambdaDTAgentConfig(
            n_tilings=10,
            n_clusters=20,
            lambda_coeff=0.99,
            # Scales the number of states sampled to train the clustering features.
            # Total training states = n_clusters * state_dim * clustering_training_states_multiplier.
            clustering_training_states_multiplier=150,
            # in this case we mean the learning_rate pre scaling.
            learning_rate=0.1,
        )
    )
    train: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            n_epochs=100,
            # the number of steps used for training in each epoch is give by the env budget / n_epochs.
            test_episodes=15,  # episode tested after every epoch of training
            final_test_episodes=100,
            eval_policy_param=0,
            train_frequency=1,
            log_frequency=10,
        )
    )
