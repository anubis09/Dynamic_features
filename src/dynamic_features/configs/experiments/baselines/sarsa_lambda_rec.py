from dataclasses import dataclass, field

from dynamic_features.configs.experiments.experiment_conf import (
    AgentConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainingConfig,
)


@dataclass
class SarsaLambdaRecAgentConfig(AgentConfig):
    n_tilings: int
    n_tiles: int
    lambda_coeff: float


@dataclass
class SarsaLambdaRecConfig(ExperimentConfig):
    _target_: str = "dynamic_features.experiments.SarsaLambdaRecExperiment"
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(initial_exp_rate=1.0, final_exp_rate=0.01, final_exp_frame=10_000)
    )
    agent: SarsaLambdaRecAgentConfig = field(
        default_factory=lambda: SarsaLambdaRecAgentConfig(
            n_tilings=32,
            # n_tiles = 5 means that in each dimension of the mdp it will have 5.
            # so if mdp state dimension is 2 we will have a 5x5 grid.
            n_tiles=5,
            lambda_coeff=0.99,
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
