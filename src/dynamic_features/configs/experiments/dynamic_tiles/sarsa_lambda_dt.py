from dataclasses import dataclass, field

from dynamic_features.configs.experiments.experiment_conf import (
    AgentConfig,
    ExperimentConfig,
    Losses,
    Optimizers,
    PolicyConfig,
    TrainingConfig,
)


@dataclass
class ApproximatorConfig:
    n_hidden_layers: int
    hidden_dim: int
    dropout: float
    batch_size: int
    lr: float
    eval_size: float
    loss: Losses
    optimizer: Optimizers
    n_epochs: int
    min_delta: float
    patience: int


@dataclass
class SarsaLambdaDTAgentConfig(AgentConfig):
    lambda_coeff: float
    ### features tilings
    n_tilings: int
    n_clusters: int
    clustering_training_states_multiplier: int
    approximator: ApproximatorConfig
    retrain_features: bool
    retrain_interval: int
    retrain_regularizer: float
    weight_transfer_states_multiplier: int
    splithalf_stability_check: bool


@dataclass
class SarsaLambdaDTConfig(ExperimentConfig):
    _target_: str = "dynamic_features.experiments.SarsaLambdaDTExperiment"
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(initial_exp_rate=1.0, final_exp_rate=0.01, final_exp_frame=30_000)
    )
    agent: SarsaLambdaDTAgentConfig = field(
        default_factory=lambda: SarsaLambdaDTAgentConfig(
            n_tilings=24,
            n_clusters=48,
            lambda_coeff=0.9,
            # Scales the number of states sampled to train the clustering features.
            # Total training states = n_clusters * clustering_training_states_multiplier.
            clustering_training_states_multiplier=200,
            # in this case we mean the learning_rate pre scaling.
            learning_rate=0.1,
            approximator=ApproximatorConfig(
                n_hidden_layers=2,
                hidden_dim=32,
                dropout=0.0,
                ### approximator train
                batch_size=512,
                lr=1e-3,
                eval_size=0.1,
                loss=Losses.CE,
                optimizer=Optimizers.ADAM,
                n_epochs=7,
                min_delta=0.005,
                patience=5,
            ),
            retrain_features=True,
            retrain_interval=10,
            retrain_regularizer=1e-3,
            weight_transfer_states_multiplier=100,
            # Diagnostic only: at each retrain, refit two independent KMeans on independent
            # on-policy draws of the configured training size and compare their partitions
            # (split-half ARI). Isolates sampling noise from distribution drift; adds a second
            # round of clustering per retrain, so keep off for normal runs.
            splithalf_stability_check=False,
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
