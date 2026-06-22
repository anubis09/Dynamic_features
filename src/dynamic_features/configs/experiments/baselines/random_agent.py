from dataclasses import dataclass, field

from dynamic_features.configs.experiments.experiment_conf import (
    AgentConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainingConfig,
)


@dataclass
class RandomAgentConfig(AgentConfig):
    # A random agent has nothing to learn, so it carries no extra hyperparameters.
    # ``learning_rate`` is inherited from AgentConfig but unused (kept at 0.0).
    pass


@dataclass
class RandomConfig(ExperimentConfig):
    _target_: str = "dynamic_features.experiments.RandomExperiment"
    # The policy schedule is unused by the random agent (it always acts uniformly at
    # random) but a valid PolicyConfig is kept so the config schema matches the others.
    policy: PolicyConfig = field(
        default_factory=lambda: PolicyConfig(initial_exp_rate=1.0, final_exp_rate=1.0, final_exp_frame=1)
    )
    agent: RandomAgentConfig = field(default_factory=lambda: RandomAgentConfig(learning_rate=0.0))
    train: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            n_epochs=100,
            test_episodes=15,
            final_test_episodes=100,
            eval_policy_param=0,
            train_frequency=1,
            log_frequency=10,
        )
    )
