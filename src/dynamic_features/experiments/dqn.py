from typing import List, Tuple, final
import torch
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from torch.optim.adam import Adam
from typing_extensions import override

from dynamic_features.configs import DqnAgentConfig, EnvConfig, PolicyConfig, TrainingConfig, WandbConfig
from dynamic_features.models.Linear import Linear

from .abstract_experiment import AbstractExp


@final
class DQN_exp(AbstractExp[DqnAgentConfig]):
    __slots__ = ["_initial_replay_size", "_max_replay_size"]

    def __init__(
        self,
        policy: PolicyConfig,
        agent: DqnAgentConfig,
        train: TrainingConfig,
        env: EnvConfig,
        wandb: WandbConfig,
        seed: int,
    ) -> None:
        _ = torch.manual_seed(seed)
        self._initial_replay_size: int = agent.initial_replay_size
        self._max_replay_size: int = agent.max_replay_size
        super().__init__(policy=policy, agent=agent, train=train, env=env, wandb=wandb, seed=seed)

    @override
    def _configure_policy_exploration_behavior(self, param: Parameter) -> None:
        self._policy.set_epsilon(param)

    @override
    def _create_policy(self) -> None:
        # in DQN we start with a random epsilon
        epsilon_random = Parameter(value=1.0)
        self._policy: EpsGreedy = EpsGreedy(epsilon=epsilon_random)

    @override
    def _create_agent(self) -> None:
        loss = self._agent_config.loss.value.function

        # Approximator
        input_shape = self._mdp.info.observation_space.shape
        approximator_params = dict(
            network=Linear,
            optimizer={"class": Adam, "params": {"lr": self._agent_config.learning_rate}},
            loss=loss,
            n_features=self._agent_config.n_features,
            input_shape=input_shape,
            output_shape=self._mdp.info.action_space.size,
            n_actions=self._mdp.info.action_space.n,
        )

        # Agent
        self._agent: DQN = DQN(
            self._mdp.info,
            self._policy,
            TorchApproximator,
            approximator_params=approximator_params,
            batch_size=self._agent_config.batch_size,
            initial_replay_size=self._initial_replay_size,
            max_replay_size=self._max_replay_size,
            target_update_frequency=self._agent_config.target_update_frequency,
        )

    @override
    def train(
        self,
        use_wandb: bool = False,
    ) -> Tuple[List[float], List[float]]:
        # first we need to fill the replay memory.
        self._core.learn(n_steps=self._initial_replay_size, n_steps_per_fit=self._initial_replay_size)
        return super().train(
            use_wandb=use_wandb,
        )
