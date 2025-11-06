from typing import final

import torch
import torch.nn.functional as F
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from torch.optim.adam import Adam
from typing_extensions import override

from dynamic_features.algorithms.abc_algorithm import AbstractAlg
from dynamic_features.configs.algorithm.algorithm_conf import PolicyConfig, TrainingConfig
from dynamic_features.configs.algorithm.dqn import DqnAgentConfig
from dynamic_features.configs.env_config import EnvConfig
from dynamic_features.configs.wandb_config import WandbConfig
from dynamic_features.models.Linear import Linear


@final
class DQN_exp(AbstractAlg[DqnAgentConfig]):
    __slots__ = ["initial_replay_size", "max_replay_size"]

    def __init__(
        self,
        policy: PolicyConfig,
        agent: DqnAgentConfig,
        train: TrainingConfig,
        env: EnvConfig,
        wandb: WandbConfig,
        seed: int,
    ):
        _ = torch.manual_seed(seed)
        self.initial_replay_size = agent.initial_replay_size
        self.max_replay_size = agent.max_replay_size
        super().__init__(policy=policy, agent=agent, train=train, env=env, wandb=wandb, seed=seed)

    @override
    def _create_policy(self):
        super()._create_policy()
        # in DQN we start with a random epsilon
        epsilon_random = Parameter(value=1.0)
        self.policy = EpsGreedy(epsilon=epsilon_random)

    @override
    def _create_agent(self):
        loss = F.smooth_l1_loss if self._agent_config.loss_name == "smooth_l1_loss" else F.mse_loss

        # Approximator
        input_shape = self.mdp.info.observation_space.shape
        approximator_params = dict(
            network=Linear,
            optimizer={"class": Adam, "params": {"lr": self._agent_config.learning_rate}},
            loss=loss,
            n_features=self._agent_config.n_features,
            input_shape=input_shape,
            output_shape=self.mdp.info.action_space.size,
            n_actions=self.mdp.info.action_space.n,
        )

        # Agent
        self.agent = DQN(
            self.mdp.info,
            self.policy,
            TorchApproximator,
            approximator_params=approximator_params,
            batch_size=self._agent_config.batch_size,
            initial_replay_size=self.initial_replay_size,
            max_replay_size=self.max_replay_size,
            target_update_frequency=self._agent_config.target_update_frequency,
        )

    @override
    def train(
        self,
        use_wandb: bool = False,
    ) -> None:
        # first we need to fill the replay memory.
        self.core.learn(n_steps=self.initial_replay_size, n_steps_per_fit=self.initial_replay_size)
        return super().train(
            use_wandb=use_wandb,
        )
