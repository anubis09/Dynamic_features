from typing import Optional

import torch
import torch.nn.functional as F
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter
from torch.optim.adam import Adam

from dynamic_features.exp_class import AbstractExp
from dynamic_features.models.Linear import Linear


class DQN_exp(AbstractExp):
    __slots__ = ["initial_replay_size", "max_replay_size"]

    def __init__(self, game_name: str, seed: Optional[int]):
        torch.manual_seed(seed)
        super().__init__(game_name, seed)

    def create_policy(self, initial_exp_rate: float, final_exp_rate: float, final_exp_frame: int):
        # initializing epsilon for the epsilon greedy exploration
        self.exploration_policy_param = LinearParameter(
            value=initial_exp_rate,
            threshold_value=final_exp_rate,
            n=final_exp_frame,
        )
        # in DQN we start with a random epsilon
        epsilon_random = Parameter(value=1.0)
        self.policy = EpsGreedy(epsilon=epsilon_random)

    def create_agent(
        self,
        learning_rate: float,
        n_features: int,
        batch_size: int,
        initial_replay_size: int,
        max_replay_size: int,
        target_update_frequency: int,
        loss_name: str = "mse_loss",
    ):
        self.initial_replay_size = initial_replay_size
        self.max_replay_size = max_replay_size
        loss = F.smooth_l1_loss if loss_name == "smooth_l1_loss" else F.mse_loss

        # Approximator
        input_shape = self.mdp.info.observation_space.shape
        approximator_params = dict(
            network=Linear,
            optimizer={"class": Adam, "params": {"lr": learning_rate}},
            loss=loss,
            n_features=n_features,
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
            batch_size=batch_size,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            target_update_frequency=target_update_frequency,
        )

    def train(
        self,
        n_steps: int,
        test_samples: int,
        eval_policy_param: Parameter,
        train_frequency: int = 1,
        n_epochs: int = 100,
        use_wandb: bool = False,
    ) -> None:
        # first we need to fill the replay memory.
        self.core.learn(n_steps=self.initial_replay_size, n_steps_per_fit=self.initial_replay_size)
        return super().train(
            n_steps=n_steps,
            test_samples=test_samples,
            eval_policy_param=eval_policy_param,
            train_frequency=train_frequency,
            n_epochs=n_epochs,
            use_wandb=use_wandb,
        )
