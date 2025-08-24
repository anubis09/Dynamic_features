from typing import Optional

import torch.nn.functional as F
import torch.optim as optim
import yaml
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from dynamic_features import ALGORITHMS_CONFIG_PATH
from dynamic_features.exp_class import AbstractExp
from dynamic_features.models.Linear import Linear


class DQN_exp(AbstractExp):
    __slots__ = ["initial_replay_size", "max_replay_size"]

    def __init__(self, game_name: str, seed: Optional[int]):
        super().__init__(game_name, seed)

    def create_policy(self, initial_exp_rate: float, final_exp_rate: float, final_exp_frame: int):
        # initializing epsilon for the epsilon greedy exploration
        self.epsilon = LinearParameter(
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
            optimizer={"class": optim.Adam, "params": {"lr": learning_rate}},  # type: ignore
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
        eval_exploration_param: Parameter,
        train_frequency: int = 1,
        n_epochs: int = 100,
        use_wandb: bool = False,
    ) -> None:
        # first we need to fill the replay memory.
        self.core.learn(n_steps=self.initial_replay_size, n_steps_per_fit=self.initial_replay_size)
        return super().train(
            n_steps=n_steps,
            test_samples=test_samples,
            eval_exploration_param=eval_exploration_param,
            train_frequency=train_frequency,
            n_epochs=n_epochs,
            use_wandb=use_wandb,
        )


def experiment(
    game_name: str = "MountainCar-v0",
    seed: int = 0,
    n_epochs: int = 2,
    wandb_log: bool = False,
    wandb_job_type: str = "",
    **kwargs,
):
    dqn_exp_class = DQN_exp(game_name=game_name, seed=42)
    with open(ALGORITHMS_CONFIG_PATH, "r") as f:
        params = yaml.load(f, yaml.Loader)["DQN"]

    dqn_exp_class.create_policy(**params["policy"])

    dqn_exp_class.create_agent(**params["agent"])

    dqn_exp_class.create_core()

    if wandb_log:
        dqn_exp_class._wandb_conf["name"] = f"seed_{seed}"
        dqn_exp_class._wandb_conf["job_type"] = wandb_job_type
        dqn_exp_class._wandb_conf["group"] = game_name
        dqn_exp_class._wandb_conf["config"] = params
        dqn_exp_class._wandb_conf["config"]["n_epochs"] = n_epochs

    dqn_exp_class.train(n_epochs=n_epochs, **params["train"])


if __name__ == "__main__":
    experiment()
