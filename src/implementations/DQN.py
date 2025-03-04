import os
import sys

import torch.nn.functional as F
import torch.optim as optim
import yaml
from experiment_launcher import run_experiment, single_experiment
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import LinearParameter, Parameter

sys.path.append("../src")

from exp_class import AbstractExp
from models.DQN_networks import Linear


class DQN_exp(AbstractExp):
    __slots__ = ["initial_replay_size", "max_replay_size"] + AbstractExp.__slots__

    def __init__(self, game_name: str):
        super().__init__(game_name)

    def create_policy(
        self, initial_exp_rate: float, final_exp_rate: float, final_exp_frame: int
    ):
        # initializing epsilon for the epsilon greedy exploration
        self.epsilon = LinearParameter(
            value=initial_exp_rate,
            threshold_value=final_exp_rate,
            n=final_exp_frame,
        )
        # in DQN we start with a random epsilon
        epsilon_random = Parameter(value=1.0)
        self.pi = EpsGreedy(epsilon=epsilon_random)

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
            self.pi,
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
        train_frequency: int = 1,
        n_epochs: int = 100,
    ) -> None:
        # first we need to fill the replay memory.
        self.core.learn(
            n_steps=self.initial_replay_size, n_steps_per_fit=self.initial_replay_size
        )
        return super().train(n_steps, test_samples, train_frequency, n_epochs)


@single_experiment
def experiment(
    game_name: str = "MountainCar-v0",
    seed: int = 0,
    result_dir: str = "logs",
    n_epochs: int = 1,
    wandb_log: bool = False,
    wandb_job_type: str = "",
    **kwargs,
):
    dqn_exp_class = DQN_exp(
        game_name=game_name,
    )
    starting_folder = "src" if "src" in os.listdir() else "."
    with open(
        os.path.join(starting_folder, "configs", "algorithms_conf.yaml"), "r"
    ) as f:
        params = yaml.load(f, yaml.Loader)["DQN"]

    dqn_exp_class.create_policy(**params["policy"])

    dqn_exp_class.create_agent(**params["agent"])

    dqn_exp_class.create_core()

    if wandb_log:
        dqn_exp_class.set_wandb()
        dqn_exp_class.wandb_obj["name"] = f"seed_{seed}"
        dqn_exp_class.wandb_obj["job_type"] = wandb_job_type
        dqn_exp_class.wandb_obj["group"] = game_name
        dqn_exp_class.wandb_obj["config"] = params
        dqn_exp_class.wandb_obj["config"]["n_epochs"] = n_epochs

    dqn_exp_class.train(n_epochs=n_epochs, **params["train"])


if __name__ == "__main__":
    run_experiment(experiment)
