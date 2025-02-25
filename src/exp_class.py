import os
from abc import ABC, abstractmethod

import numpy as np
import wandb
import yaml
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter


class AbstractExp(ABC):
    __slots__ = ["mdp", "gamma", "epsilon", "pi", "agent", "core", "wandb_obj"]

    def __init__(self, mdp_name: str) -> None:
        np.random.seed()
        self.__get_mdp(mdp_name)

    def set_wandb(self) -> None:
        with open(os.path.join("src", "configs", "wandb_conf.yaml"), "r") as f:
            self.wandb_obj = yaml.load(f, yaml.Loader)

    def __get_mdp(self, mdp_name: str) -> None:
        if "-" in mdp_name:
            game_name = mdp_name.split("-")[0]
            with open(os.path.join("src", "configs", "game_env_conf.yaml"), "r") as f:
                game_config = yaml.load(f, yaml.Loader)[game_name]
            self.gamma = game_config["gamma"]
            # TODO: Once I will have unified the horizon everywhere, I will have
            # a try catch here for the environments.
            if "horizon" not in game_config.keys():
                self.mdp = Gym(game_name, gamma=self.gamma)
            else:
                self.mdp = Gym(
                    game_name, horizon=game_config["horizon"], gamma=self.gamma
                )
        else:
            raise Exception(f"The following mdp ({mdp_name}) is not supported.")

    @abstractmethod
    def create_policy(self):
        pass

    @abstractmethod
    def create_agent(self):
        pass

    def create_core(self):
        self.core = Core(self.agent, self.mdp)

    def __set_pi_parameter(self, param: Parameter) -> None:
        # I might check with hasattr
        try:
            self.pi.set_epsilon(param)
        except:  # TODO change this to the right exception when the time comes.
            self.pi.set_beta(param)

    def train(
        self,
        n_steps: int,
        test_samples: int,
        train_frequency: int = 1,
        n_epochs: int = 100,
    ) -> None:
        if not hasattr(self, "core"):
            raise Exception("core is not initalized.")

        if not hasattr(self, "wandb_obj"):
            wandb_log = False
        else:
            wandb_log = True
            wandb.init(**self.wandb_obj)

        J_list = []
        dJ_list = []

        # deterministic epsilon for evaluation step.
        epsilon_test = Parameter(value=0)
        gamma_eval = 1.0
        self.pi.set_epsilon(epsilon_test)
        dataset = self.core.evaluate(n_steps=test_samples, render=False)
        # This is the value of the reward function at the beginning, with epsilon=0
        J_list.append(np.mean(compute_J(dataset, gamma_eval)))
        dJ_list.append(np.mean(compute_J(dataset, self.gamma)))

        if wandb_log:
            wandb.log(
                {
                    f"{self.wandb_obj['group']}/average_return": J_list[-1],
                    f"{self.wandb_obj['group']}/discounted_average_return": dJ_list[-1],
                },
                step=0,
                commit=True,
            )

        for e in range(n_epochs):
            # Learning step
            self.pi.set_epsilon(self.epsilon)
            self.core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)
            # Evaluation step
            self.pi.set_epsilon(epsilon_test)
            dataset = self.core.evaluate(n_steps=test_samples, render=False)
            # this is the cumulative reward computed for the given policy.
            J_list.append(np.mean(compute_J(dataset, gamma_eval)))
            dJ_list.append(np.mean(compute_J(dataset, self.gamma)))
            if wandb_log:
                wandb.log(
                    {
                        f"{self.wandb_obj['group']}/average_return": J_list[-1],
                        f"{self.wandb_obj['group']}/discounted_average_return": dJ_list[
                            -1
                        ],
                    },
                    step=e + 1,
                    commit=True,
                )

        if wandb_log:
            wandb.finish()

    def save_agent(self, path: str) -> None:
        if ".msh" not in path:
            path += ".msh"
        if not hasattr(self, "agent"):
            raise Exception("agent not initialized")
        else:
            self.agent.save(path)
