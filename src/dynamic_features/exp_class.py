import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import yaml
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

import wandb
from dynamic_features import GAME_ENV_CONFIG_PATH, WANDB_CONFIG_PATH


class AbstractExp(ABC):
    __slots__ = [
        "mdp",
        "gamma",
        "exploration_policy_param",
        "policy",
        "agent",
        "core",
        "seed",
        "_wandb_conf",
        "_logger",
    ]

    def __init__(self, mdp_name: str, seed: Optional[int]) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self.seed = seed
        np.random.seed(seed)
        self._setup_wandb()
        self._setup_mdp(mdp_name)

    def _setup_wandb(self) -> None:
        """Initialize wandb configuration from config file."""
        try:
            with open(WANDB_CONFIG_PATH, "r") as f:
                self._wandb_conf = yaml.load(f, yaml.SafeLoader)
        except Exception as e:
            self._logger.warning(f"Failed to load wandb config: {e}")
            self._wandb_conf = {}

    def update_wandb_conf(self, **kwargs) -> None:
        """Update configs for wandb"""
        if kwargs:
            self._logger.debug(f"Updating wandb config with: {list(kwargs.keys())}")
        self._wandb_conf.update(kwargs)

    def _setup_mdp(self, mdp_name: str) -> None:
        """Initialize the MDP environment from configuration."""
        if "-" not in mdp_name:
            raise ConfigurationError(f"The following mdp ({mdp_name}) is not supported.")

        game_name = mdp_name.split("-")[0]
        with open(GAME_ENV_CONFIG_PATH, "r") as f:
            game_config = yaml.load(f, yaml.SafeLoader).get(game_name, None)
        if game_config is None:
            raise ConfigurationError(f"Game '{game_name}' not found in config.")
        self.gamma = game_config.get("gamma")
        horizon = game_config.get("horizon", None)
        try:
            self.mdp = Gym(mdp_name, horizon=horizon, gamma=self.gamma)  # Use full mdp_name
            self.mdp.seed(self.seed)
            self._logger.info(f"Initialized MDP: {mdp_name} with gamma={self.gamma}")
        except Exception as e:
            raise ConfigurationError(f"Failed to create environment '{mdp_name}': {e}")

    @abstractmethod
    def create_policy(self, initial_exp_rate: float, final_exp_rate: float, final_exp_frame: int) -> None:
        """Create the policy for the agent."""
        pass

    @abstractmethod
    def create_agent(
        self,
        learning_rate: float,
        n_features: int,
        batch_size: int,
        initial_replay_size: int,
        max_replay_size: int,
        target_update_frequency: int,
        loss_name: str,
    ) -> None:
        """Create the RL agent."""
        pass

    def create_core(self):
        """Create the MushroomRL Core object."""
        if not hasattr(self, "agent"):
            raise ExperimentError("Agent must be created before core. Call create_agent() first.")
        self.core = Core(agent=self.agent, mdp=self.mdp)
        self._logger.info("Core initialized successfully")

    def _configure_policy_exploration_behavior(self, param: Parameter) -> None:
        """Configure policy behavior based on parameter value."""
        if not hasattr(self, "policy"):
            raise ExperimentError("Policy not initialized")

        if hasattr(self.policy, "set_epsilon"):
            self.policy.set_epsilon(param)
        elif hasattr(self.policy, "set_beta"):
            self.policy.set_beta(param)
        else:
            raise ExperimentError("Policy has neither epsilon nor beta parameter")

    def train(
        self,
        n_steps: int,
        test_samples: int,
        eval_policy_param: Parameter,
        train_frequency: int = 1,
        n_epochs: int = 100,
        use_wandb: bool = False,
    ) -> None:
        """
        Train the agent with periodic evaluation.
        """
        if not hasattr(self, "core"):
            raise ExperimentError("core is not initalized.")

        if not hasattr(self, "policy"):
            raise ExperimentError("Policy not initialized. Call create_policy() first.")

        # Initialize wandb if requested and configured
        wandb_active = use_wandb and self._wandb_conf
        if wandb_active:
            try:
                wandb.init(**self._wandb_conf)
                self._logger.info("Wandb initialized successfully")
            except Exception as e:
                self._logger.warning(f"Failed to initialize wandb: {e}")
                wandb_active = False

        J_list = []
        dJ_list = []

        initial_J, initial_dJ = self._evaluate(test_samples, eval_policy_param)
        J_list.append(initial_J)
        dJ_list.append(initial_dJ)

        if wandb_active:
            self._log_to_wandb(J=initial_J, dJ=initial_dJ, step=0)

        self._logger.info(f"Starting training: {n_epochs} epochs, {n_steps} steps per epoch")

        for epoch in range(1, n_epochs + 1):
            # Learning step
            self._configure_policy_exploration_behavior(self.exploration_policy_param)
            self.core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)

            # Evaluation step
            J_curr, dJ_curr = self._evaluate(test_samples, eval_policy_param)
            J_list.append(J_curr)
            dJ_list.append(dJ_curr)
            if wandb_active:
                self._log_to_wandb(J=J_curr, dJ=dJ_curr, step=epoch)

        if wandb_active:
            wandb.finish()

        self._logger.info("Training completed")

    def _evaluate(self, test_samples: int, eval_exploration_param: Parameter) -> Tuple[np.floating, np.floating]:
        """Evaluate the current policy and return average returns."""

        # Set deterministic policy for evaluation
        self._configure_policy_exploration_behavior(eval_exploration_param)

        dataset = self.core.evaluate(n_steps=test_samples, render=False)

        J = np.mean(compute_J(dataset, gamma=1.0))  # Undiscounted return
        dJ = np.mean(compute_J(dataset, gamma=self.gamma))  # Discounted return

        return J, dJ

    def _log_to_wandb(self, J: np.floating, dJ: np.floating, step: int) -> None:
        """Log metrics to wandb."""
        group_name = self._wandb_conf.get("group")
        wandb.log(
            {
                f"{group_name}/average_return": J,
                f"{group_name}/discounted_average_return": dJ,
            },
            step=step,
            commit=True,
        )

    def save_agent(self, path: str) -> None:
        if not hasattr(self, "agent"):
            raise ExperimentError("agent not initialized")
        if ".msh" not in path:
            path += ".msh"

        self.agent.save(path)


class ExperimentError(Exception):
    """Custom exception for experiment-related errors"""

    pass


class ConfigurationError(ExperimentError):
    """Exception raised for configuration-related errors"""

    pass
