import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Generic, Optional, Tuple

import numpy as np
import wandb
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import Parameter

from dynamic_features.configs.algorithm.algorithm_conf import PolicyConfig, TAgentConfig, TrainingConfig
from dynamic_features.configs.env_config import EnvConfig
from dynamic_features.configs.wandb_config import WandbConfig


# Putting here Generic[TAgentConfig] allows the class to be generic over the type of agent configuration
# A subclass of AbstractAlgorith can use a specific subclass of AgentConfig without getting lsp errors.
class AbstractAlg(ABC, Generic[TAgentConfig]):
    __slots__ = [
        "mdp",
        "_mdp_config",
        "gamma",
        "exploration_policy_param",
        "policy",
        "_policy_config",
        "agent",
        "_agent_config",
        "core",
        "_training_config",
        "seed",
        "_wandb_conf",
        "_logger",
    ]

    def __init__(
        self,
        policy: PolicyConfig,
        agent: TAgentConfig,
        train: TrainingConfig,
        env: EnvConfig,
        wandb: WandbConfig,
        seed: Optional[int],
    ) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)
        self.seed = seed
        np.random.seed(seed)
        self._mdp_config = env
        self._create_mdp()
        self._policy_config = policy
        self._create_policy()
        self._agent_config = agent
        self._create_agent()
        self._create_core()
        self._training_config = train
        self._wandb_conf = {
            **asdict(wandb),
            "config": {**asdict(agent), **asdict(policy), **asdict(train), **asdict(env)},
        }

    def _create_mdp(self) -> None:
        """Initialize the MDP environment from configuration."""
        mdp_name = self._mdp_config.name
        if "-" not in mdp_name:
            raise ConfigurationError(f"The following mdp ({mdp_name}) is not supported.")
        self.gamma = self._mdp_config.core.gamma
        try:
            self.mdp = Gym(mdp_name, horizon=self._mdp_config.core.horizon, gamma=self.gamma)
            self.mdp.seed(self.seed)
            self._logger.info(f"initialized mdp: {mdp_name} with gamma={self.gamma}")
        except Exception as e:
            raise ConfigurationError(f"Failed to create environment '{mdp_name}': {e}")

    @abstractmethod
    def _create_policy(self) -> None:
        """Create the policy for the agent."""
        pass

    @abstractmethod
    def _create_agent(self) -> None:
        """Create the RL agent."""
        pass

    def _create_core(self):
        """Create the MushroomRL Core object."""
        if not hasattr(self, "mdp"):
            raise ExperimentError("Mdp must be created before core. Call create_mdp() first.")
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
        use_wandb: bool = False,
    ) -> None:
        """
        Train the agent with periodic evaluation.
        """

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

        eval_policy_param = Parameter(self._training_config.eval_policy_param)
        test_samples = self._training_config.test_samples

        initial_J, initial_dJ = self._evaluate(test_samples, eval_policy_param)
        J_list.append(initial_J)
        dJ_list.append(initial_dJ)

        if wandb_active:
            self._log_to_wandb(J=initial_J, dJ=initial_dJ, step=0)

        n_epochs = self._training_config.n_epochs
        n_steps = self._training_config.n_steps
        train_frequency = self._training_config.train_frequency

        self._logger.info(
            f"Starting training: {n_epochs} epochs, {n_steps} steps per epoch and {train_frequency} steps per fit"
        )

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
