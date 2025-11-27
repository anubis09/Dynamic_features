import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from logging import Logger
from typing import Generic, List, Tuple

import numpy as np
import wandb
from mushroom_rl.core import Core
from mushroom_rl.environments import Gym
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from dynamic_features.configs.env_config import EnvConfig
from dynamic_features.configs.experiments.experiment_conf import PolicyConfig, TAgentConfig, TrainingConfig
from dynamic_features.configs.wandb_config import WandbConfig


# Putting here Generic[TAgentConfig] allows the class to be generic over the type of agent configuration
# A subclass of AbstractAlgorith can use a specific subclass of AgentConfig without getting lsp errors.
class AbstractExp(ABC, Generic[TAgentConfig]):
    __slots__ = [
        # Environment
        "_mdp",
        "_mdp_config",
        "_gamma",
        # Policy
        "_exploration_policy_param",
        "_policy",
        "_policy_config",
        # Agent & Core
        "_agent",
        "_agent_config",
        "_core",
        # Configuration
        "_training_config",
        "_seed",
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
        seed: int,
    ) -> None:
        """Save all configs and initialize all components."""
        self._logger: Logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)

        # Store all configs
        self._seed = seed
        self._mdp_config = env
        self._policy_config = policy
        self._agent_config = agent
        self._training_config = train
        self._wandb_conf = {
            **asdict(wandb),
            "config": {**asdict(agent), **asdict(policy), **asdict(train), **asdict(env)},
        }

        # Initialize all components
        self._setup()

    def _setup(self) -> None:
        """Initialize all components in the correct order."""
        np.random.seed(self._seed)
        self._create_mdp()
        self._create_policy()
        self._create_agent()
        self._create_core()
        self._logger.info("Experiment setup complete")

    def _create_mdp(self) -> None:
        """Initialize the MDP environment from configuration."""
        mdp_name = self._mdp_config.name
        if "-" not in mdp_name:
            raise ConfigurationError(f"The following mdp ({mdp_name}) is not supported.")
        self._gamma: float = self._mdp_config.core.gamma
        try:
            self._mdp: Gym = Gym(mdp_name, horizon=self._mdp_config.core.horizon, gamma=self._gamma)
            self._mdp.seed(self._seed)
            self._logger.info(f"initialized mdp: {mdp_name} with gamma={self._gamma}")
        except Exception as e:
            raise ConfigurationError(f"Failed to create environment '{mdp_name}': {e}")

    def _create_policy(self) -> None:
        """Create exploration parameter and policy."""
        self._create_exploration_param()
        self._create_policy_instance()

    def _create_exploration_param(self) -> None:
        """Create the exploration parameter."""
        self._exploration_policy_param = LinearParameter(
            value=self._policy_config.initial_exp_rate,
            threshold_value=self._policy_config.final_exp_rate,
            n=self._policy_config.final_exp_frame,
        )

    @abstractmethod
    def _create_policy_instance(self) -> None:
        """Create the actual policy object. Must set self._policy."""
        pass

    @abstractmethod
    def _create_agent(self) -> None:
        """Create the RL agent."""
        pass

    def _create_core(self) -> None:
        """Create the MushroomRL Core object."""
        self._core: Core = Core(agent=self._agent, mdp=self._mdp)
        self._logger.info("Core initialized successfully")

    def _configure_policy_exploration_behavior(self, param: Parameter) -> None:
        """Configure policy behavior based on parameter value."""
        if hasattr(self._policy, "set_epsilon"):
            self._policy.set_epsilon(param)
        elif hasattr(self._policy, "set_beta"):
            self._policy.set_beta(param)
        else:
            raise ExperimentError("Policy has neither epsilon nor beta parameter")

    def _initialize_wandb(self) -> bool:
        """Initialize wandb and return success status."""
        if not self._wandb_conf:
            return False
        try:
            wandb.init(**self._wandb_conf)
            self._logger.info("Wandb initialized successfully")
            return True
        except Exception as e:
            self._logger.warning(f"Failed to initialize wandb: {e}")
            return False

    def train(
        self,
        use_wandb: bool = False,
    ) -> Tuple[List[np.float_], List[np.float_]]:
        """
        Train the agent with periodic evaluation.
        """
        # Initialize wandb if requested and configured
        wandb_active = use_wandb and self._initialize_wandb()

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
            self._configure_policy_exploration_behavior(self._exploration_policy_param)
            self._core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)

            # Evaluation step
            J_curr, dJ_curr = self._evaluate(test_samples, eval_policy_param)
            J_list.append(J_curr)
            dJ_list.append(dJ_curr)
            if wandb_active:
                self._log_to_wandb(J=J_curr, dJ=dJ_curr, step=epoch)

        if wandb_active:
            wandb.finish()

        self._logger.info("Training completed")
        return J_list, dJ_list

    def _evaluate(self, test_samples: int, eval_exploration_param: Parameter) -> Tuple[np.float_, np.float_]:
        """Evaluate the current policy and return average returns."""

        # Set deterministic policy for evaluation
        self._configure_policy_exploration_behavior(eval_exploration_param)

        dataset = self._core.evaluate(n_steps=test_samples, render=False)

        J = np.mean(compute_J(dataset, gamma=1.0))  # Undiscounted return
        dJ = np.mean(compute_J(dataset, gamma=self._gamma))  # Discounted return

        return J, dJ

    def _log_to_wandb(self, J: np.float_, dJ: np.float_, step: int) -> None:
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
        if not hasattr(self, "_agent"):
            raise ExperimentError("agent not initialized")

        if not path.endswith(".msh"):
            path = f"{path}.msh"

        try:
            self._agent.save(path)
            self._logger.info(f"Agent saved successfully to {path}")
        except Exception as e:
            raise ExperimentError(f"Failed to save agent to {path}: {e}")


class ExperimentError(Exception):
    """Custom exception for experiment-related errors"""

    pass


class ConfigurationError(ExperimentError):
    """Exception raised for configuration-related errors"""

    pass
