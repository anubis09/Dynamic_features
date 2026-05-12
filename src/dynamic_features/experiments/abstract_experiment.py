import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from logging import Logger
from typing import Dict, Generic, List, Optional, Tuple

import numpy as np
import wandb
from mushroom_rl.core import Core, Environment
from mushroom_rl.utils.dataset import compute_J
from mushroom_rl.utils.parameters import LinearParameter, Parameter

from dynamic_features.configs import EnvConfig, PolicyConfig, TAgentConfig, TrainingConfig, WandbConfig

from .exceptions import ConfigurationError, ExperimentError


# Putting here Generic[TAgentConfig] allows the class to be generic over the type of agent configuration
# A subclass of AbstractAlgorith can use a specific subclass of AgentConfig without getting lsp errors.
class AbstractExp(ABC, Generic[TAgentConfig]):
    """Base abstract class for running reproducible RL experiments for my thesis.

    This class implements the Template Method pattern to standardize the
    initialization, training, evaluation, and logging loop for RL agents
    using the MushroomRL library.

    Subclasses must implement the abstract methods to define specific
    agent and policy creation logic.

    Attributes:
        _mdp (Gym): The MushroomRL wrapped Gym environment instance.
        _mdp_config (EnvConfig): Configuration object for the environment.
        _gamma (float): Discount factor extracted from the environment config.

        _policy (Policy): The initialized MushroomRL policy.
        _policy_config (PolicyConfig): Configuration object for the policy.
        _exploration_policy_param (Parameter): The exploration schedule (e.g. LinearParameter)
            used during training.

        _agent (Agent): The initialized MushroomRL agent.
        _agent_config (TAgentConfig): Configuration object for the agent.
        _core (Core): The MushroomRL Core object managing the agent-environment loop.

        _training_config (TrainingConfig): Configuration for training epochs and steps.
        _seed (int): Global random seed applied to Numpy, Python, and Gym.
        _wandb_conf (dict): Dictionary configuration for Weights & Biases logging.
        _logger (Logger): Class-specific Python logger instance.
    """

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
        "_pre_train_metrics",
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
        """Initializes the experiment configuration and components.

        This method stores the configurations, sets up namespaces for logging,
        calculates derived parameters (like gamma), and triggers the
        component setup sequence.

        Args:
            policy (PolicyConfig): Configuration for the agent's policy.
            agent (TAgentConfig): Algorithm-specific agent configuration.
            train (TrainingConfig): Training hyperparameters (epochs, steps).
            env (EnvConfig): Environment parameters (id, horizon, gamma).
            wandb (WandbConfig): Weights & Biases logging configuration.
            seed (int): Random seed for reproducibility.
        """
        self._logger: Logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.DEBUG)

        # Store all configs
        self._mdp_config = env
        self._policy_config = policy
        self._agent_config = agent
        self._training_config = train
        wandb_dict = asdict(wandb)
        self._wandb_conf = {
            **wandb_dict,
            "config": {
                # This is to not overwrite possible values of config.
                **wandb_dict.get("config", {}),
                "agent": asdict(agent),
                "policy": asdict(policy),
                "train": asdict(train),
                "env": asdict(env),
            },
        }

        self._seed = seed
        self._gamma: float = env.core.gamma
        self._exploration_policy_param: LinearParameter = LinearParameter(
            value=policy.initial_exp_rate,
            threshold_value=policy.final_exp_rate,
            n=policy.final_exp_frame,
        )

        self._pre_train_metrics: Dict[str, float] = {}
        self._setup()

    def _setup(self) -> None:
        """Orchestrates the initialization of all RL components.

        This method ensures the correct instantiation order:
        1. Seed numpy random generator
        2. Create environment (also seeded)
        3. Create policy
        4. Create agent
        5. Create core
        """
        self._logger.info(f"Initializing experiment with seed {self._seed}")
        np.random.seed(self._seed)
        self._create_mdp()
        self._create_policy()
        self._create_agent()
        self._create_core()
        self._logger.info("Experiment setup complete")

    def _create_mdp(self) -> None:
        """Initializes the MDP environment from configuration.

        Raises:
            ConfigurationError: If the environment ID is invalid, missing dependencies,
                or fails to initialize via the MushroomRL Gym wrapper.
        """
        mdp_name = self._mdp_config.name
        try:
            self._mdp: Environment = Environment.make(
                mdp_name, horizon=self._mdp_config.core.horizon, gamma=self._gamma
            )
            self._mdp.seed(self._seed)
            self._logger.info(
                f"Initialized environment: {mdp_name} (horizon={self._mdp._mdp_info.horizon}, γ={self._gamma})"
            )
        except Exception as e:
            self._logger.error(f"Failed to create environment '{mdp_name}': {e}")
            raise ConfigurationError(f"Failed to create environment '{mdp_name}': {e}") from e

    @abstractmethod
    def _create_policy(self) -> None:
        """Abstract method to instantiate the policy.

        Implementations should initialize `self._policy` using `self._policy_config`.
        """

    @abstractmethod
    def _create_agent(self) -> None:
        """Abstract method to instantiate the RL Agent.

        Implementations should initialize `self._agent` using `self._agent_config`,
        `self._mdp.info`, and `self._policy`.
        """
        pass

    def _create_core(self) -> None:
        """Creates the MushroomRL Core object to manage the agent-environment loop."""
        self._core: Core = Core(agent=self._agent, mdp=self._mdp)
        self._logger.info("Core initialized successfully")

    @abstractmethod
    def _configure_policy_exploration_behavior(self, param: Parameter) -> None:
        """Updates the policy's exploration parameter for the current step.

        Args:
            param (Parameter): The current exploration parameter (e.g., epsilon or beta).
        """

    def _initialize_wandb(self) -> bool:
        """Initializes the WandB run if configuration is present.

        Returns:
            bool: True if WandB was successfully initialized, False otherwise.
        """
        if not self._wandb_conf:
            return False
        try:
            wandb.init(**self._wandb_conf)
            self._logger.info(
                f"WandB initialized: project={self._wandb_conf.get('project')}, group={self._wandb_conf.get('group')}"
            )
            return True
        except Exception as e:
            self._logger.warning(f"Failed to initialize WandB: {e}")
            return False

    def train(
        self,
        use_wandb: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """Executes the main training loop.

        This method runs the experiment for `n_epochs`. In each epoch, it:
        1. Updates the exploration parameter.
        2. Learns for `n_steps` (divided by `train_frequency` fits).
        3. Evaluates the agent on test episodes.
        4. Logs metrics to WandB (if enabled).

        Args:
            use_wandb (bool, optional): Whether to log metrics to Weights & Biases.
                Defaults to False.

        Returns:
            Tuple[List[float], List[float]]: A tuple containing two lists:
                - J_list: Average undiscounted returns per epoch.
                - dJ_list: Average discounted returns per epoch.
        """
        # Initialize wandb if requested and configured
        wandb_active = use_wandb and self._initialize_wandb()

        J_list = []
        dJ_list = []

        eval_policy_param = Parameter(self._training_config.eval_policy_param)
        test_episodes = self._training_config.test_episodes
        final_test_episodes = self._training_config.final_test_episodes

        initial_J, initial_dJ = self._evaluate(test_episodes, eval_policy_param)
        J_list.append(initial_J)
        dJ_list.append(initial_dJ)
        self._logger.info(f"Initial evaluation: J={initial_J:.3f}, dJ={initial_dJ:.3f}")

        if wandb_active:
            self._log_to_wandb(J=initial_J, dJ=initial_dJ, step=0, extra_metrics=self._pre_train_metrics)

        n_epochs = self._training_config.n_epochs
        n_steps = self._mdp_config.budget // n_epochs
        train_frequency = self._training_config.train_frequency

        self._logger.info(
            f"Starting training: {n_epochs} epochs, {n_steps} steps/epoch, "
            f"{train_frequency} steps/fit, {test_episodes} test episodes"
        )

        log_frequency = self._training_config.log_frequency

        for epoch in range(1, n_epochs + 1):
            # Learning step
            self._configure_policy_exploration_behavior(self._exploration_policy_param)
            self._core.learn(n_steps=n_steps, n_steps_per_fit=train_frequency)

            if epoch == n_epochs:
                # if training is over, we want to evaluate with a final test episodes.
                test_episodes = final_test_episodes
            # Evaluation step
            J_curr, dJ_curr = self._evaluate(test_episodes, eval_policy_param)
            J_list.append(J_curr)
            dJ_list.append(dJ_curr)

            if epoch % log_frequency == 0 or epoch == n_epochs:
                exp_param = self._exploration_policy_param.get_value()
                self._logger.info(
                    f"Epoch {epoch}/{n_epochs}: J={J_curr:.3f}, dJ={dJ_curr:.3f}, exp_param={exp_param:.3f}"
                )

            if wandb_active:
                self._log_to_wandb(J=J_curr, dJ=dJ_curr, step=epoch)

        if wandb_active:
            wandb.finish()

        self._logger.info(f"Training completed: Final J={J_list[-1]:.3f}, dJ={dJ_list[-1]:.3f}")
        return J_list, dJ_list

    def _evaluate(self, test_episodes: int, eval_exploration_param: Parameter) -> Tuple[float, float]:
        """Performs a deterministic evaluation of the current policy.

        Args:
            test_episodes (int): Number of episodes to simulate for evaluation.
            eval_exploration_param (Parameter): Fixed parameter for evaluation
                (usually 0.0 for epsilon-greedy).

        Returns:
            Tuple[float, float]: A pair of (Undiscounted Return, Discounted Return).
        """
        # Set deterministic policy for evaluation
        self._configure_policy_exploration_behavior(eval_exploration_param)

        dataset = self._core.evaluate(n_episodes=test_episodes, render=False)

        J = np.mean(compute_J(dataset, gamma=1.0)).item()  # Undiscounted return
        dJ = np.mean(compute_J(dataset, gamma=self._gamma)).item()  # Discounted return

        return J, dJ

    def _log_to_wandb(self, J: float, dJ: float, step: int, extra_metrics: Optional[Dict[str, float]] = None) -> None:
        """Logs scalar metrics to WandB.

        Args:
            J (float): Average undiscounted return.
            dJ (float): Average discounted return.
            step (int): Current epoch number.
            extra_metrics (dict, optional): Additional scalar metrics merged into
                the same log call and therefore aligned to the same step.
        """
        group_name = self._wandb_conf.get("group")
        metrics = {
            f"{group_name}/average_return": J,
            f"{group_name}/discounted_average_return": dJ,
        }
        if extra_metrics:
            metrics.update(extra_metrics)
        wandb.log(metrics, step=step, commit=True)

    def save_agent(self, path: str) -> None:
        """Persists the agent state to disk.

        Args:
            path (str): The target file path. The extension '.msh' will be
                appended if missing.

        Raises:
            ExperimentError: If the serialization fails.
        """

        if not path.endswith(".msh"):
            path = f"{path}.msh"

        try:
            self._agent.save(path)
            self._logger.info(f"Agent saved successfully to {path}")
        except Exception as e:
            self._logger.error(f"Failed to save agent to '{path}': {e}")
            raise ExperimentError(f"Failed to save agent to {path}: {e}") from e
