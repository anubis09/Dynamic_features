from typing import List, Tuple, final

import torch
from mushroom_rl.algorithms.value import DQN
from mushroom_rl.approximators.parametric.torch_approximator import TorchApproximator
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from typing_extensions import override

from dynamic_features.backbones.Linear import Linear
from dynamic_features.configs import DqnAgentConfig, EnvConfig, PolicyConfig, TrainingConfig, WandbConfig

from .abstract_experiment import AbstractExp


@final
class DqnExperiment(AbstractExp[DqnAgentConfig]):
    """Concrete experiment implementation for Deep Q-Networks (DQN).

    This class extends `AbstractExp` to implement the specific logic for DQN,
    including the initialization of the PyTorch approximation network and
    the management of the replay buffer warm-up phase.

    Attributes:
        _initial_replay_size (int): Number of steps to populate the replay buffer
            with random actions before the actual training starts.
        _max_replay_size (int): The maximum capacity of the experience replay buffer.
    """

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
        """Initializes the DQN experiment and seeds the PyTorch backend.

        Args:
            policy (PolicyConfig): Configuration for the agent's policy.
            agent (DqnAgentConfig): Algorithm-specific agent configuration.
            train (TrainingConfig): Training hyperparameters (epochs, steps).
            env (EnvConfig): Environment parameters (id, horizon, gamma).
            wandb (WandbConfig): Weights & Biases logging configuration.
            seed (int): Random seed for reproducibility.
        """
        # Seeding must happen before super().__init__ triggers _setup()
        _ = torch.manual_seed(seed)
        self._initial_replay_size: int = agent.initial_replay_size
        self._max_replay_size: int = agent.max_replay_size
        super().__init__(policy=policy, agent=agent, train=train, env=env, wandb=wandb, seed=seed)

    @override
    def _configure_policy_exploration_behavior(self, param: Parameter) -> None:
        """Updates the epsilon parameter for the EpsGreedy policy."""
        self._policy.set_epsilon(param)

    @override
    def _create_policy(self) -> None:
        """Creates an EpsGreedy policy initialized with epsilon=1.0.

        The policy starts fully random to support the replay buffer warm-up phase.
        The actual decay schedule is applied during the training loop.
        """
        # in DQN we start with a random epsilon
        epsilon_random = Parameter(value=1.0)
        self._policy: EpsGreedy = EpsGreedy(epsilon=epsilon_random)

    @override
    def _create_agent(self) -> None:
        """Instantiates the DQN agent with a Linear PyTorch approximator.

        This sets up the optimizer, loss function, and network architecture based
        on the agent configuration and environment info.
        """

        # Approximator
        input_shape = self._mdp.info.observation_space.shape
        output_shape = self._mdp.info.action_space.size
        approximator_params = dict(
            network=Linear,
            optimizer={
                "class": self._agent_config.optimizer.get_class(),
                "params": {"lr": self._agent_config.learning_rate},
            },
            loss=self._agent_config.loss.get_function(),
            n_features=self._agent_config.n_features,
            input_shape=input_shape,
            output_shape=output_shape,
            n_actions=self._mdp.info.action_space.n,
        )

        self._logger.info(
            f"Initializing DQN Agent: "
            f"Input={input_shape}, Output={output_shape}, "
            f"LR={self._agent_config.learning_rate}, Batch={self._agent_config.batch_size}"
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
        """Executes the training pipeline including the replay buffer warm-up.

        This method first runs a 'warm-up' phase to fill the replay buffer with
        random samples (using the initial policy state), and then delegates
        to the parent `train` method for the actual learning loop.

        Args:
            use_wandb (bool): Whether to enable WandB logging.

        Returns:
            Tuple[List[float], List[float]]: The training metrics (J, dJ).
        """
        # Fill replay memory using the initial policy state (Epsilon=1.0)
        self._logger.info(f"Filling replay memory ({self._agent_config.initial_replay_size} steps)...")
        self._core.learn(n_steps=self._initial_replay_size, n_steps_per_fit=self._initial_replay_size)
        return super().train(
            use_wandb=use_wandb,
        )
