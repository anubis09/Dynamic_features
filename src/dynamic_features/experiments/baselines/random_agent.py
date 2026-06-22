from typing import final

import numpy as np
from mushroom_rl.core import Agent
from mushroom_rl.policy import Policy
from mushroom_rl.utils.parameters import Parameter
from typing_extensions import override

from dynamic_features.configs import EnvConfig, PolicyConfig, RandomAgentConfig, TrainingConfig, WandbConfig
from dynamic_features.experiments import AbstractExp


class RandomPolicy(Policy):
    """A policy that samples actions uniformly at random from a discrete action space.

    Unlike ``EpsGreedy``, this policy never queries a Q-approximator, so the random
    agent needs no function approximator at all.
    """

    def __init__(self, n_actions: int) -> None:
        """Constructor.

        Args:
            n_actions (int): number of discrete actions available in the environment.
        """
        super().__init__()
        self._n_actions = n_actions
        self._add_save_attr(_n_actions="primitive")

    def __call__(self, *args) -> np.ndarray:
        """Returns the (uniform) probability of each action in the given state."""
        return np.ones(self._n_actions) / self._n_actions

    def draw_action(self, state) -> np.ndarray:
        """Samples a single action uniformly at random, ignoring the state."""
        return np.array([np.random.choice(self._n_actions)])


class RandomAgent(Agent):
    """An agent that acts via ``RandomPolicy`` and never learns."""

    def fit(self, dataset, **info) -> None:
        """No-op: the random agent does not learn from experience."""
        pass


@final
class RandomExperiment(AbstractExp[RandomAgentConfig]):
    """Baseline experiment whose agent always takes uniformly random actions.

    Useful as a lower-bound reference for the learning agents. Both training and
    evaluation use the same uniform-random behaviour, since
    ``_configure_policy_exploration_behavior`` is a no-op.
    """

    __slots__ = []

    def __init__(
        self,
        policy: PolicyConfig,
        agent: RandomAgentConfig,
        train: TrainingConfig,
        env: EnvConfig,
        wandb: WandbConfig,
        seed: int,
    ) -> None:
        super().__init__(policy=policy, agent=agent, train=train, env=env, wandb=wandb, seed=seed)

    @override
    def _configure_policy_exploration_behavior(self, param: Parameter) -> None:
        """No-op: the random agent ignores the exploration schedule and the
        evaluation parameter, staying uniformly random at all times."""
        pass

    @override
    def _create_policy(self) -> None:
        """Creates a uniform-random policy over the environment's discrete actions."""
        self._policy: RandomPolicy = RandomPolicy(self._mdp.info.action_space.n)

    @override
    def _create_agent(self) -> None:
        """Creates the non-learning random agent."""
        self._agent: RandomAgent = RandomAgent(self._mdp.info, self._policy)
        self._logger.info("Initialized Random Agent (uniform random actions, no learning)")
