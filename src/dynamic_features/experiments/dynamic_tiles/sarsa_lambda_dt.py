from typing import final

import numpy as np
from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.features import Features
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from typing_extensions import override

from dynamic_features.configs import EnvConfig, PolicyConfig, SarsaLambdaDTAgentConfig, TrainingConfig, WandbConfig
from dynamic_features.experiments import AbstractExp
from dynamic_features.methods import ClusteringFeatures, get_states_from_mdp


@final
class SarsaLambdaDTExperiment(AbstractExp[SarsaLambdaDTAgentConfig]):
    __slots__ = ["_n_tilings", "_n_tiles", "_features"]

    def __init__(
        self,
        policy: PolicyConfig,
        agent: SarsaLambdaDTAgentConfig,
        train: TrainingConfig,
        env: EnvConfig,
        wandb: WandbConfig,
        seed: int,
    ) -> None:
        """Concrete experiment implementation for True Online SARSA(λ).

        This class extends `AbstractExp` to implement the specific logic for SARSA(λ),
        including the initialization of clustring based tile coding features
        for function approximation and the configuration of the eligibility trace mechanism.

        Attributes:
            _n_tilings (int): Number of tilings used in the tile coding feature representation.
            _n_tiles (int): Number of tiles per dimension in each tiling.
            _features (Features): Clustering-based feature extractor for state representation.
        """
        self._n_tilings: int = agent.n_tilings
        self._n_tiles: int = agent.n_clusters
        super().__init__(policy=policy, agent=agent, train=train, env=env, wandb=wandb, seed=seed)

    @override
    def _configure_policy_exploration_behavior(self, param: Parameter) -> None:
        """Updates the epsilon parameter for the EpsGreedy policy."""
        self._policy.set_epsilon(param)

    @override
    def _create_policy(self) -> None:
        """Creates an EpsGreedy policy initialized with epsilon=1.0.

        The policy starts with 1.0 but this is just for initialization purposes.
        The actual decay schedule is applied during the training loop.
        """
        self._policy: EpsGreedy = EpsGreedy(epsilon=1.0)

    def _create_features(self) -> None:
        """Creates clustering based features for linear function approximation.

        This method generates multiple clustering algorithms(tilings) that
        partition the state space into tiles, providing a rich feature
        representation.
        """
        input_shape = self._mdp.info.observation_space.shape[0]
        tilings = ClusteringFeatures(
            input_shape=input_shape,
            n_tilings=self._n_tilings,
            n_clusters=self._n_tiles,
            approximator_config=self._agent_config.approximator,
            seed=self._seed,
        )
        self._logger.debug(f"input shape: {input_shape}")
        n_states_for_train = self._agent_config.clustering_training_states_multiplier * self._n_tiles
        X_list = [get_states_from_mdp(mdp=self._mdp, n_states=n_states_for_train) for _ in range(self._n_tilings)]
        tilings.train(X_per_tilings=X_list)
        X_train = np.concatenate(X_list, axis=0)
        X_test = get_states_from_mdp(mdp=self._mdp, n_states=n_states_for_train)
        per_head_acc, overall_acc = tilings.train_approximator(X=X_train, X_test=X_test)
        self._pre_train_metrics = {
            "approximator/accuracy": overall_acc,
            **{f"approximator/head_{i}_accuracy": acc.item() for i, acc in enumerate(per_head_acc)},
        }
        self._features = Features(n_outputs=tilings.size, function=tilings.encode_approx)

    @override
    def _create_agent(self) -> None:
        """Instantiates the True Online SARSA(λ) agent with tile coding features.

        This sets up the linear function approximator with tile coding features,
        configuring the learning rate and eligibility trace decay (lambda) based
        on the agent configuration and environment info.
        """

        self._create_features()

        approximator_params = dict(
            input_shape=(self._features.size,),
            output_shape=(self._mdp.info.action_space.n,),
            n_actions=self._mdp.info.action_space.n,
        )

        self._logger.info(
            f"Initializing True Online SARSA(λ) Agent: "
            f"N_tilings={self._n_tilings}, N_tiles={self._n_tiles}, "
            f"LR={self._agent_config.learning_rate}, Lambda={self._agent_config.lambda_coeff}"
        )

        scaled_learning_rate = Parameter(self._agent_config.learning_rate / self._n_tilings)

        # Agent
        self._agent: TrueOnlineSARSALambda = TrueOnlineSARSALambda(
            self._mdp.info,
            self._policy,
            approximator_params=approximator_params,
            features=self._features,
            learning_rate=scaled_learning_rate,
            lambda_coeff=self._agent_config.lambda_coeff,
        )
