from typing import Dict, Optional, final

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
    __slots__ = ["_n_tilings", "_n_tiles", "_tilings", "_features"]

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

    def _train_feature_representation(self, is_retrain: bool = False) -> Dict:
        """Collects environment states, trains the tiling clustering and approximator, and returns accuracy metrics.

        On initial training (`is_retrain=False`) states are sampled randomly from the MDP.
        On retraining (`is_retrain=True`) states are collected via ``self._core``, biasing the
        sample toward regions the agent actually visits.

        Args:
            is_retrain: If ``True``, collects states on-policy via ``self._core`` and resets the
                        approximator heads before retraining (the KMeans tilings are always refit
                        from scratch). If ``False``, samples states randomly and trains the
                        approximator from its current initialization.

        Returns:
            Dict mapping approximator accuracy metric names to their scalar values.
        """
        core = self._core if is_retrain else None
        n_states = self._agent_config.clustering_training_states_multiplier * self._n_tiles
        X_list = [get_states_from_mdp(mdp=self._mdp, n_states=n_states, core=core) for _ in range(self._n_tilings)]
        self._tilings.train(X_per_tilings=X_list)
        X_train = np.concatenate(X_list, axis=0)
        X_test = get_states_from_mdp(mdp=self._mdp, n_states=n_states, core=core)
        per_head_acc, overall_acc = self._tilings.train_approximator(X=X_train, X_test=X_test, reset_heads=is_retrain)
        return {
            "approximator/accuracy": overall_acc,
            **{f"approximator/head_{i}_accuracy": acc.item() for i, acc in enumerate(per_head_acc)},
        }

    def _create_features(self) -> None:
        """Creates clustering based features for linear function approximation.

        Instantiates the ``ClusteringFeatures`` object, runs the initial feature representation
        training via :meth:`_train_feature_representation`, and wraps the result in a MushroomRL
        ``Features`` object for use by the agent.
        """
        input_shape = self._mdp.info.observation_space.shape[0]
        self._tilings = ClusteringFeatures(
            input_shape=input_shape,
            n_tilings=self._n_tilings,
            n_clusters=self._n_tiles,
            approximator_config=self._agent_config.approximator,
            seed=self._seed,
        )
        self._logger.debug(f"input shape: {input_shape}")
        self._pre_train_metrics = self._train_feature_representation()
        self._features = Features(n_outputs=self._tilings.size, function=self._tilings.encode_approx)

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

    @override
    def _on_epoch_end(self, epoch: int) -> Optional[Dict]:
        """Optionally retrains the feature representation at the configured interval.

        Snapshots the current exploration epsilon before retraining so that the policy rollouts
        used for on-policy state collection do not consume scheduled exploration steps.

        Args:
            epoch: Current training epoch index.

        Returns:
            Approximator accuracy metrics if retraining occurred, else ``None``.
        """
        if (
            self._agent_config.retrain_features
            and epoch % self._agent_config.retrain_interval == 0
            and epoch != self._training_config.n_epochs
        ):
            snapshot_eps = self._exploration_policy_param.get_value()
            self._configure_policy_exploration_behavior(Parameter(snapshot_eps))
            return self._train_feature_representation(is_retrain=True)
        return None
