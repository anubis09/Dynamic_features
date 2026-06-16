from typing import Dict, List, Optional, final

import numpy as np
from mushroom_rl.algorithms.value import TrueOnlineSARSALambda
from mushroom_rl.features import Features
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from sklearn.metrics import adjusted_rand_score
from typing_extensions import override

from dynamic_features.configs import (
    EnvConfig,
    PolicyConfig,
    RegularizationPrior,
    RetrainTrigger,
    SarsaLambdaDTAgentConfig,
    TrainingConfig,
    WandbConfig,
)
from dynamic_features.experiments import AbstractExp
from dynamic_features.methods import ClusteringFeatures, get_states_from_mdp


@final
class SarsaLambdaDTExperiment(AbstractExp[SarsaLambdaDTAgentConfig]):
    """Concrete experiment running True Online SARSA(λ) over clustering-based tile coding.

    Extends :class:`AbstractExp` with the SARSA(λ)-specific logic: it builds the linear
    function approximator on top of KMeans clustering features, scales the learning rate
    by the number of tilings, and drives the eligibility-trace agent. When feature
    retraining is enabled the tilings are periodically refit on freshly collected
    on-policy states, after which the learned Q-function is re-projected onto the new
    feature basis (see :meth:`_naive_weight_transfer`).

    Attributes:
        _n_tilings (int): Number of tilings in the tile coding representation; also the
            divisor applied to the configured learning rate.
        _n_tiles (int): Number of KMeans clusters per tiling (one tile per cluster).
        _tilings (ClusteringFeatures): Clustering-based encoder backing the feature
            extractor; owns the KMeans models and the approximator.
        _features (Features): MushroomRL feature wrapper consumed by the agent.
    """

    __slots__ = [
        "_n_tilings",
        "_n_tiles",
        "_tilings",
        "_features",
        "_drift_reference",
        "_last_retrain_epoch",
        "_drift_trigger_count",
    ]

    def __init__(
        self,
        policy: PolicyConfig,
        agent: SarsaLambdaDTAgentConfig,
        train: TrainingConfig,
        env: EnvConfig,
        wandb: WandbConfig,
        seed: int,
    ) -> None:
        """Stores the tiling/cluster counts and defers the rest of setup to ``AbstractExp``."""
        self._n_tilings: int = agent.n_tilings
        self._n_tiles: int = agent.n_clusters
        self._drift_reference: Optional[float] = None
        self._last_retrain_epoch: Optional[int] = None
        self._drift_trigger_count: int = 0
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

    def _train_feature_representation(self, X_list: List[np.ndarray], X_test: np.ndarray) -> Dict:
        """Trains the tiling clustering and approximator from pre-collected states.

        Args:
            X_list: One state array per tiling, used to fit the KMeans clusters.
            X_test: Held-out states for approximator evaluation.

        Returns:
            Dict mapping approximator accuracy metric names to their scalar values.
        """
        self._tilings.train(X_per_tilings=X_list)
        X_train = np.concatenate(X_list, axis=0)
        per_head_acc, overall_acc = self._tilings.train_approximator(X=X_train, X_test=X_test)
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
        n_states = self._agent_config.clustering_training_states_multiplier * self._n_tiles
        X_list = [get_states_from_mdp(mdp=self._mdp, n_states=n_states) for _ in range(self._n_tilings)]
        X_test = get_states_from_mdp(mdp=self._mdp, n_states=n_states)
        self._pre_train_metrics = self._train_feature_representation(X_list=X_list, X_test=X_test)
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

    @staticmethod
    def _relative_mae(q_ref: np.ndarray, q_other: np.ndarray) -> float:
        """Mean |q_ref - q_other| normalized by mean |q_ref|; NaN while q_ref is still ~0."""
        denom = np.mean(np.abs(q_ref))
        return float(np.mean(np.abs(q_ref - q_other)) / denom) if denom > 1e-8 else float("nan")

    def _naive_weight_transfer(self, X_list: List[np.ndarray], X_test: np.ndarray) -> Dict:
        """Re-projects the learned Q-function onto the retrained feature basis.

        When the tilings are refit the feature vectors change meaning, so the agent's
        weight vector becomes stale. The linear Q-function keeps an independent weight
        block per action (``Q(s, a) = phi(s) . w_a``), so per action this solves a
        ridge-regularized least squares for new weights that reproduce the old Q-values
        under the new features:

            min_{w_a_new} || phi_new @ w_a_new - phi_old @ w_a_old ||^2 + eps ||w_a_new||^2

        Because every action shares the same feature matrix, all action blocks are solved
        together as a single normal-equations system: one Gram matrix
        ``A = phi_new.T @ phi_new`` with a multi-column right-hand side (one column per
        action). The ridge term ``eps`` is scaled to the Gram matrix
        (``1e-3 * diag(A)``) so it conditions the solve without depending on the
        feature scale.

        The old representation/value function is snapshotted *before* retraining, the
        features are retrained, the new weights are solved and installed, and the
        eligibility trace is zeroed (the old trace indexes a feature space that no
        longer exists).

        Args:
            X_list: One on-policy state array per tiling (used to fit the clustering).
            X_test: Held-out on-policy states used for transfer diagnostics.

        Returns:
            Approximator-accuracy metrics merged with the transfer diagnostics.
        """
        X = np.concatenate(X_list, axis=0)
        n_states_wt = min(len(X), self._agent_config.weight_transfer_states_multiplier * self._features.size)
        X_train = X[np.random.choice(len(X), n_states_wt, replace=False)]
        n_actions = self._mdp.info.action_space.n

        old_phi_states = self._tilings.encode_approx(X_train)
        old_q_test = self._agent.Q.predict(self._tilings.encode_approx(X_test))  # (N_test, n_actions)
        old_clusters_test = self._tilings.predict(X_test)  # (N_test, n_tilings), for partition-drift
        old_weights = self._agent.Q.get_weights()
        old_weights = np.reshape(old_weights, (n_actions, self._features.size))

        old_q_values = old_phi_states @ old_weights.T  # (N_train, N_actions)

        to_log = self._train_feature_representation(X_list=X_list, X_test=X_test)

        new_phi_states = self._tilings.encode_approx(X_train)

        A = new_phi_states.T @ new_phi_states  # (n_features, n_features)
        b = new_phi_states.T @ old_q_values  # (n_features, n_actions)
        diag = np.diag(A).copy()
        diag[diag < 1e-12] = 1e-12  # -> if < 1e-12 the multiplication could become zero -> singular matrix
        ridge_term = self._agent_config.retrain_regularizer * diag  # (n_features,)
        A += np.diag(ridge_term)
        if self._agent_config.regularization_prior == RegularizationPrior.OLDWEIGHTS:
            b += ridge_term.reshape(-1, 1) * old_weights.T  # this way this operation becomes (n_features, n_actions)
        new_weights = np.linalg.solve(a=A, b=b)  # (n_features, n_actions)
        new_weights = new_weights.T.flatten()

        self._agent.Q.set_weights(new_weights)
        self._agent.e = np.zeros(self._agent.Q.weights_size)

        to_log.update(
            self._evaluate_weight_transfer(
                X_test=X_test,
                old_q_test=old_q_test,
                old_clusters_test=old_clusters_test,
                old_weights=old_weights,
            )
        )
        return to_log

    def _evaluate_weight_transfer(
        self,
        X_test: np.ndarray,
        old_q_test: np.ndarray,
        old_clusters_test: np.ndarray,
        old_weights: np.ndarray,
    ) -> Dict:
        """Computes diagnostics explaining whether the weight transfer preserved policy and value.

        Compares the post-transfer tilings/agent against the pre-transfer snapshots, all on the
        on-policy ``X_test`` (the distribution that tracks the return). All keys are prefixed
        ``transfer/`` and individually commented below.

        Args:
            X_test: Held-out on-policy states.
            old_q_test: Pre-transfer Q-values on ``X_test``, shape (N_test, n_actions).
            old_clusters_test: Pre-transfer cluster indices on ``X_test``, shape (N_test, n_tilings).
            old_weights: Pre-transfer flat weight vector.

        Returns:
            Dict mapping ``transfer/<name>`` metric keys to scalar floats.
        """
        # Post-transfer quantities, read from the already-updated tilings and agent.
        new_weights = self._agent.Q.get_weights()
        new_q_test = self._agent.Q.predict(self._tilings.encode_approx(X_test))
        new_clusters_test = self._tilings.predict(X_test)

        old_greedy, new_greedy = np.argmax(old_q_test, axis=1), np.argmax(new_q_test, axis=1)

        # Permutation-invariant partition drift, averaged over tilings.
        ari = np.mean(
            [adjusted_rand_score(old_clusters_test[:, t], new_clusters_test[:, t]) for t in range(self._n_tilings)]
        )

        old_weight_norm = np.linalg.norm(old_weights)

        return {
            # value fidelity: mean |Q_old - Q_new| relative to |Q_old| (0 = preserved, NaN early on)
            "transfer/rMAE_test": self._relative_mae(q_ref=old_q_test, q_other=new_q_test),
            # policy preservation: % of states with unchanged greedy action
            "transfer/preserved_greedy_actions": float(np.mean(old_greedy == new_greedy) * 100),
            # representation drift: adjusted Rand index of old vs new partition (1 = unchanged, ~0 = reshuffled)
            "transfer/cluster_ari": float(ari),
            # numerical sanity: ||w_new|| / ||w_old||, >> 1 flags a ridge/conditioning blow-up
            "transfer/weight_norm_ratio": float(np.linalg.norm(new_weights) / old_weight_norm)
            if old_weight_norm > 1e-8
            else float("nan"),
        }

    def _evaluate_split_half_stability(self, X_list: List[np.ndarray], X_test: np.ndarray) -> Dict:
        """Split-half KMeans stability check that isolates sampling noise from distribution drift.

        Reuses the states already collected for the retrain (no extra rollouts, which dominate the
        cost): the per-tiling arrays in ``X_list`` are pooled and split into two *disjoint* halves,
        each tiling is refit on an ``n_states`` subsample drawn from one half, and the two resulting
        partitions are compared on ``X_test`` (already an independent held-out on-policy batch) via
        the adjusted Rand index, averaged over tilings. The two fits never share a state, so this is
        a true disjoint split-half rather than a bootstrap; the pool (``n_tilings * n_states``) is far
        larger than the ``2 * n_states`` needed, so a disjoint split is always feasible here.

        Both clusterings are built with the same base seed, so per-tiling KMeans initialisations
        match and init variance cannot masquerade as sampling noise. Because both halves come from
        the same frozen-policy collection, genuine distribution drift is zero by construction: the
        only thing that can move the partition is the finite sample, so the resulting ARI is a pure
        sampling-stability score for the *current* training size:

            ARI ~ 1  -> ``n_states`` is enough; the partition is set by the distribution, not the draw.
            ARI < 1  -> the clustering is under-sampled at the current training size.

        This is the diagnostic counterpart to ``transfer/cluster_ari`` (old vs new partition across
        a retrain), which mixes sampling noise with genuine distribution drift and so cannot, on its
        own, tell whether a low value means "too few states" or "the distribution moved".

        Args:
            X_list: The per-tiling state arrays already collected for the retrain.
            X_test: The held-out on-policy batch already collected for the retrain, used as eval.

        Returns:
            Dict with the single ``transfer/splithalf_ari`` metric.
        """
        n_states = len(X_list[0])
        pool = np.concatenate(X_list, axis=0)
        rng = np.random.default_rng()
        perm = rng.permutation(len(pool))
        mid = len(perm) // 2
        disjoint_halves = (pool[perm[:mid]], pool[perm[mid:]])  # no state appears in both

        input_shape = self._mdp.info.observation_space.shape[0]
        fitted = []
        for source in disjoint_halves:
            # One independent n_states subsample per tiling, mirroring the real per-tiling fit size.
            per_tiling = [source[rng.choice(len(source), n_states, replace=False)] for _ in range(self._n_tilings)]
            cf = ClusteringFeatures(
                input_shape=input_shape,
                n_tilings=self._n_tilings,
                n_clusters=self._n_tiles,
                approximator_config=self._agent_config.approximator,
                seed=self._seed,
            )
            cf.train(X_per_tilings=per_tiling)
            fitted.append(cf)

        labels_a = fitted[0].predict(X_test)  # (N_test, n_tilings)
        labels_b = fitted[1].predict(X_test)  # (N_test, n_tilings)
        ari = np.mean([adjusted_rand_score(labels_a[:, t], labels_b[:, t]) for t in range(self._n_tilings)])
        return {"transfer/splithalf_ari": float(ari)}

    def _do_retrain(self, epoch: int) -> Dict:
        """Collects fresh on-policy states, refits the features, and transfers the Q-weights.

        Snapshots the current exploration epsilon before collection so that the on-policy
        rollouts used for state collection do not consume scheduled exploration steps. After
        the transfer, the drift reference ``I_ref`` is recomputed on the most recent evaluation
        batch under the *new* centroids and the last-retrain epoch is recorded, so subsequent
        drift checks measure staleness relative to this fresh fit.

        Args:
            epoch: Current training epoch index, recorded as the last-retrain epoch.

        Returns:
            Approximator-accuracy metrics merged with the transfer (and optional split-half)
            diagnostics.
        """
        snapshot_eps = self._exploration_policy_param.get_value()
        self._configure_policy_exploration_behavior(Parameter(snapshot_eps))
        n_states = self._agent_config.clustering_training_states_multiplier * self._n_tiles
        X_list = [
            get_states_from_mdp(mdp=self._mdp, n_states=n_states, core=self._core) for _ in range(self._n_tilings)
        ]
        X_test = get_states_from_mdp(mdp=self._mdp, n_states=n_states, core=self._core)
        self._configure_policy_exploration_behavior(self._exploration_policy_param)

        splithalf = (
            self._evaluate_split_half_stability(X_list=X_list, X_test=X_test)
            if self._agent_config.splithalf_stability_check
            else {}
        )
        metrics = self._naive_weight_transfer(X_list=X_list, X_test=X_test)
        metrics.update(splithalf)

        self._reset_drift_reference()
        self._last_retrain_epoch = epoch
        self._drift_trigger_count = 0
        return metrics

    def _reset_drift_reference(self) -> None:
        """Recomputes ``I_ref`` on the latest evaluation states under the current centroids.

        Uses the greedy-distribution states stashed by ``_evaluate`` (the same distribution the
        per-epoch drift check uses), so the drift ratio ``I_now / I_ref`` is apples-to-apples.
        No-op while no evaluation has run yet (reference stays ``None`` and is set lazily later).
        """
        if self._last_eval_states is not None:
            self._drift_reference = self._tilings.quantization_error(self._last_eval_states)

    def _drift_triggered_retrain(self, epoch: int) -> Optional[Dict]:
        """DRIFT strategy: retrain only when the tiling has gone stale, logging drift every epoch.

        Computes the inertia ratio ``I_now / I_ref`` on the latest greedy evaluation batch and
        retrains when it stays above ``drift_threshold`` for ``drift_patience`` consecutive checks,
        provided at least ``drift_min_interval`` epochs have elapsed since the last fit. The drift
        ratio and threshold are returned every epoch so the signal can be plotted continuously.

        Args:
            epoch: Current training epoch index.

        Returns:
            A metrics dict (drift signal, plus retrain diagnostics when a retrain fires), or
            ``None`` if drift cannot be measured yet (no evaluation states available).
        """
        if self._last_eval_states is None:
            return None
        # Lazily anchor the reference on the first available evaluation batch.
        if self._drift_reference is None:
            self._reset_drift_reference()
            return {"drift/ratio": 1.0, "drift/threshold": float(self._agent_config.drift_threshold)}

        i_now = self._tilings.quantization_error(self._last_eval_states)
        drift = i_now / self._drift_reference if self._drift_reference > 1e-12 else float("inf")
        threshold = float(self._agent_config.drift_threshold)
        log: Dict = {"drift/ratio": float(drift), "drift/threshold": threshold}

        cooled_down = (
            self._last_retrain_epoch is None
            or epoch - self._last_retrain_epoch >= self._agent_config.drift_min_interval
        )
        if drift > threshold and cooled_down:
            self._drift_trigger_count += 1
        else:
            self._drift_trigger_count = 0

        if cooled_down and self._drift_trigger_count >= self._agent_config.drift_patience:
            # _do_retrain resets the reference, the last-retrain epoch and the trigger count.
            metrics = self._do_retrain(epoch)
            metrics.update(log)
            return metrics
        return log

    @override
    def _on_epoch_end(self, epoch: int) -> Optional[Dict]:
        """Retrains the feature representation per the configured trigger strategy.

        Dispatches on ``retrain_trigger``: ``INTERVAL`` retrains on a fixed epoch schedule
        (legacy behavior), while ``DRIFT`` retrains only when the quantization-error drift
        ratio crosses its threshold and logs that ratio every epoch. Never retrains on the
        final epoch.

        Args:
            epoch: Current training epoch index.

        Returns:
            Metrics to log this epoch (retrain diagnostics and/or the drift signal), or
            ``None`` when nothing is logged.
        """
        if not self._agent_config.retrain_features or epoch == self._training_config.n_epochs:
            return None

        if self._agent_config.retrain_trigger == RetrainTrigger.DRIFT:
            return self._drift_triggered_retrain(epoch)

        # INTERVAL: fixed-schedule retrain (byte-for-byte the legacy behavior).
        if epoch % self._agent_config.retrain_interval == 0:
            return self._do_retrain(epoch)
        return None
