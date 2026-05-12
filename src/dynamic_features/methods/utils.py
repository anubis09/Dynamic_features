from typing import Union

import numpy as np
from mushroom_rl.core import Core, Environment
from mushroom_rl.utils.dataset import parse_dataset


def get_states_from_mdp(mdp: Environment, n_states: int, core: Union[Core, None] = None) -> np.ndarray:
    """Samples random states from an environment.

    Collects states by running episodes in the environment. If no core is
    provided, the environment is stepped through directly using a uniform
    random policy. Otherwise, the provided core's current policy is used
    to generate the trajectories.

    Args:
        mdp: The environment to sample states from.
        n_states: The number of states to return.
        core: A pre-configured Core instance with an agent and environment.
            If None, random actions are sampled directly from the action
            space. Requires a discrete 1D action space (``Discrete``);
            ``Box`` or multi-dimensional action spaces are not supported
            in the random-policy path.

    Returns:
        An array of shape (n_states, state_dim) containing the sampled states.
    """
    if core is None:
        state_dim = mdp.info.observation_space.shape[0]
        dtype = mdp.info.observation_space.high.dtype
        states = np.empty((n_states, state_dim), dtype=dtype)
        actions = np.random.randint(mdp.info.action_space.n, size=n_states)
        mdp.reset()
        for i in range(n_states):
            state, _, done, _ = mdp.step([actions[i]])
            states[i] = state
            if done:
                mdp.reset()
        return states
    else:
        # The dataset is a list of tuples all containing:
        # 0. the previous state,
        # 1. the action sampled by the agent,
        # 2. the reward obtained,
        # 3. the reached state,
        # 4. the absorbing flag of the reached state
        # 5. and the last step flag.
        dataset = core.evaluate(n_steps=n_states, render=False)
        # Given a dataset, this function groups together in a list all the elements of it.
        # It returns list of:
        # 0. states
        # 1. actions
        # 2. rewards
        # 3. next_states
        # 4. absorbing
        # 5. lasts
        #
        # So we just return the [0] element
        return parse_dataset(dataset)[0]
