import hydra
from omegaconf import DictConfig, OmegaConf

from dynamic_features import CONFIGS_PATH
from dynamic_features.algorithms.dqn import DQN


@hydra.main(version_base=None, config_path=str(CONFIGS_PATH), config_name="config")
def experiment(
    cfg: DictConfig,
):
    dqn_exp_class = DQN(seed=cfg.seed)

    dqn_exp_class.create_mdp(mdp_name=cfg.environment.name, **cfg.environment.core)

    dqn_exp_class.create_policy(**cfg.algorithm.policy)

    dqn_exp_class.create_agent(**cfg.algorithm.agent)

    dqn_exp_class.create_core()

    use_wandb = cfg.wandb_log

    if use_wandb:
        # this serves for having a good looking logging on wandb
        env_container = OmegaConf.to_container(cfg.environment, resolve=True)
        alg_container = OmegaConf.to_container(cfg.algorithm, resolve=True)

        env_dict = env_container if isinstance(env_container, dict) else {}
        alg_dict = alg_container if isinstance(alg_container, dict) else {}

        config = {**env_dict, **alg_dict}
        dqn_exp_class.update_wandb_conf(
            **cfg.wandb,
            config=config,
        )

    dqn_exp_class.train(**cfg.algorithm.train, use_wandb=use_wandb)


if __name__ == "__main__":
    experiment()
