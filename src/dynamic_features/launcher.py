import hydra
from omegaconf import DictConfig

from dynamic_features import ALGORITHMS_CONFIG_PATH
from dynamic_features.implementations.DQN import DQN_exp


@hydra.main(version_base=None, config_path=str(ALGORITHMS_CONFIG_PATH), config_name="")
def experiment(
    cfg: DictConfig,
):
    dqn_exp_class = DQN_exp(**cfg.environment)

    dqn_exp_class.create_policy(**cfg.policy)

    dqn_exp_class.create_agent(**cfg.agent)

    dqn_exp_class.create_core()
    
    # implement the parser here.
    wandb_log = kwargs.get("wandb_log", False)

    if wandb_log:
        dqn_exp_class.update_wandb_conf(
            name=f"seed_{cfg.environment.seed}",
            job_type=kwargs.get("wandb_job_type", ""),
            group=cfg.environment.game_name,
            config=cfg,
        )

    dqn_exp_class.train(**cfg.train, use_wandb=wandb_log)


if __name__ == "__main__":
    experiment()
