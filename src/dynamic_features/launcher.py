import hydra
from hydra.utils import instantiate
from omegaconf.omegaconf import DictConfig

from dynamic_features import CONFIGS_PATH
from dynamic_features.configs import register_configs

# register configs is needed to load hydra config store
register_configs()


@hydra.main(version_base=None, config_path=str(CONFIGS_PATH), config_name="config")
def experiment(
    cfg: DictConfig,
) -> None:
    # instantiate allows us to create an instance cfg.algorithm._target_ class.
    # _convert_ is used to convert the DictConfig to the backing structured object.
    # the other parameters are input for the __init__ method.
    exp_class = instantiate(cfg.algorithm, env=cfg.environment, wandb=cfg.wandb, seed=cfg.seed, _convert_="object")

    exp_class.train(use_wandb=cfg.wandb_log)


if __name__ == "__main__":
    experiment()
