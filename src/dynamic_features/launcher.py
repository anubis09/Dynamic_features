import hashlib
import json

import hydra
from hydra.utils import instantiate
from omegaconf.omegaconf import DictConfig, OmegaConf

from dynamic_features import CONFIGS_PATH
from dynamic_features.configs import register_configs

# register configs is needed to load hydra config store
register_configs()


def dict_hash(cfg: DictConfig, hash_length: int = 10) -> str:
    """
    Compute a stable hash of a Hydra config.

    Args:
        cfg: The config to hash (typically cfg.algorithm)
        hash_length: Length of the returned hex digest (default: 10)

    Returns:
        Hex string hash of the config

    Note:
        Works with dict, list, or primitive types returned by to_container.
        json.dumps handles all JSON-serializable types consistently.
    """
    # Convert to container, resolving interpolations and converting enums
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True, throw_on_missing=True)

    # Create stable JSON representation and hash it
    # json.dumps handles dict, list, str, int, float, bool, None
    encoded = json.dumps(container, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()[:10]


@hydra.main(version_base=None, config_path=str(CONFIGS_PATH), config_name="config")
def experiment(
    cfg: DictConfig,
) -> None:
    # instantiate allows us to create an instance cfg.algorithm._target_ class.
    # _convert_ is used to convert the DictConfig to the backing structured object.
    # the other parameters are input for the __init__ method.
    hash = dict_hash(cfg.algorithm)
    cfg.wandb.config["sweep_id"] = hash

    exp_class = instantiate(cfg.algorithm, env=cfg.environment, wandb=cfg.wandb, seed=cfg.seed, _convert_="object")

    exp_class.train(use_wandb=cfg.wandb_log)


if __name__ == "__main__":
    experiment()
