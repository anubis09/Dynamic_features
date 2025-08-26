from pathlib import Path

_current_dir = Path(__file__).parent
_configs_dir = _current_dir / "configs"

# Alternative approach: Config paths as constants
ALGORITHMS_CONFIG_PATH = _configs_dir / "implementations"
GAME_ENV_CONFIG_PATH = _configs_dir / "game_env_conf.yaml"
WANDB_CONFIG_PATH = _configs_dir / "wandb_conf.yaml"


__all__ = [
    "ALGORITHMS_CONFIG_PATH",
    "GAME_ENV_CONFIG_PATH",
    "WANDB_CONFIG_PATH",
]
