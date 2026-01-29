from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EnvCore:
    gamma: float
    horizon: Optional[int]


@dataclass
class RainbowParams:
    v_min: int
    v_max: int


@dataclass
class EnvConfig:
    name: str
    budget: int
    core: EnvCore
    rainbow: RainbowParams


@dataclass
class MountainCarEnvConfig(EnvConfig):
    name: str = "Gym.MountainCar-v0"
    budget: int = 600_000
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=None))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=-200, v_max=0))


@dataclass
class AcrobotEnvConfig(EnvConfig):
    name: str = "Gym.Acrobot-v1"
    budget: int = 600_000
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=None))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=-500, v_max=0))


@dataclass
class CartPoleEnvConfig(EnvConfig):
    name: str = "Gym.CartPole-v1"
    budget: int = 600_000
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=None))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=0, v_max=500))


@dataclass
class LunarLanderEnvConfig(EnvConfig):
    name: str = "Gym.LunarLander-v2"
    budget: int = 1_000_000
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=None))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=-200, v_max=300))


@dataclass
class PuddleWorldEnvConfig(EnvConfig):
    name: str = "PuddleWorld"
    budget: int = 600_000
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=500))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=-500, v_max=0))
