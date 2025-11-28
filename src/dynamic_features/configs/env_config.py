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
    core: EnvCore
    rainbow: RainbowParams


@dataclass
class MountainCarEnvConfig(EnvConfig):
    name: str = "MountainCar-v0"
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=None))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=-11_000, v_max=-1))


@dataclass
class LunarLanderEnvConfig(EnvConfig):
    name: str = "LunarLander-v2"
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=1000))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=-1_000, v_max=300))


@dataclass
class AcrobotEnvConfig(EnvConfig):
    name: str = "Acrobot-v1"
    core: EnvCore = field(default_factory=lambda: EnvCore(gamma=0.99, horizon=None))
    rainbow: RainbowParams = field(default_factory=lambda: RainbowParams(v_min=-500, v_max=1))
