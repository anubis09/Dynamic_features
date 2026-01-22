from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class WandbMode(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"


@dataclass
class WandbConfig:
    project: str
    entity: str
    job_type: str
    group: str
    mode: WandbMode
    name: str
    config: Dict[str, Any]
