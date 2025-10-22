from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WandbConfig:
    project: str
    entity: str
    job_type: str
    group: str
    name: str
    config: Dict[str, Any]
