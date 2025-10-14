from dataclasses import dataclass
from typing import Dict


@dataclass
class WandbConfig:
    project: str
    entity: str
    job_type: str
    group: str
    name: str
    config: Dict
