"""
Shared helpers for fetching and shaping W&B run data, reused by the
results/analysis scripts (average_return_plot, metric_summary, pilot_study)
and by the hyperparameter_tuning marimo notebook.
"""

import math
from typing import Iterable, List, Optional

import polars as pl
import wandb

ENTITY = "Dynamic_Tiles_Thesis"
PROJECT = "Dynamic_Features"


def get_runs(job_types: Iterable[str], api: Optional[wandb.Api] = None):
    """Fetch all runs in the project whose job_type is one of job_types."""
    api = api or wandb.Api()
    return api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"jobType": {"$in": list(job_types)}},
        lazy=False,
    )


def env_name_of(run) -> str:
    """Env name from a run's own config (e.g. "Gym.MountainCar-v0"), since it
    encodes the exact gym id."""
    env_cfg = run.config.get("env", {})
    return env_cfg.get("name", "unknown") if isinstance(env_cfg, dict) else str(env_cfg)


def matches_env_filters(env_name: str, env_filters: Iterable[str]) -> bool:
    return any(env_filter in env_name for env_filter in env_filters)


def short_env_name(env_name: str, env_filters: Iterable[str]) -> str:
    """Map a full env name back to the short filter substring that matched it
    (e.g. "Gym.MountainCar-v0" -> "MountainCar")."""
    for env_filter in env_filters:
        if env_filter in env_name:
            return env_filter
    return env_name


def fetch_history(run, metrics: List[str]) -> List[dict]:
    """A run's history for the given metric keys, as a list of dict rows."""
    history = run.history(keys=metrics, pandas=False)
    if not hasattr(history, "__len__"):
        history = list(history)
    return history


def is_valid_value(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def mean_metric(run, metric: str) -> Optional[float]:
    """Mean of a metric over a run's whole training history (as opposed to
    run.summary[metric], which is just the last logged value)."""
    values = [row[metric] for row in fetch_history(run, [metric]) if metric in row and is_valid_value(row[metric])]
    return sum(values) / len(values) if values else None


def empty_frame(schema: dict) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)
