"""
Download wandb runs for a set of job types and environments, and plot the
"{env_name}/average_return" metric (logged every epoch) aggregated across
seeds: mean line with a standard-error shaded band, one figure per
environment, saved as SVG for direct inclusion in the thesis.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import polars as pl

from dynamic_features.results.wandb_common import (
    empty_frame,
    env_name_of,
    fetch_history,
    get_runs,
    is_valid_value,
    matches_env_filters,
    short_env_name,
)

# job_type -> legend label
JOB_TYPES: Dict[str, str] = {
    "nwt_ridge_an_batched_drift": "Dynamic tiling",
}
# substrings matched against run.config["env"]["name"]
ENV_FILTERS = [
    # "MountainCar",
    # "PuddleWorld",
    # "CartPole",
    # "Acrobot",
    "LunarLander",
]
# restrict to specific run.config["sweep_id"] values, or None for no filter
SWEEP_IDS: Optional[List[str]] = "4f64adf1be"

OUT_DIR = Path(__file__).parent / "plots"

# Color per environment (Okabe-Ito colorblind-safe palette by default —
# override any entry to pick a specific color for that env).
ENV_COLORS: Dict[str, str] = {
    "MountainCar": "#0072B2",
    "PuddleWorld": "#E69F00",
    "CartPole": "#009E73",
    "Acrobot": "#D55E00",
    "LunarLander": "#CC79A7",
}
FALLBACK_COLORS = ["#56B4E9", "#F0E442", "#000000"]

# Line style per job_type, used to tell overlaid job_types apart within the
# same (single-color) env figure.
LINESTYLES = ["-", "--", "-.", ":"]

FONT_NAME = "Atkinson Hyperlegible Next"
FONT_DIR = Path.home() / "Library" / "Fonts"


def _register_font():
    """Register the Atkinson Hyperlegible Next font files with matplotlib
    and make it the default font family for all plots."""
    for weight_file in [
        "AtkinsonHyperlegibleNext-Regular.otf",
        "AtkinsonHyperlegibleNext-Bold.otf",
        "AtkinsonHyperlegibleNext-RegularItalic.otf",
        "AtkinsonHyperlegibleNext-BoldItalic.otf",
    ]:
        path = FONT_DIR / weight_file
        if path.exists():
            fm.fontManager.addfont(str(path))

    plt.rcParams.update(
        {
            "font.family": FONT_NAME,
            "svg.fonttype": "none",  # keep text as text (editable) in the SVG
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "axes.axisbelow": True,
        }
    )


def fetch_return_df(
    runs,
    job_types: List[str],
    env_filters: List[str],
    sweep_ids: Optional[List[str]] = None,
) -> pl.DataFrame:
    """
    For each run matching job_types/env_filters/sweep_ids, fetch the
    per-epoch history of "{env_name}/average_return" (env_name taken from
    the run's own config, since it encodes the exact gym id, e.g.
    "Gym.MountainCar-v0").
    Returns a long-form DataFrame: job_type, run_id, env_name, step, value.
    """
    records = []
    for run in runs:
        job_type = run.job_type
        if job_type not in job_types:
            continue

        if sweep_ids is not None and run.config.get("sweep_id") not in sweep_ids:
            continue

        env_name = env_name_of(run)
        if not matches_env_filters(env_name, env_filters):
            continue

        metric = f"{env_name}/average_return"
        history = fetch_history(run, [metric])

        for row in history:
            value = row.get(metric)
            step = row.get("_step")
            if step is None or not is_valid_value(value):
                continue
            records.append(
                {
                    "job_type": job_type,
                    "run_id": run.id,
                    "env_name": env_name,
                    "step": int(step),
                    "value": float(value),
                }
            )

    schema = {
        "job_type": pl.Utf8,
        "run_id": pl.Utf8,
        "env_name": pl.Utf8,
        "step": pl.Int64,
        "value": pl.Float64,
    }
    return pl.DataFrame(records) if records else empty_frame(schema)


def aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """Mean and standard error of the mean across seeds, per epoch."""
    return (
        df.group_by("job_type", "env_name", "step")
        .agg(
            [
                pl.mean("value").alias("mean"),
                pl.std("value").alias("std"),
                pl.col("value").count().alias("n_runs"),
            ]
        )
        .with_columns((pl.col("std") / pl.col("n_runs").sqrt()).fill_null(0.0).alias("sem"))
        .sort("job_type", "env_name", "step")
    )


def plot_env(
    agg_df: pl.DataFrame,
    env_name: str,
    job_types: Dict[str, str],
    env_filters: List[str],
):
    env_df = agg_df.filter(pl.col("env_name") == env_name)
    short_name = short_env_name(env_name, env_filters)
    color = ENV_COLORS.get(short_name, FALLBACK_COLORS[hash(short_name) % len(FALLBACK_COLORS)])

    fig, ax = plt.subplots(figsize=(6, 4))

    n_runs_seen = set()
    for i, (job_type, label) in enumerate(job_types.items()):
        jt_df = env_df.filter(pl.col("job_type") == job_type).sort("step")
        if jt_df.is_empty():
            continue
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        x = jt_df["step"].to_list()
        y = jt_df["mean"].to_list()
        sem = jt_df["sem"].to_list()
        n_runs_seen.update(jt_df["n_runs"].to_list())

        ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.8, label=label)
        ax.fill_between(
            x,
            [m - e for m, e in zip(y, sem)],
            [m + e for m, e in zip(y, sem)],
            color=color,
            alpha=0.4,
            linewidth=0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average return")
    ax.set_title(short_name)

    if len(job_types) > 1:
        ax.legend(frameon=False, loc="best")

    n_runs_note = "/".join(str(n) for n in sorted(n_runs_seen)) if n_runs_seen else "?"
    ax.text(
        0.99,
        0.02,
        f"shaded: ± s.e.m. (n={n_runs_note} seeds)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
    )

    fig.tight_layout()
    return fig


def main():
    _register_font()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = get_runs(JOB_TYPES.keys())
    print(f"Found {len(runs)} candidate runs (before env filtering)")

    df = fetch_return_df(runs, list(JOB_TYPES.keys()), ENV_FILTERS, SWEEP_IDS)
    print(f"Loaded {len(df)} observations | {df['run_id'].n_unique()} runs | {df['env_name'].n_unique()} envs")

    agg_df = aggregate(df)

    env_names = sorted(agg_df["env_name"].unique().to_list())
    for env_name in env_names:
        fig = plot_env(agg_df, env_name, JOB_TYPES, ENV_FILTERS)
        fname = OUT_DIR / f"average_return_{short_env_name(env_name, ENV_FILTERS)}.svg"
        fig.savefig(fname, format="svg", bbox_inches="tight")
        print(f"Saved {fname}")
        plt.close(fig)


if __name__ == "__main__":
    main()
