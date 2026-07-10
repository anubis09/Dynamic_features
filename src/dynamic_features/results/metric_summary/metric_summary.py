"""
Download wandb runs for a set of job types and environments, and print
per (job_type, env) summary statistics (mean/std) for a fixed set of metrics.

Unlike drift_transfer_analysis.py, this script does NOT assume that the
occurrence index (0th, 1st, ... logged value) lines up across runs at the
same timestep — it only looks at:
    - the first logged value per run (occurrence_idx == 0)
    - the last logged value per run (occurrence_idx == max for that run)
    - all logged values per run (overall)
and aggregates mean/std of those across the ~15 runs in each (job_type, env)
combination. No plots, just printed tables.
"""

from typing import List

import polars as pl

from dynamic_features.results.wandb_common import (
    empty_frame,
    env_name_of,
    fetch_history,
    get_runs,
    is_valid_value,
    matches_env_filters,
)

JOB_TYPES = [
    "nwt_ridge_an_batched_drift",
]
ENV_FILTERS = [
    "MountainCar",
    "Acrobot",
    "LunarLander",
]
N_CLUSTERS = 48
METRICS = [
    "transfer/rMAE_test",
    "transfer/preserved_greedy_actions",
    "transfer/cluster_ari",
    "transfer/weight_norm_ratio",
    "approximator/accuracy",
]


def fetch_occurrence_df(runs, job_types: List[str], env_filters: List[str], n_clusters: int, metrics: List[str]):
    """
    For each run, fetch the history of the given metrics and assign an
    occurrence index (0, 1, 2, …) to each logged value per metric.
    Returns:
        - a long-form DataFrame with columns:
            job_type, run_id, env_name, metric, occurrence_idx, value
        - a dict {(job_type, env_name): set(run_id)} of every run that
          matched the job_type/env/n_clusters filters, regardless of
          whether it ever logged the metrics (e.g. a drift-triggered run
          that never drifted has zero rows but still matched the filters).
    """
    records = []
    matched_run_ids = {}
    for run in runs:
        job_type = run.job_type
        if job_type not in job_types:
            continue

        env_name = env_name_of(run)
        if not matches_env_filters(env_name, env_filters):
            continue

        if run.config.get("agent", {}).get("n_clusters") != n_clusters:
            continue

        matched_run_ids.setdefault((job_type, env_name), set()).add(run.id)

        history = fetch_history(run, metrics)

        for metric in metrics:
            rows_with_metric = [row for row in history if metric in row and is_valid_value(row[metric])]
            for occ_idx, row in enumerate(rows_with_metric):
                records.append(
                    {
                        "job_type": job_type,
                        "run_id": run.id,
                        "env_name": env_name,
                        "metric": metric,
                        "occurrence_idx": occ_idx,
                        "value": float(row[metric]),
                    }
                )

    schema = {
        "job_type": pl.Utf8,
        "run_id": pl.Utf8,
        "env_name": pl.Utf8,
        "metric": pl.Utf8,
        "occurrence_idx": pl.Int64,
        "value": pl.Float64,
    }
    df = pl.DataFrame(records) if records else empty_frame(schema)
    return df, matched_run_ids


def summarize(df: pl.DataFrame) -> pl.DataFrame:
    """
    For each (job_type, env_name, metric), compute mean/std across runs of:
        - the first logged value per run
        - the last logged value per run
        - all logged values per run (overall, per-run-then-across-runs)
        - all logged points pooled together (overall_pooled, across every
          reading of every run — larger n, smaller std since it also
          captures within-run variance over time)
    """
    last_idx = df.group_by("job_type", "env_name", "metric", "run_id").agg(
        pl.col("occurrence_idx").max().alias("last_idx")
    )

    first_vals = (
        df.filter(pl.col("occurrence_idx") == 0)
        .group_by("job_type", "env_name", "metric")
        .agg(
            [
                pl.mean("value").alias("first_mean"),
                pl.std("value").alias("first_std"),
                pl.col("value").count().alias("first_n"),
            ]
        )
    )

    last_vals = (
        df.join(last_idx, on=["job_type", "env_name", "metric", "run_id"])
        .filter(pl.col("occurrence_idx") == pl.col("last_idx"))
        .group_by("job_type", "env_name", "metric")
        .agg(
            [
                pl.mean("value").alias("last_mean"),
                pl.std("value").alias("last_std"),
                pl.col("value").count().alias("last_n"),
            ]
        )
    )

    # Per-run overall mean first (averaging that run's own trajectory), then
    # mean/std of those per-run means across runs — so overall_n matches
    # n_runs, same as first_vals/last_vals, instead of pooling every logged
    # point (which conflates within-run variance-over-time with
    # across-run variance and inflates n).
    per_run_overall = df.group_by("job_type", "env_name", "metric", "run_id").agg(pl.mean("value").alias("run_mean"))

    overall_vals = per_run_overall.group_by("job_type", "env_name", "metric").agg(
        [
            pl.mean("run_mean").alias("overall_mean"),
            pl.std("run_mean").alias("overall_std"),
            pl.col("run_mean").count().alias("overall_n"),
            pl.col("run_id").n_unique().alias("n_runs"),
        ]
    )

    overall_pooled_vals = df.group_by("job_type", "env_name", "metric").agg(
        [
            pl.mean("value").alias("overall_pooled_mean"),
            pl.std("value").alias("overall_pooled_std"),
            pl.col("value").count().alias("overall_pooled_n"),
        ]
    )

    return (
        overall_vals.join(first_vals, on=["job_type", "env_name", "metric"], how="left")
        .join(last_vals, on=["job_type", "env_name", "metric"], how="left")
        .join(overall_pooled_vals, on=["job_type", "env_name", "metric"], how="left")
        .sort("job_type", "env_name", "metric")
    )


def compute_retrain_counts(df: pl.DataFrame, matched_run_ids: dict, reference_metric: str) -> pl.DataFrame:
    """
    Number of retrains for a run = number of times it logged
    `reference_metric` (metrics are logged once per retrain trigger).
    Runs that matched the filters but never triggered a retrain (and thus
    never logged anything) are zero-filled rather than dropped, so the
    average is taken across all seeds, not just the ones that drifted.
    """
    all_runs = pl.DataFrame(
        [
            {"job_type": job_type, "env_name": env_name, "run_id": run_id}
            for (job_type, env_name), run_ids in matched_run_ids.items()
            for run_id in run_ids
        ],
        schema={"job_type": pl.Utf8, "env_name": pl.Utf8, "run_id": pl.Utf8},
    )

    per_run_counts = (
        df.filter(pl.col("metric") == reference_metric)
        .group_by("job_type", "env_name", "run_id")
        .agg(pl.col("occurrence_idx").count().alias("n_retrains"))
    )

    return (
        all_runs.join(per_run_counts, on=["job_type", "env_name", "run_id"], how="left")
        .with_columns(pl.col("n_retrains").fill_null(0))
        .group_by("job_type", "env_name")
        .agg(
            [
                pl.mean("n_retrains").alias("mean_retrains"),
                pl.std("n_retrains").alias("std_retrains"),
                pl.min("n_retrains").alias("min_retrains"),
                pl.max("n_retrains").alias("max_retrains"),
                pl.col("run_id").n_unique().alias("n_runs"),
            ]
        )
        .sort("job_type", "env_name")
    )


def print_retrain_counts(retrain_df: pl.DataFrame):
    print("\n=== Retrains per run (averaged across all seeds, incl. 0-retrain runs) ===")
    for row in retrain_df.iter_rows(named=True):
        print(
            f"  job_type={row['job_type']} | env={row['env_name']}: "
            f"mean={row['mean_retrains']:.2f}  std={row['std_retrains']:.2f}  "
            f"min={row['min_retrains']}  max={row['max_retrains']}  (n_runs={row['n_runs']})"
        )


def print_summary(summary_df: pl.DataFrame, matched_counts: dict):
    for (job_type, env_name), group in summary_df.group_by(["job_type", "env_name"], maintain_order=True):
        n_matched = matched_counts.get((job_type, env_name), "?")
        print(f"\n=== job_type={job_type} | env={env_name} | matched_runs={n_matched} ===")
        for row in group.sort("metric").iter_rows(named=True):
            print(f"  {row['metric']} (n_runs_with_data={row['n_runs']}/{n_matched}):")
            print(
                f"    first:           mean={row['first_mean']:.4f}  std={row['first_std']:.4f}  (n={row['first_n']})"
            )
            print(f"    last:            mean={row['last_mean']:.4f}  std={row['last_std']:.4f}  (n={row['last_n']})")
            print(
                f"    overall:         mean={row['overall_mean']:.4f}  std={row['overall_std']:.4f}  (n={row['overall_n']})"
            )


def main():
    runs = get_runs(JOB_TYPES)
    print(f"Found {len(runs)} candidate runs (before env/n_clusters filtering)")

    df, matched_run_ids = fetch_occurrence_df(runs, JOB_TYPES, ENV_FILTERS, N_CLUSTERS, METRICS)
    matched_counts = {key: len(ids) for key, ids in matched_run_ids.items()}
    print(
        f"\nLoaded {len(df)} observations | "
        f"{df['run_id'].n_unique()} runs with metric data | "
        f"{df['job_type'].n_unique()} job types | "
        f"{df['env_name'].n_unique()} envs"
    )
    for (job_type, env_name), n_matched in sorted(matched_counts.items()):
        print(f"  matched (job_type={job_type}, env={env_name}): {n_matched} runs total")

    retrain_df = compute_retrain_counts(df, matched_run_ids, reference_metric=METRICS[0])
    print_retrain_counts(retrain_df)

    summary_df = summarize(df)
    print_summary(summary_df, matched_counts)


if __name__ == "__main__":
    main()
