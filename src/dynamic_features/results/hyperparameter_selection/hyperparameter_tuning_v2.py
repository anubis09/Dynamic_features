import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    JOB_TYPES = [
        "DQN_sweep",
        "Rainbow_sweep",
        "Sarsa_lambda_rec_sweep",
        "sl_dt_noretr_sweep",
        "sarsa_lambda_dt_retr_sweep",
        "random_sweep",
    ]
    return (JOB_TYPES,)


@app.cell
def __(mo):
    mo.md(
        r"""Now that the algorithm for which we are going to perform the hyperparameter analysis has been chosen, we can start the analysis."""
    )
    return


@app.cell
def __(mo):
    mo.md(r"""# Analysis""")
    return


@app.cell
def __():
    from dynamic_features.results.wandb_common import env_name_of, get_runs, mean_metric

    return env_name_of, get_runs, mean_metric


@app.cell
def __(JOB_TYPES, get_runs):
    runs = get_runs(JOB_TYPES)
    print(len(runs))
    return (runs,)


@app.cell
def __():
    import polars as pl

    return (pl,)


@app.cell
def __(env_name_of, mean_metric, pl, runs):
    info = []
    for run in runs:
        env_name = env_name_of(run)
        run_info = {
            "config": run.config,
            "sweep_id": run.config["sweep_id"],
            "job_type": run.job_type,
            "env_name": env_name,
            "average_return": mean_metric(run, f"{env_name}/average_return"),
        }
        info.append(run_info)
    df = pl.DataFrame(info, infer_schema_length=None)
    configs = df.drop_in_place("config").struct.unnest().drop("env").unique()
    df.head()
    return configs, df, env_name, info, run, run_info


@app.cell
def __(JOB_TYPES, df):
    n_hyperparameter_configs = 48
    assert (
        df["sweep_id"].unique().count() == n_hyperparameter_configs * (len(JOB_TYPES) - 1) + 1
    )  # +1 is the random sweep.
    return (n_hyperparameter_configs,)


@app.cell
def __(df):
    df["env_name"].unique().count()
    return


@app.cell
def __(df, pl):
    cdf_df = df.with_columns(
        cdf_score=(pl.col("average_return").rank(method="min").over("env_name") - 1)
        / pl.col("average_return").count().over("env_name")
    )
    best_sweeps_across_env_alg = (
        cdf_df.group_by("sweep_id", "job_type")
        .agg(pl.mean("cdf_score").alias("mean_cdf_score"))
        .sort("mean_cdf_score", descending=True)
    )
    return best_sweeps_across_env_alg, cdf_df


@app.cell
def __(best_sweeps_across_env_alg):
    best_sweeps_across_env_alg
    return


@app.cell
def __(best_sweeps_across_env_alg):
    top_sweep_per_job_type = (
        best_sweeps_across_env_alg.sort("mean_cdf_score", descending=True).group_by("job_type").head(1)
    )
    top_sweep_per_job_type
    return (top_sweep_per_job_type,)


@app.cell
def __(configs, pl, top_sweep_per_job_type):
    for row in top_sweep_per_job_type.iter_rows(named=True):
        best_config = configs.filter(pl.col("sweep_id") == row["sweep_id"])
        print("-----")
        print(f"job_type: {row['job_type']}")
        print(best_config.to_dicts()[0])
    return best_config, row


if __name__ == "__main__":
    app.run()
