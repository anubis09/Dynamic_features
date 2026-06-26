import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __():
    JOB_TYPES = ["DQN_sweep", "Rainbow_sweep", "Sarsa_lambda_rec_sweep", "sarsa_lambda_dt_retr_sweep", "random_sweep"]
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
    import wandb

    api = wandb.Api()
    return api, wandb


@app.cell
def __():
    ENTITY = "Dynamic_Tiles_Thesis"
    PROJECT = "Dynamic_Features"
    return ENTITY, PROJECT


@app.cell
def __(ENTITY, JOB_TYPES, PROJECT, api):
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"jobType": {"$in": JOB_TYPES}}, lazy=False)

    print(len(runs))
    return (runs,)


@app.cell
def __():
    import polars as pl

    return (pl,)


@app.cell
def __(pl, runs):
    info = []
    for run in runs:
        env_name = run.config["env"]["name"]
        run_info = {
            "config": run.config,
            "sweep_id": run.config["sweep_id"],
            "job_type": run.job_type,
            "env_name": env_name,
            "average_return": run.summary[f"{env_name}/average_return"],
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
    grouped_averaged_df = df.group_by("env_name", "sweep_id").agg(pl.mean("average_return"), pl.first("job_type"))
    grouped_averaged_df.head()
    return (grouped_averaged_df,)


@app.cell
def __(grouped_averaged_df):
    len(grouped_averaged_df)
    return


@app.cell
def __(grouped_averaged_df, pl):
    cdf_df = grouped_averaged_df.with_columns(
        cdf_score=pl.col("average_return").rank().over("env_name") / pl.col("average_return").count().over("env_name")
    )
    return (cdf_df,)


@app.cell
def __(cdf_df, pl):
    best_sweeps_across_env_alg = (
        cdf_df.group_by("sweep_id", "job_type")
        .agg(pl.mean("cdf_score").alias("mean_cdf_score"))
        .sort("mean_cdf_score", descending=True)
    )
    best_sweeps_across_env_alg
    return (best_sweeps_across_env_alg,)


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
