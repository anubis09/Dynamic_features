import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    dropdown = mo.ui.dropdown(
        options=["DQN_sweep", "Rainbow_sweep", "Sarsa_lambda_rec_sweep"],
        allow_select_none=False,
        value="DQN_sweep",
        label="choose sweep",
    )
    return (dropdown,)


@app.cell
def __(dropdown, mo):
    mo.vstack([mo.hstack([dropdown, mo.md(f"Has value: {dropdown.value}")])])
    return


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
def __(ENTITY, PROJECT, api, dropdown):
    runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"jobType": dropdown.value}, lazy=False)

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
            "env_name": env_name,
            "average_return": run.summary[f"{env_name}/average_return"],
        }
        info.append(run_info)

    df = pl.DataFrame(info)
    configs = df.drop_in_place("config").struct.unnest().drop("env").unique()
    df.head()
    return configs, df, env_name, info, run, run_info


@app.cell
def __(df):
    df["sweep_id"].unique().count()
    return


@app.cell
def __(df):
    df["env_name"].unique().count()
    return


@app.cell
def __(df, pl):
    grouped_averaged_df = df.group_by("env_name", "sweep_id").agg(pl.mean("average_return"))
    grouped_averaged_df.head()
    return (grouped_averaged_df,)


@app.cell
def __(grouped_averaged_df):
    len(grouped_averaged_df)
    return


@app.cell
def __(grouped_averaged_df, pl):
    cdf_df = grouped_averaged_df.with_columns(
        cdf_score=pl.col("average_return").rank(method="ordinal").over("env_name")
        / pl.col("average_return").count().over("env_name")
    )
    return (cdf_df,)


@app.cell
def __(cdf_df, pl):
    best_sweeps_across_env = (
        cdf_df.group_by("sweep_id")
        .agg(pl.mean("cdf_score").alias("mean_cdf_score"))
        .sort("mean_cdf_score", descending=True)
    )
    best_sweeps_across_env
    return (best_sweeps_across_env,)


@app.cell
def __(best_sweeps_across_env):
    best_sweep = best_sweeps_across_env.head(1)["sweep_id"].item()
    best_sweep
    return (best_sweep,)


@app.cell
def __(best_sweep, configs, pl):
    best_config = configs.filter(pl.col("sweep_id") == best_sweep)
    best_config
    return (best_config,)


if __name__ == "__main__":
    app.run()
