import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


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
def __():
    import marimo as mo

    dropdown = mo.ui.dropdown(
        options=["sarsa_lambda_dt_plain_sweep", "sarsa_lambda_dt_pilot"],
        allow_select_none=False,
        value="sarsa_lambda_dt_pilot",
        label="choose sweep",
    )
    return dropdown, mo


@app.cell
def __(dropdown, mo):
    mo.vstack([mo.hstack([dropdown, mo.md(f"Has value: {dropdown.value}")])])
    return


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
def __():
    hyperparams = ["n_tilings", "n_clusters", "lambda_coeff", "learning_rate"]
    return (hyperparams,)


@app.cell
def __(hyperparams, pl, runs):
    info = []
    for run in runs:
        env_name = run.config["env"]["name"]
        run_info = {
            "sweep_id": run.config["sweep_id"],
            "env_name": env_name,
            "average_return": run.summary[f"{env_name}/average_return"],
            **{hyperparam: run.config["agent"][hyperparam] for hyperparam in hyperparams},
        }
        info.append(run_info)

    df = pl.DataFrame(info)
    df.head()
    return df, env_name, info, run, run_info


@app.cell
def __(df, hyperparams, pl):
    grouped_averaged_df = (
        df.group_by("env_name", "sweep_id", *hyperparams)
        .agg(pl.mean("average_return"))
        .sort("average_return", descending=True)
    )
    grouped_averaged_df.head()
    return (grouped_averaged_df,)


@app.cell
def __(grouped_averaged_df, hyperparams):
    for (env,), env_df in grouped_averaged_df.group_by("env_name"):
        k = max(1, int(env_df.height * 0.05))
        top = env_df.top_k(k, by="average_return")
        print(f"\n=== {env} (top {k} of {env_df.height} configs) ===")
        print(f"min return: {top['average_return'].min()}")
        print(f"max return:{top['average_return'].max()}")
        for hp in hyperparams:
            print(f"\n{hp}:")
            print(top[hp].value_counts().sort(hp))
    return env, env_df, hp, k, top


@app.cell
def __(grouped_averaged_df, pl):
    pair_a = "n_tilings"  # change to your column
    pair_b = "n_clusters"  # change to your column

    pair_results = {}

    for (pair_env,), pair_env_df in grouped_averaged_df.group_by("env_name"):
        pair_k = max(1, int(pair_env_df.height * 0.05))
        pair_top = pair_env_df.top_k(pair_k, by="average_return")

        # count of good (top-k) configs per (pair_a, pair_b) cell
        pair_good_counts = pair_top.group_by([pair_a, pair_b]).agg(pl.len().alias("good_count"))

        # marginal mean return over ALL configs per cell
        pair_cell_stats = pair_env_df.group_by([pair_a, pair_b]).agg(
            pl.col("average_return").mean().alias("cell_mean"),
            pl.len().alias("cell_n"),
        )

        pair_table = (
            pair_cell_stats.join(pair_good_counts, on=[pair_a, pair_b], how="left")
            .with_columns(pl.col("good_count").fill_null(0))
            .sort([pair_a, pair_b])
        )
        pair_results[pair_env] = pair_table
        print(f"\n=== {pair_env} (top {pair_k} of {pair_env_df.height}) ===")
        print(pair_table)
    return (
        pair_a,
        pair_b,
        pair_cell_stats,
        pair_env,
        pair_env_df,
        pair_good_counts,
        pair_k,
        pair_results,
        pair_table,
        pair_top,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
