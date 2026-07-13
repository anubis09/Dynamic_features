import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Bootstrap statistics for the thesis — v2

        Downloads the final runs from W&B for the chosen job types, then bootstraps
        (`BOOTSTRAP_DIM` resamples) to produce, with 95% confidence intervals:

        - average CDF across environments (per job type)
        - average CDF per environment (per job type)
        - mean return during training (per environment, per job type)
        - median return during training (per environment, per job type)

        **v2 difference:** the per-run scalar used for the CDF is the **mean of the
        `{env}/average_return` history over the whole training** (not the run-summary value,
        which is just the last logged epoch). The run summary is not used at all.

        The CDF follows the `rank / count` normalization applied to **individual runs**: per
        environment, all runs (across job types) are pooled, each run gets its normalized rank
        (`rank / pool size` = empirical CDF position), and only *after* that are the scores
        aggregated (averaged) per job type. This is the probability-of-superiority /
        common-language effect size. There is no mean-based variant in this notebook.

        Every job type keeps a **fixed color** and a **fixed order** across all plots. All figures
        are saved as SVG in the `plots_v2/` folder next to this notebook (separate from v1's
        `plots/`, so the two notebooks never overwrite each other's figures).
        """
    )
    return


@app.cell
def __(mo):
    mo.md(r"""## Configuration (edit here)""")
    return


@app.cell
def __():
    # ---- Job types to download (edit this list) -------------------------------
    JOB_TYPES = [
        "DQN_final",
        "rainbow_final_v2",
        "sarsa_lambda_rec_final_v2",
        "sl_dt_noretr_final_v2",
        "sarsa_lambda_dt_final_v2",
        "random_final",
    ]

    # Fixed left-to-right order used for every bar plot.
    JOB_TYPE_ORDER = [
        "DQN_final",
        "rainbow_final_v2",
        "sarsa_lambda_rec_final_v2",
        "sl_dt_noretr_final_v2",
        "sarsa_lambda_dt_final_v2",
        "random_final",
    ]

    # Human-readable labels shown on the plots.
    JOB_TYPE_LABELS = {
        "DQN_final": "DQN",
        "rainbow_final_v2": "Rainbow",
        "sarsa_lambda_rec_final_v2": "Sarsa(λ)-Rec_Tiles",
        "sl_dt_noretr_final_v2": "Sarsa(λ)-Clusters",
        "sarsa_lambda_dt_final_v2": "Sarsa(λ)-Clusters-R",
        "random_final": "Random",
    }

    # ---- Accessible color palette to pick from --------------------------------
    # Okabe-Ito colorblind-safe palette. Pick colors from here and assign them to
    # each job type below; the assigned color is reused in every plot.
    ACCESSIBLE_COLORS = {
        "orange": "#E69F00",
        "sky_blue": "#56B4E9",
        "bluish_green": "#009E73",
        "yellow": "#F0E442",
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "reddish_purple": "#CC79A7",
        "black": "#000000",
        "grey": "#999999",
    }

    # ---- job_type -> color (assign one accessible color per job type) ----------
    JOB_TYPE_COLORS = {
        "DQN_final": ACCESSIBLE_COLORS["blue"],  # readable color
        "rainbow_final_v2": ACCESSIBLE_COLORS["bluish_green"],
        "sarsa_lambda_rec_final_v2": ACCESSIBLE_COLORS["orange"],
        "sl_dt_noretr_final_v2": ACCESSIBLE_COLORS["reddish_purple"],
        "sarsa_lambda_dt_final_v2": ACCESSIBLE_COLORS["yellow"],
        "random_final": ACCESSIBLE_COLORS["grey"],  # grey for the random baseline
    }

    # Substrings matched against run.config["env"]["name"], also used to order and
    # to shorten env names in titles/filenames.
    ENV_FILTERS = ["MountainCar", "PuddleWorld", "CartPole", "Acrobot", "LunarLander"]

    BOOTSTRAP_DIM = 10_000
    SEED = 0
    return (
        ACCESSIBLE_COLORS,
        BOOTSTRAP_DIM,
        ENV_FILTERS,
        JOB_TYPES,
        JOB_TYPE_COLORS,
        JOB_TYPE_LABELS,
        JOB_TYPE_ORDER,
        SEED,
    )


@app.cell
def __():
    from pathlib import Path

    import matplotlib.font_manager as fm
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    from dynamic_features.results.wandb_common import (
        env_name_of,
        fetch_history,
        get_runs,
        is_valid_value,
        short_env_name,
    )

    return (
        Path,
        env_name_of,
        fetch_history,
        fm,
        get_runs,
        is_valid_value,
        mpatches,
        np,
        pl,
        plt,
        short_env_name,
    )


@app.cell
def __(Path, __file__):
    OUT_DIR = Path(__file__).parent / "plots_v2"
    return (OUT_DIR,)


@app.cell
def __(Path, fm, plt):
    # Font / SVG styling, mirrored from average_return_plot.py so all thesis
    # figures share the same look and keep editable text in the SVG.
    FONT_NAME = "Atkinson Hyperlegible Next"
    _FONT_DIR = Path.home() / "Library" / "Fonts"

    def _register_font():
        for weight_file in [
            "AtkinsonHyperlegibleNext-Regular.otf",
            "AtkinsonHyperlegibleNext-Bold.otf",
            "AtkinsonHyperlegibleNext-RegularItalic.otf",
            "AtkinsonHyperlegibleNext-BoldItalic.otf",
        ]:
            path = _FONT_DIR / weight_file
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

    _register_font()
    return (FONT_NAME,)


@app.cell
def __(mo):
    mo.md(r"""## 1. Download runs and build the per-run table""")
    return


@app.cell
def __(
    JOB_TYPES,
    env_name_of,
    fetch_history,
    get_runs,
    is_valid_value,
    pl,
):
    runs = get_runs(JOB_TYPES)
    print(f"Found {len(runs)} runs for job types {JOB_TYPES}")

    # hist_df: one row per (run, epoch) — the full "{env}/average_return" training
    # curve, used both for the mean/median return-during-training line plots AND to
    # derive each run's training_return (mean over the whole history).
    # df: one row per run — training_return = mean of that run's history, used for
    # the CDF. The run summary (last logged value) is NOT used in v2.
    _records = []
    _hist_records = []
    _skipped = 0
    for run in runs:
        env_name = env_name_of(run)
        metric = f"{env_name}/average_return"

        _run_values = []
        for _row in fetch_history(run, [metric]):
            _value = _row.get(metric)
            _step = _row.get("_step")
            if _step is None or not is_valid_value(_value):
                continue
            _run_values.append(float(_value))
            _hist_records.append(
                {
                    "job_type": run.job_type,
                    "env_name": env_name,
                    "run_id": run.id,
                    "step": int(_step),
                    "value": float(_value),
                }
            )

        if not _run_values:  # no usable history -> can't compute training_return
            _skipped += 1
            continue

        _records.append(
            {
                "job_type": run.job_type,
                "env_name": env_name,
                "run_id": run.id,
                "training_return": sum(_run_values) / len(_run_values),
            }
        )

    df = pl.DataFrame(_records)
    hist_df = pl.DataFrame(_hist_records)
    print(f"Kept {len(df)} runs ({_skipped} skipped) | {len(hist_df)} history rows")
    df.head()
    return df, env_name, hist_df, metric, run, runs


@app.cell
def __(df, pl):
    # How many runs per (job_type, env) — this is the count resampled in bootstrap.
    counts = df.group_by("job_type", "env_name").agg(pl.len().alias("n_runs")).sort("env_name", "job_type")
    counts
    return (counts,)


@app.cell
def __(df):
    # Sanity: every env should have every job type present.
    _n_jobtypes = df["job_type"].n_unique()
    for _env, _sub in df.group_by("env_name"):
        assert _sub["job_type"].n_unique() == _n_jobtypes, f"env {_env} is missing job types"
    print(f"OK: {df['env_name'].n_unique()} envs x {_n_jobtypes} job types")
    return


@app.cell
def __(ENV_FILTERS, df, short_env_name):
    # Env order used across all plots: follow ENV_FILTERS, then anything else.
    def _env_sort_key(env):
        short = short_env_name(env, ENV_FILTERS)
        return ENV_FILTERS.index(short) if short in ENV_FILTERS else len(ENV_FILTERS)

    ORDERED_ENVS = sorted(df["env_name"].unique().to_list(), key=_env_sort_key)
    ORDERED_ENVS
    return (ORDERED_ENVS,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 2. Bootstrap the CDF

        For each resample and each `(job_type, env)` combination we sample its runs with
        replacement (same count), then derive the per-run `cdf_score` on the runs'
        **training_return** (mean return over the whole training): pool all resampled runs
        within each env, rank each run and divide by the pool size, and only then average per
        `(job_type, env)`. (The training-return curves are bootstrapped separately below.)
        """
    )
    return


@app.cell
def __(BOOTSTRAP_DIM, SEED, df, np, pl):
    def bootstrap(data: pl.DataFrame, n_iter: int, seed: int) -> pl.DataFrame:
        """Return a long DataFrame with one row per (iter, job_type, env) holding the
        per-run CDF score (column `cdf_score`), computed on `training_return`.

        Within each (iter, env) we pool every resampled run across all job types, rank
        each run and divide by the pool size (empirical CDF position), then average those
        normalized ranks per job type — aggregation happens only AFTER the per-run ranking.
        This is the probability-of-superiority / common-language effect size, and keeps the
        full run-level spread.

        Resampling with replacement creates ties, so ranking uses polars' default "average"
        tie method.

        Reproducibility: polars' `group_by` does not guarantee group iteration order, and a
        single `rng` is consumed sequentially across groups, so an undeterministic loop order
        would make `seed` meaningless. We sort by the group keys (and by `run_id` within each
        group) first and iterate with `maintain_order=True`, so the loop order — and therefore
        every draw from `rng` — is identical on every run.
        """
        rng = np.random.default_rng(seed)
        frames = []
        data_sorted = data.sort(["job_type", "env_name", "run_id"])
        for (job_type, env_name), sub in data_sorted.group_by(["job_type", "env_name"], maintain_order=True):
            train = sub["training_return"].to_numpy()  # already sorted by run_id: deterministic order
            n = train.shape[0]

            idx = rng.integers(0, n, size=(n_iter, n))  # (n_iter, n) resample indices
            iter_col = np.repeat(np.arange(n_iter), n)  # this combo's n runs per iteration
            frames.append(
                pl.DataFrame(
                    {
                        "iter": iter_col.astype(np.int32),
                        "train_value": train[idx].reshape(-1),
                    }
                ).with_columns(
                    job_type=pl.lit(job_type),
                    env_name=pl.lit(env_name),
                )
            )

        boot_runs = pl.concat(frames)
        # Rank each individual run within its (iter, env) pool, normalized by pool size.
        boot_runs = boot_runs.with_columns(
            cdf_run=pl.col("train_value").rank().over("iter", "env_name")
            / pl.col("train_value").count().over("iter", "env_name")
        )
        # Aggregate AFTER ranking: average the per-run scores per (iter, env, job_type).
        boot = boot_runs.group_by("iter", "env_name", "job_type").agg(
            pl.mean("cdf_run").alias("cdf_score"),
        )
        return boot

    boot = bootstrap(df, BOOTSTRAP_DIM, SEED)

    # Sanity checks.
    assert boot["cdf_score"].min() > 0 and boot["cdf_score"].max() <= 1.0
    _per_group = boot.group_by("job_type", "env_name").agg(pl.len().alias("n")).sort("n")
    assert _per_group["n"].max() == BOOTSTRAP_DIM and _per_group["n"].min() == BOOTSTRAP_DIM
    boot.head()
    return boot, bootstrap


@app.cell
def __(mo):
    mo.md(r"""## 3. 95% confidence intervals""")
    return


@app.cell
def __(boot, df, pl):
    def observed_cdf(data: pl.DataFrame) -> pl.DataFrame:
        """The per-run CDF computed once on the real (un-resampled) data — used as the
        bar height / plotted point, mirroring exactly the per-iteration formula in
        `bootstrap()` but with no resampling. Returns one row per (env_name, job_type)
        with column `cdf_score`."""
        d = data.with_columns(
            cdf_run=pl.col("training_return").rank().over("env_name")
            / pl.col("training_return").count().over("env_name")
        )
        return d.group_by("env_name", "job_type").agg(
            pl.mean("cdf_run").alias("cdf_score"),
        )

    observed = observed_cdf(df)
    observed_across = observed.group_by("job_type").agg(
        pl.mean("cdf_score").alias("avg_cdf"),
    )

    def summarize(boot_data: pl.DataFrame, obs_data: pl.DataFrame, group: list[str], value: str) -> pl.DataFrame:
        """Point estimate = the observed statistic on the real data (column `value` in
        `obs_data`); 95% CI = the bootstrap percentile band for that same statistic (column
        `value` in `boot_data`, one row per iter). Bars and curves then follow the same
        convention: plot the real value, shade the bootstrap uncertainty around it."""
        ci = boot_data.group_by(group).agg(
            pl.quantile(value, 0.025, "linear").alias("ci_low"),
            pl.quantile(value, 0.975, "linear").alias("ci_high"),
        )
        point = obs_data.select(*group, pl.col(value).alias("mean"))
        return point.join(ci, on=group, how="left")

    # (1) average CDF across environments, per job type.
    across_boot = boot.group_by("iter", "job_type").agg(
        pl.mean("cdf_score").alias("avg_cdf"),
    )
    avg_cdf_across_ci = summarize(across_boot, observed_across, ["job_type"], "avg_cdf")

    # (2) average CDF per environment, per job type.
    avg_cdf_per_env_ci = summarize(boot, observed, ["job_type", "env_name"], "cdf_score")

    # (The mean/median return-during-training statistics are curves over epochs and
    # are bootstrapped separately in the next cell.)

    # Sanity: CI is well-ordered. (`mean` is the observed statistic on the real data,
    # not the bootstrap mean, so it should almost always land inside its own bootstrap CI —
    # but we don't hard-assert ci_low <= mean <= ci_high, since a pathological/highly
    # skewed resampling distribution could in principle place it just outside.)
    for _name, _tbl in [
        ("avg_cdf_across", avg_cdf_across_ci),
        ("avg_cdf_per_env", avg_cdf_per_env_ci),
    ]:
        assert (_tbl["ci_low"] <= _tbl["ci_high"]).all(), _name

    print("avg CDF per env (per job type):")
    print(avg_cdf_per_env_ci.sort("env_name", "job_type"))
    avg_cdf_across_ci.sort("mean", descending=True)
    return (
        across_boot,
        avg_cdf_across_ci,
        avg_cdf_per_env_ci,
        observed,
        observed_across,
        observed_cdf,
        summarize,
    )


@app.cell
def __(mo):
    mo.md(
        r"""
        ## 3b. Bootstrap the training curves

        Mean and median return *during training* are curves over epochs, so we bootstrap them
        per epoch. For each `(job_type, env)` we align the runs on their shared epochs into a
        `runs x epochs` matrix, then resample runs with replacement `BOOTSTRAP_DIM` times. Each
        resample gives a mean curve and a median curve; the plotted line is the statistic on the
        observed runs and the shaded band is its 2.5-97.5 percentile bootstrap CI.
        """
    )
    return


@app.cell
def __(BOOTSTRAP_DIM, SEED, hist_df, np, pl):
    def bootstrap_curves(hist: pl.DataFrame, n_iter: int, seed: int, chunk: int = 500):
        """Per (job_type, env), bootstrap the across-run mean and median return at every
        epoch. Returns two long DataFrames (mean, median) with columns:
        job_type, env_name, step, value (observed statistic), ci_low, ci_high.

        Reproducibility: as in `bootstrap`, group order and per-group array order must be
        deterministic for a single shared `rng` to give bit-identical results on rerun. We
        sort by the group keys and iterate with `maintain_order=True`, and additionally sort
        the pivoted run_id columns (`pivot` does not guarantee column order) so the
        run-axis <-> matrix-row mapping is identical every time.
        """
        rng = np.random.default_rng(seed)
        mean_frames, median_frames = [], []
        hist_sorted = hist.sort(["job_type", "env_name", "run_id", "step"])
        for (job_type, env_name), sub in hist_sorted.group_by(["job_type", "env_name"], maintain_order=True):
            # runs x epochs matrix on the epochs shared by every run in this combo.
            wide = (
                sub.pivot(on="run_id", index="step", values="value", aggregate_function="mean")
                .drop_nulls()
                .sort("step")
            )
            run_cols = sorted(c for c in wide.columns if c != "step")  # deterministic run order
            steps = wide["step"].to_numpy()
            M = wide.select(run_cols).to_numpy().T  # (n_runs, n_steps)
            n = M.shape[0]

            idx = rng.integers(0, n, size=(n_iter, n))
            means = np.empty((n_iter, steps.shape[0]))
            medians = np.empty((n_iter, steps.shape[0]))
            for s in range(0, n_iter, chunk):
                e = min(s + chunk, n_iter)
                sample = M[idx[s:e]]  # (chunk, n_runs, n_steps)
                means[s:e] = sample.mean(axis=1)
                medians[s:e] = np.median(sample, axis=1)

            mean_lo, mean_hi = np.percentile(means, [2.5, 97.5], axis=0)
            med_lo, med_hi = np.percentile(medians, [2.5, 97.5], axis=0)

            mean_frames.append(
                pl.DataFrame(
                    {"step": steps, "value": M.mean(axis=0), "ci_low": mean_lo, "ci_high": mean_hi}
                ).with_columns(job_type=pl.lit(job_type), env_name=pl.lit(env_name))
            )
            median_frames.append(
                pl.DataFrame(
                    {"step": steps, "value": np.median(M, axis=0), "ci_low": med_lo, "ci_high": med_hi}
                ).with_columns(job_type=pl.lit(job_type), env_name=pl.lit(env_name))
            )

        return pl.concat(mean_frames), pl.concat(median_frames)

    mean_curve_df, median_curve_df = bootstrap_curves(hist_df, BOOTSTRAP_DIM, SEED)
    assert (mean_curve_df["ci_low"] <= mean_curve_df["ci_high"]).all()
    assert (median_curve_df["ci_low"] <= median_curve_df["ci_high"]).all()
    mean_curve_df.head()
    return bootstrap_curves, mean_curve_df, median_curve_df


@app.cell
def __(mo):
    mo.md(r"""## 4. Plot helpers""")
    return


@app.cell
def __(
    JOB_TYPE_COLORS,
    JOB_TYPE_LABELS,
    JOB_TYPE_ORDER,
    OUT_DIR,
    mpatches,
    pl,
):
    def draw_bars(ax, stat_df: pl.DataFrame, show_labels: bool = False):
        """Draw one bar per job type (fixed order & color) with asymmetric 95% CI
        error bars. `stat_df` must have columns: job_type, mean, ci_low, ci_high.
        With many job types the names under the bars get cramped, so by default no
        x-tick labels are drawn — use `legend_patches()` + a figure/axes legend
        instead (see `add_legend`)."""
        xs, heights, colors, labels, lows, highs = [], [], [], [], [], []
        for i, jt in enumerate(JOB_TYPE_ORDER):
            row = stat_df.filter(pl.col("job_type") == jt)
            if row.is_empty():
                continue
            m = row["mean"][0]
            xs.append(i)
            heights.append(m)
            colors.append(JOB_TYPE_COLORS.get(jt, "#000000"))
            labels.append(JOB_TYPE_LABELS.get(jt, jt))
            # `mean` here is the observed statistic on the real data, so it should almost
            # always fall inside its own bootstrap CI. Clamp to >= 0 anyway as a defensive
            # guard against the rare pathological case where it doesn't (would otherwise
            # yield a negative whisker and error out matplotlib's yerr).
            lows.append(max(0.0, m - row["ci_low"][0]))
            highs.append(max(0.0, row["ci_high"][0] - m))

        ax.bar(
            xs,
            heights,
            color=colors,
            width=0.65,
            yerr=[lows, highs],
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )
        if show_labels:
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
        else:
            ax.set_xticks([])
        ax.margins(x=0.15)

    def legend_patches():
        """Proxy color patches, fixed order/color/label, for a job-type legend."""
        return [
            mpatches.Patch(facecolor=JOB_TYPE_COLORS.get(jt, "#000000"), label=JOB_TYPE_LABELS.get(jt, jt))
            for jt in JOB_TYPE_ORDER
        ]

    def add_legend(ax):
        """Attach a job-type legend to the right of `ax` (outside the axes)."""
        ax.legend(
            handles=legend_patches(),
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )

    def draw_curves(ax, curve_df: pl.DataFrame):
        """Plot one line per job type (fixed order & color) with a shaded 95% bootstrap
        CI band. `curve_df` must have columns: job_type, step, value, ci_low, ci_high."""
        for jt in JOB_TYPE_ORDER:
            jt_df = curve_df.filter(pl.col("job_type") == jt).sort("step")
            if jt_df.is_empty():
                continue
            color = JOB_TYPE_COLORS.get(jt, "#000000")
            x = jt_df["step"].to_numpy()
            ax.plot(x, jt_df["value"].to_numpy(), color=color, linewidth=1.8, label=JOB_TYPE_LABELS.get(jt, jt))
            ax.fill_between(
                x, jt_df["ci_low"].to_numpy(), jt_df["ci_high"].to_numpy(), color=color, alpha=0.25, linewidth=0
            )

    def save_svg(fig, name: str):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUT_DIR / f"{name}.svg"
        fig.savefig(path, format="svg", bbox_inches="tight")
        print(f"Saved {path}")
        return path

    return add_legend, draw_bars, draw_curves, legend_patches, save_svg


@app.cell
def __(mo):
    mo.md(r"""## 5. Plots""")
    return


@app.cell
def __(mo):
    mo.md(r"""### Average CDF across environments (per job type)""")
    return


@app.cell
def __(add_legend, avg_cdf_across_ci, draw_bars, np, plt, save_svg):
    fig_across, ax_across = plt.subplots(figsize=(4.5, 4))
    draw_bars(ax_across, avg_cdf_across_ci)
    ax_across.set_ylabel("Average CDF score")
    ax_across.set_ylim(0, 1)
    ax_across.set_yticks(np.arange(0, 1.01, 0.1))
    add_legend(ax_across)
    fig_across.tight_layout()
    save_svg(fig_across, "avg_cdf_across_env")
    fig_across
    return ax_across, fig_across


@app.cell
def __(mo):
    mo.md(r"""### Average CDF per environment (per job type)""")
    return


@app.cell
def __(
    ENV_FILTERS,
    ORDERED_ENVS,
    add_legend,
    avg_cdf_per_env_ci,
    draw_bars,
    np,
    pl,
    plt,
    save_svg,
    short_env_name,
):
    fig_cdf_env, axes_cdf_env = plt.subplots(1, len(ORDERED_ENVS), figsize=(3.0 * len(ORDERED_ENVS), 4), sharey=True)
    if len(ORDERED_ENVS) == 1:
        axes_cdf_env = [axes_cdf_env]

    for _ax, _env in zip(axes_cdf_env, ORDERED_ENVS):
        draw_bars(_ax, avg_cdf_per_env_ci.filter(pl.col("env_name") == _env))
        _ax.set_title(short_env_name(_env, ENV_FILTERS))
        _ax.set_ylim(0, 1)
    axes_cdf_env[0].set_ylabel("CDF score")
    axes_cdf_env[0].set_yticks(np.arange(0, 1.01, 0.1))
    add_legend(axes_cdf_env[-1])
    fig_cdf_env.tight_layout()
    save_svg(fig_cdf_env, "avg_cdf_per_env")
    fig_cdf_env
    return axes_cdf_env, fig_cdf_env


@app.cell
def __(mo):
    mo.md(r"""### Mean return during training (per environment) — line plots with 95% CI band""")
    return


@app.cell
def __(
    ENV_FILTERS,
    ORDERED_ENVS,
    draw_curves,
    mean_curve_df,
    pl,
    plt,
    save_svg,
    short_env_name,
):
    fig_mean, axes_mean = plt.subplots(1, len(ORDERED_ENVS), figsize=(3.5 * len(ORDERED_ENVS), 4))
    if len(ORDERED_ENVS) == 1:
        axes_mean = [axes_mean]

    for _ax, _env in zip(axes_mean, ORDERED_ENVS):
        draw_curves(_ax, mean_curve_df.filter(pl.col("env_name") == _env))
        _ax.set_title(short_env_name(_env, ENV_FILTERS))
        _ax.set_xlabel("Epoch")
    axes_mean[0].set_ylabel("Mean return during training")
    axes_mean[-1].legend(frameon=False, loc="best")
    fig_mean.tight_layout()
    save_svg(fig_mean, "mean_training_return_per_env")
    fig_mean
    return axes_mean, fig_mean


@app.cell
def __(mo):
    mo.md(r"""### Median return during training (per environment) — line plots with 95% CI band""")
    return


@app.cell
def __(
    ENV_FILTERS,
    ORDERED_ENVS,
    draw_curves,
    median_curve_df,
    pl,
    plt,
    save_svg,
    short_env_name,
):
    fig_median, axes_median = plt.subplots(1, len(ORDERED_ENVS), figsize=(3.5 * len(ORDERED_ENVS), 4))
    if len(ORDERED_ENVS) == 1:
        axes_median = [axes_median]

    for _ax, _env in zip(axes_median, ORDERED_ENVS):
        draw_curves(_ax, median_curve_df.filter(pl.col("env_name") == _env))
        _ax.set_title(short_env_name(_env, ENV_FILTERS))
        _ax.set_xlabel("Epoch")
    axes_median[0].set_ylabel("Median return during training")
    axes_median[-1].legend(frameon=False, loc="best")
    fig_median.tight_layout()
    save_svg(fig_median, "median_training_return_per_env")
    fig_median
    return axes_median, fig_median


@app.cell
def __(mo):
    mo.md(
        r"""
        ### Combined: average CDF across environments alongside per-environment CDF

        The leftmost panel is the average across environments; a dotted divider separates it
        from the per-environment panels.
        """
    )
    return


@app.cell
def __(
    ENV_FILTERS,
    ORDERED_ENVS,
    add_legend,
    avg_cdf_across_ci,
    avg_cdf_per_env_ci,
    draw_bars,
    np,
    pl,
    plt,
    save_svg,
    short_env_name,
):
    _n = len(ORDERED_ENVS)
    fig_comb, axes_comb = plt.subplots(1, _n + 1, figsize=(3.0 * (_n + 1), 4), sharey=True)

    # Leftmost panel: average across environments.
    draw_bars(axes_comb[0], avg_cdf_across_ci)
    axes_comb[0].set_ylabel("CDF score")
    axes_comb[0].set_ylim(0, 1)
    axes_comb[0].set_yticks(np.arange(0, 1.01, 0.1))

    # Remaining panels: one per environment.
    for _ax, _env in zip(axes_comb[1:], ORDERED_ENVS):
        draw_bars(_ax, avg_cdf_per_env_ci.filter(pl.col("env_name") == _env))
        _ax.set_title(short_env_name(_env, ENV_FILTERS))
        _ax.set_ylim(0, 1)

    add_legend(axes_comb[-1])
    fig_comb.tight_layout()

    # Dotted divider between the across-env panel and the per-env panels.
    fig_comb.canvas.draw()
    _x = (axes_comb[0].get_position().x1 + axes_comb[1].get_position().x0) / 2
    _divider = plt.Line2D(
        [_x, _x], [0.05, 0.95], transform=fig_comb.transFigure, color="gray", linestyle=":", linewidth=1.2
    )
    fig_comb.add_artist(_divider)

    save_svg(fig_comb, "cdf_across_and_per_env")
    fig_comb
    return axes_comb, fig_comb


if __name__ == "__main__":
    app.run()
