import os

from experiment_launcher import Launcher, is_local

LOCAL = is_local()
TEST = False
USE_CUDA = True

N_SEEDS = 10

if LOCAL:
    N_EXPS_IN_PARALLEL = 3
else:
    N_EXPS_IN_PARALLEL = 3

N_CORES = N_EXPS_IN_PARALLEL
MEMORY_SINGLE_JOB = 1000
MEMORY_PER_CORE = N_EXPS_IN_PARALLEL * MEMORY_SINGLE_JOB // N_CORES
PARTITION = "stud"  # 'amd', 'rtx'
GRES = "gpu:1" if USE_CUDA else None  # gpu:rtx2080:1, gpu:rtx3080:1
CONDA_ENV = ".env" if "CONDA_PREFIX" in os.environ else None


def launcher():
    launcher = Launcher(
        exp_name="test_launcher",
        exp_file="implementations.DQN",
        # project_name='project01234',  # for hrz cluster
        n_seeds=N_SEEDS,
        n_exps_in_parallel=N_EXPS_IN_PARALLEL,
        n_cores=N_CORES,
        memory_per_core=MEMORY_PER_CORE,
        days=0,
        hours=23,
        minutes=59,
        seconds=0,
        partition=PARTITION,
        gres=GRES,
        use_timestamp=True,
        compact_dirs=False,
    )

    games = ["Acrobot-v1", "MountainCar-v0", "LunarLander-v2"]
    n_epochs = 100
    # TODO: the launcher should take as input the n_epochs and the wandb_log so that I can test it easily.

    for game in games:
        launcher.add_experiment(
            config__=game, game_name=game, n_epochs=n_epochs, wandb_log=True
        )

    launcher.run(LOCAL, TEST)


# TODO: here I need an input parser with a switch for the paths.
# I will also need to be able to choose the n_epochs for testing
if __name__ == "__main__":
    launcher()
