# Dynamic_features

## How to launch an Experiment
To launch an experiment with one of the supported RL algorithms and environments, follow these steps:

1. Navigate to the `src/dynamic_features` directory.
2. Run the `launcher.py` script with the desired feature as an argument.

For example, to launch a multirun DQN experiment with three seed on Acrobot, run the following command:

```bash
python launcher.py --multirun seed=0,1,2 environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole wandb_log=True wandb.job_type="DQN_Run" wandb.mode="ONLINE"
```
## How to log the offline run to wandb
```bash
wandb sync --include-offline --sync-all path/to/wandb
```
