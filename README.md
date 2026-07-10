# Dynamic_features

## How to launch an Experiment
To launch an experiment with one of the supported RL algorithms and environments, follow these steps:

1. Navigate to the `src/dynamic_features` directory.
2. Run the `launcher.py` script with the desired feature as an argument.

For example, to launch a multirun DQN experiment with three seed on Acrobot, run the following command:

```bash
python launcher.py --multirun seed=0,1,2 environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole
algorithm=dqn
wandb_log=True wandb.job_type="DQN_Run" wandb.mode="ONLINE"
```
## How to log the offline run to wandb
```bash
wandb sync --include-offline --sync-all path/to/wandb
```

## Hyperparameter Grid Search in practice

### Random
```bash
python launcher.py --multirun seed=0,1,2,3 \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=random \
wandb_log=True wandb.job_type="random_sweep"
```

### DQN
```bash
python launcher.py --multirun seed=0,1,2,3 \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=dqn \
algorithm.policy.final_exp_frame=15_000,30_000 \
algorithm.agent.target_update_frequency=100,500 \
algorithm.agent.batch_size=64,128 \
algorithm.agent.n_features=64,128 \
algorithm.agent.learning_rate=1e-4,5e-4,1e-3 \
wandb_log=True wandb.job_type="DQN_sweep"
```

### Rainbow
```bash
python launcher.py --multirun seed=0,1,2,3 \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=rainbow \
algorithm.policy.final_exp_frame=15_000,30_000 \
algorithm.agent.max_replay_size=50_000,100_000 \
algorithm.agent.target_update_frequency=500,1_000,2_000 \
algorithm.agent.batch_size=64,128 \
algorithm.agent.learning_rate=1e-4,5e-4 \
wandb_log=True wandb.job_type="Rainbow_sweep"
```

### Sarsa Lambda with Rec Tiles
```bash
python launcher.py --multirun seed=0,1,2,3 \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=sarsa_lambda_rec \
algorithm.policy.final_exp_frame=15_000,30_000 \
algorithm.agent.n_tilings=8,16,32 \
algorithm.agent.n_tiles=4,8 \
algorithm.agent.lambda_coeff=0.5,0.75,0.9,0.99 \
wandb_log=True wandb.job_type="sarsa_lambda_rec_sweep"
```

### Sarsa Lambda with Dynamic Tiles
```bash
python launcher.py --multirun seed=0,1,2,3 \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=sarsa_lambda_dt \
algorithm.policy.final_exp_frame=15_000,30_000 \
algorithm.agent.n_tilings=24,32 \
algorithm.agent.n_clusters=32,48,64 \
algorithm.agent.lambda_coeff=0.5,0.75,0.9,0.99 \
wandb_log=True wandb.job_type="sarsa_lambda_dt_retr_sweep"
```

## Final runs commands

### DQN
```bash
uv run python launcher.py --multirun seed=$(seq 100 149 | paste -sd, -) \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=dqn \
wandb_log=True wandb.job_type="DQN_final_v2"
```

### Rainbow
```bash
uv run python launcher.py --multirun seed=$(seq 100 149 | paste -sd, -) \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=rainbow \
wandb_log=True wandb.job_type="rainbow_final_v2"
```

### Sarsa Lambda with Rec Tiles
```bash
uv run python launcher.py --multirun seed=$(seq 100 149 | paste -sd, -) \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=sarsa_lambda_rec \
wandb_log=True wandb.job_type="sarsa_lambda_rec_final_v2"
```

### Sarsa Lambda DT no retr
```bash
uv run python launcher.py --multirun seed=$(seq 100 149 | paste -sd, -) \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=sarsa_lambda_dt algorithm.agent.retrain_features=False \
wandb_log=True wandb.job_type="sl_dt_noretr_final_v2"
```

### Sarsa Lambda DT
```bash
uv run python launcher.py --multirun seed=$(seq 100 149 | paste -sd, -) \
environment=Acrobot,PuddleWorld,MountainCar,LunarLander,CartPole \
algorithm=sarsa_lambda_dt \
wandb_log=True wandb.job_type="sarsa_lambda_dt_final_v2"
```
