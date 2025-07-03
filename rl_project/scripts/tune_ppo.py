import optuna

from cleanrl_utils.tuner import Tuner


# Register the env
from envs import register_envs

tuner = Tuner(
    script="cleanrl/cleanrl/ppo_continuous_action.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    study_name="finger_ppo_tune",
    wandb_kwargs={
        "project": "rl_finger_ppo_tune",
        "entity": "<insert-username>",
    },
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "FingerPush-v0": [0, 150],
    },
    params_fn=lambda trial: {
        "learning-rate": trial.suggest_float("learning-rate", 0.0003, 0.001, log=True),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [2, 4, 8]),
        "update-epochs": trial.suggest_categorical("update-epochs", [5, 10]),
        "ent-coef": trial.suggest_float("ent-coef", 0.0001, 0.1, log=True),
        "num-steps": trial.suggest_categorical("num-steps", [5, 16, 32, 64, 128, 256]),
        "vf-coef": trial.suggest_float("vf-coef", 0, 2),
        "max-grad-norm": trial.suggest_categorical("max-grad-norm", [0.5, 1.0, 1.5]),
        "total-timesteps": 250000,
        "num-envs": [1, 32, 64, 128],
    },
    
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
tuner.tune(
    num_trials=100,
    num_seeds=3,
)
