import sys
import runpy

# Register the env
from envs import register_envs

num_envs = 1
# num_steps = 24

# command-line arguments for CleanRL
sys.argv = [
    "ppo_continuous_action.py",
    "--env-id", "FingerPush-v0",
    # "--num_envs", str(num_envs),
    # "--num_steps", str(num_steps),
    # "--update_epochs", "5",
    # "--num_minibatches", "4",
    # "--learning_rate", "1e-3",
    # "--target_kl", "0.01",
    # "--ent_coef", "0.01",
    # "--max_grad_norm", "1.0",
    # "--vf_coef", "1.0",
    "--track", "--wandb_entity=<insert-username>", "--wandb_project_name=rl_finger_ppo",
    "--total_timesteps=120000",
    "--save-model",
    "--capture-video"
]

# Run the CleanRL script in the current interpreter
runpy.run_path("cleanrl/cleanrl/ppo_continuous_action.py", run_name="__main__")

# runpy.run_path("scripts/tune_ppo.py", run_name="__main__")