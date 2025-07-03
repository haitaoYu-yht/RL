"""
Register custom environments with Gymnasium.
"""

import gymnasium as gym
from envs.finger_env import FingerPushEnv

# Register the environment
gym.register(
    id="FingerPush-v0",
    entry_point="envs.finger_env:FingerPushEnv",
    max_episode_steps=300,
) 
