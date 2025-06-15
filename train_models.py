"""
Train PPO, DQN, and A2C models using Gymnasium's LunarLander-v2 environment.
"""

import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

def train_model(algorithm, env_id="LunarLander-v2", timesteps=100_000, save_path="models"):
    env = DummyVecEnv([lambda: gym.make(env_id)])
    model_class = {"PPO": PPO, "DQN": DQN, "A2C": A2C}[algorithm]
    model = model_class("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(f"{save_path}/{algorithm}_LunarLander")
    print(f"{algorithm} training complete and saved.")
    return model

if __name__ == "__main__":
    for algo in ["PPO", "DQN", "A2C"]:
        train_model(algo)

