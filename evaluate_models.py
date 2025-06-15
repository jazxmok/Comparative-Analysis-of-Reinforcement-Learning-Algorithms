"""
Evaluate trained RL models over multiple episodes and log rewards.
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, DQN, A2C

def evaluate_model(algorithm, env_id="LunarLander-v2", episodes=100, model_path="models"):
    model_class = {"PPO": PPO, "DQN": DQN, "A2C": A2C}[algorithm]
    model = model_class.load(f"{model_path}/{algorithm}_LunarLander")
    env = gym.make(env_id, render_mode=None)
    rewards = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    print(f"{algorithm} - Average Reward: {mean_reward:.2f}, Std Dev: {std_reward:.2f}")
    np.save(f"plots/{algorithm}_rewards.npy", rewards)

if __name__ == "__main__":
    for algo in ["PPO", "DQN", "A2C"]:
        evaluate_model(algo)

