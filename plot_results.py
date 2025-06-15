"""
Plot the reward curves and summary comparison for all models.
"""

import matplotlib.pyplot as plt
import numpy as np

algorithms = ["PPO", "DQN", "A2C"]
rewards_dict = {algo: np.load(f"plots/{algo}_rewards.npy") for algo in algorithms}

# Line Plot
plt.figure(figsize=(10, 6))
for algo, rewards in rewards_dict.items():
    plt.plot(rewards, label=algo)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode for PPO, DQN, A2C")
plt.legend()
plt.grid(True)
plt.savefig("plots/reward_lineplot.png")
plt.show()

# Bar Chart
avg_rewards = [np.mean(r) for r in rewards_dict.values()]
std_rewards = [np.std(r) for r in rewards_dict.values()]

plt.figure(figsize=(8, 5))
plt.bar(algorithms, avg_rewards, yerr=std_rewards, capsize=5)
plt.ylabel("Average Total Reward")
plt.title("Average Reward Comparison")
plt.savefig("plots/reward_bar_chart.png")
plt.show()

