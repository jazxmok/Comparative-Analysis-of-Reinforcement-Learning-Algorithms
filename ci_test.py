import gymnasium as gym

env = gym.make("LunarLander-v2")
obs, _ = env.reset()
print("Test environment works. Observation shape:", obs.shape)
