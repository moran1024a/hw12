import gymnasium as gym

env = gym.make("LunarLander-v3")
obs, info = env.reset()

print("state_dim:", len(obs))
print("action_space:", env.action_space)

env.close()