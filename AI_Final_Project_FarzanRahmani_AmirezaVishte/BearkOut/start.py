import gymnasium as gym

env = gym.make("ALE/Breakout-v5", render_mode="human")

observation, info = env.reset(seed=42)
# print(observation) []
# print(info) {}

for _ in range(1000):
    a = env.action_space
    # print(a) # Discrete(4)
    action = env.action_space.sample() # this is where you would insert your policy
    # print(action) # 0,1,2

    # insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation) # []
    # print(reward) # 0.0
    # print(terminated) # True or False
    # print(truncated) # True or False
    # print(info) # {}

    print(f"reward: {reward} for action {action}")
    if terminated or truncated:
        observation, info = env.reset()
env.close()