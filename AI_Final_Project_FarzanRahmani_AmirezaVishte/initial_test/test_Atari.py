import gymnasium as gym
env = gym.make("ALE/Asteroids-v5", render_mode="human")
# env = gym.make("ALE/Asteroids-v5")
# env = gym.make("ALE/Atlantis-v5", render_mode="human")
# env = gym.make("ALE/Breakout-v5", render_mode="human")
# ataripython 
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would
    # insert your policy
    observation, reward, terminated, truncated, info = env.step(
        action)
    print(f"reward: {reward} for action {action}")
    if terminated or truncated:
        observation, info = env.reset()
env.close()