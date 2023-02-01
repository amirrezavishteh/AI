import gymnasium as gym
# env = gym.make("Taxi-v3", render_mode="human")
env = gym.make("CliffWalking", render_mode="human")
# toy text
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
