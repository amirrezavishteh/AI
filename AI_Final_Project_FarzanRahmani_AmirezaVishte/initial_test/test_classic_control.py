import gymnasium as gym
# env = gym.make("MountainCar-v0",render_mode="human")
env = gym.make("CartPole-v1",render_mode="human")
# env = gym.make("CartPole-v1",render_mode="rgb_array")
# Classic Control
observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()