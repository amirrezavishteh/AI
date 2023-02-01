import gymnasium as gym
# Classic Control
env = gym.make("MountainCar-v0", render_mode="human")
# Action Space: Discrete(3) (0: Accelerate to the left, 1: Don't accelerate, 2: Accelerate to the right)
# velocityt+1 = velocityt + (action - 1) * force - cos(3 * positiont) * gravity
# positiont+1 = positiont + velocityt+1
# Observation Shape: (2,) [position of the car along the x-axis(position (m)), velocity of the car(velocity (v))]
# Observation High: [0.6 0.07]
# Observation Low: [-1.2 -0.07]
# The goal is to reach the flag placed on top of the right hill as quickly as possible, as
#  such the agent is penalised with a reward of -1 for each timestep.
# End Condition: 1.Termination:The car reaches the flag (position >= 0.5) or 2.Truncation: 200 timesteps have passed.

observation, info = env.reset(seed=42)
# print(observation) [[-0.6, -0.4]]
# print(info) {}

for _ in range(1000):
    a = env.action_space
    # print(a) # Discrete(3)
    # action = env.action_space.sample() # this is where you would insert your policy
    # print(action) # 0,1,2

    # action = 2 # naive thought , accelerate to the right

    if (observation[1] >= 0):  # velocity is positive
        action = 2  # accelarate right
    else:  # velocity is negative
        action = 0  # accelerate left
    # do not work !

    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation) # [-0.44479132, 0.00041747934]
    # print(reward) # -1.0
    # print(terminated) # True or False
    # print(truncated) # True or False
    # print(info) # {}

    if terminated or truncated:
        # On reset, the options parameter allows the user to change the bounds used to determine the new random state.
        observation, info = env.reset()

env.close()
