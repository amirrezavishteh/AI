import gymnasium as gym
# Classic Control
env = gym.make("MountainCarContinuous-v0",render_mode="human")
# Action Space: Box(-1.0, 1.0, (1,), float32) (representing the directional force applied on the car. The action is clipped in the range [-1,1] and multiplied by a power of 0.0015.)
# velocityt+1 = velocityt + (action - 1) * force - cos(3 * positiont) * gravity
# positiont+1 = positiont + velocityt+1
# Observation Shape: (2,) [position of the car along the x-axis(position (m)), velocity of the car(velocity (v))]
# Observation High: [0.6 0.07]
# Observation Low: [-1.2 -0.07]
# The goal is to reach the flag placed on top of the right hill as quickly as possible, as
#  such the agent is penalised for each timestep.
# End Condition: 1.Termination:The car reaches the flag (position >= 0.5) or 2.Truncation: 999 timesteps have passed.
# Reward: A negative reward of -0.1 * action^2 is received at each timestep to penalise for taking actions of large 
# magnitude. If the mountain car reaches the goal then a positive reward of +100 is added to the negative 
# reward for that timestep.
# https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/

observation, info = env.reset(seed=42)
# print(observation) [[-0.6, -0.4] ,0.0]
# print(info) {}

for _ in range(1000):
    # a = env.action_space
    # print(a) # Box(-1.0, 1.0, (1,), float32)
    # action = env.action_space.sample() # this is where you would insert your policy
    # print(action) # [0.4021235]
    
    # action = 2 # naive thought , accelerate to the right
    action =[0.1]
    if (observation[1] > 0):
        action = [1]
    elif (observation[1] < 0):
        action = [-1]
    # do not work ! 

    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation) # [-0.44518813, 2.06646e-05]
    # print(reward) # -0.016170331796832117 = -0.1 * action2
    # print(terminated) # True or False
    # print(truncated) # True or False
    # print(info) # {}

    if terminated or truncated:
        observation, info = env.reset() # On reset, the options parameter allows the user to change the bounds used to determine the new random state.

env.close()