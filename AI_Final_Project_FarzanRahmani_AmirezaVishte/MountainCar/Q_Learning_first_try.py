import gymnasium as gym
import utils
import random
import math
# import numpy as np

# Classic Control
env = gym.make("MountainCar-v0", render_mode="human")
# env = gym.make("MountainCar-v0")
# Action Space: Discrete(3) (0: Accelerate to the left, 1: Don't accelerate, 2: Accelerate to the right)
# Observation Shape: (2,) [position of the car along the x-axis(position (m)), velocity of the car(velocity (v))]
# Observation High: [0.6 0.07]
# Observation Low: [-1.2 -0.07]
# End Condition: 1.Termination:The car reaches the flag (position >= 0.5) or 2.Truncation: 200 timesteps have passed.

# solve the problem using Q-learning


def getQValue(Qvalues, state, action):  # Q(state, action) ->  state is tuple, action is string
    """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
    """
    return Qvalues[(state, action)]


def computeValueFromQValues(Qvalues, state, legalActions):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    if len(legalActions) == 0:
        return 0.0  # terrminal state
    value = -float('inf')  # if value = 0.0 -> negative rewards doesnt count
    for action in legalActions:
        value = max(value, getQValue(Qvalues, state, action))
    return value


def computeActionFromQValues(Qvalues, state, legalActions):
    """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
    """
    action = None
    if len(legalActions) == 0:
        return action  # terrminal state

    action = max(legalActions, key=lambda a: getQValue(Qvalues, state, a))

    return action


def getAction(Qvalues, state, legalActions, epsilon):
    """
        Compute the action to take in the current state.  With
        probability self.epsilon, we should take a random action and
        take the best policy action otherwise.  Note that if there are
        no legal actions, which is the case at the terminal state, you
        should choose None as the action.
        HINT: You might want to use util.flipCoin(prob)
        HINT: To pick randomly from a list, use random.choice(list)
    """
    action = None
    if len(legalActions) == 0:  # terminal state
        return action

    epsilon_greedy = utils.flipCoin(epsilon)
    if epsilon_greedy:
        action = random.choice(legalActions)
    else:
        action = computeActionFromQValues(Qvalues, state, legalActions)

    return action


def update(Qvalues, state, action, nextState, reward: float, discount: float, alpha: float, legalActions):
    """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here
        NOTE: You should never call this function,
        it will be called on your behalf
    """
    sample = reward + discount * \
        computeValueFromQValues(Qvalues, nextState, legalActions)
    Qvalues[(state, action)] = (1 - alpha) * \
        getQValue(Qvalues, state, action) + alpha * sample


def getPolicy(Qvalues, state, legalActions):
    # optimal policy
    return computeActionFromQValues(Qvalues, state, legalActions)


def getValue(Qvalues, state, legalActions):
    return computeValueFromQValues(Qvalues, state, legalActions)


def fix_observation(observation, position_bins, velocity_bins):
    observation = [round(observation[0], position_bins),
                   round(observation[1], velocity_bins)]
    return tuple(observation)


def qLearning(env, num_episodes: int, discount: float, alpha: float, epsilon: float, epsilon_decay: float):
    """
        Returns:
            Qvalues: A dictionary that maps from (state,action) pairs
            to Q values.
    """
    Qvalues = utils.Counter()  # Qvalues[(state, action)] = Qvalue
    # Qvalues = utils.Counter({"1":1, "2":2}) # Qvalues[(state, action)] = Qvalue
    for episode in range(num_episodes):
        # state = env.reset(seed=42) # observe the initial state
        observation, info = env.reset()
        state = fix_observation(observation, 3, 4)
        terminated = False
        truncated = False
        while (not terminated) and (not truncated):
            legalActions = [0, 1, 2]  # env.action_space
            action = getAction(Qvalues, state, legalActions, epsilon)
            observation, reward, terminated, truncated, info = env.step(
                action)  # nextState, reward, done, _ = env.step(action)
            nextState = fix_observation(observation, 3, 4)
            update(Qvalues, state, action, nextState,
                   reward, discount, alpha, legalActions)
            state = nextState

        epsilon *= epsilon_decay  # decay epsilon
    return Qvalues


epsilon = 0.9  # could be lower as the agent learns
alpha = 0.1  # learning rate
discount = 0.9  # discount factor for future rewards (gamma)
# Qvalues = utils.Counter() # Qvalues[(state, action)] = Qvalue
observation, info = env.reset(seed=42)
learned_Qvalues = qLearning(env, 10, discount, alpha, epsilon, 0.99)
# save Qvalues in a file
# Qvalues_dictionary = dict(learned_Qvalues)


# for _ in range(10000):
#     a = env.action_space
#     # print(a) # Discrete(3)
#     action = env.action_space.sample() # this is where you would insert your policy
#     # print(action) # 0,1,2

#     observation, reward, terminated, truncated, info = env.step(action)
#     # print(observation) # [-0.44479132, 0.00041747934]
#     # print(reward) # -1.0
#     # print(terminated) # True or False
#     # print(truncated) # True or False
#     # print(info) # {}

#     if terminated or truncated:
#         observation, info = env.reset() # On reset, the options parameter allows the user to change the bounds used to determine the new random state.

env.close()
