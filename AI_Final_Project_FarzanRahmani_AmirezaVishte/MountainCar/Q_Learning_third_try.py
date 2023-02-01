import gymnasium as gym
import utils
import random
import ujson

# Classic Control
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


def Q_learning(env, num_episodes: int, discount: float, alpha: float, epsilon: float, epsilon_decay: float, Qvalues):
    """
        Returns:
            Qvalues: A dictionary that maps from (state,action) pairs
            to Q values.
    """
    for episode in range(num_episodes):
        # state = env.reset(seed=42) # observe the initial state
        observation, info = env.reset()
        state = fix_observation(observation, 2, 2)
        terminated = False
        truncated = False
        i = 0
        while (not terminated) and (not truncated):
            i += 1
            # Render environment for last five episodes
            # if episode >= (num_episodes - 5):
            # env.render_mode = 'human'
            # env = gym.make("MountainCar-v0",render_mode="human")

            legalActions = [0, 1, 2]  # env.action_space
            action = getAction(Qvalues, state, legalActions, epsilon)
            my_reward = 0
            if (observation[1] > 0):
                if (action == 2):
                    my_reward += 10
                elif (action == 0):
                    my_reward -= 10
            elif (observation[1] < 0):
                if(action == 0):
                    my_reward += 10
                else:
                    my_reward -= 10

            observation, reward, terminated, truncated, info = env.step(
                action)  # nextState, reward, done, _ = env.step(action)
            reward += my_reward
            nextState = fix_observation(observation, 2, 2)
            update(Qvalues, state, action, nextState,
                    reward, discount, alpha, legalActions)
            state = nextState
            if (terminated):
                print("Episode {} finished after {} timesteps with epsilon {} and alpha {}".format(
                    episode, i, epsilon, alpha))

        epsilon *= epsilon_decay  # decay epsilon
        # epsilon -= epsilon/num_episodes
        alpha *= 0.999  # decay alpha
    return Qvalues


# env = gym.make("MountainCar-v0",render_mode="human")
env = gym.make("MountainCar-v0")

epsilon = 0.8  # could be lower as the agent learns (primitive tests = 0.9)
# epsilon = 0.1 # could be lower as the agent learns
alpha = 0.3  # learning rate (start: 0.8)
discount = 0.9  # discount factor for future rewards (gamma)
Qvalues_dictionary = ''
# comment after you remove file and it is your first try
with open(r'Qvalues_dictionary_2.txt', 'r') as f:
    Qvalues_dictionary = ujson.load(f)

# Qvalues[(state, action)] = Qvalue
Qvalues = utils.Counter(Qvalues_dictionary)
observation, info = env.reset(seed=42)
# learned_Qvalues = Q_learning(env, 5000, discount, alpha, epsilon, 0.99, Qvalues)
learned_Qvalues = Q_learning(
    env, 22, discount, 0.15, 0.0001, 0.99, Qvalues)  # for testing
# save Qvalues in a file
Qvalues_dictionary = dict(learned_Qvalues)
# with open('Qvalues_dictionary_2.txt', 'w') as file:
#     file.write(ujson.dumps(Qvalues_dictionary))

env.close()
