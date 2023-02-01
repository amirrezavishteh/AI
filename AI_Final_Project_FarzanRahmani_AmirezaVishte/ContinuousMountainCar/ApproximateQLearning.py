import gymnasium as gym
from utils import Counter, flipCoin
import random
import ujson

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np


def draw_3D_chart(chart_data_time, chart_data_reward, chart_data_action):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(chart_data_time, chart_data_reward,
               chart_data_action, c='r', marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    ax.set_zlabel('Action')
    plt.show()
    # save it in a image file
    # plt.savefig('chart.png')
    ax.figure.savefig('chart.png')


def computeFeatureAccelarating(state, action):
    """
        Returns 1 if action is accelerating, 0 otherwise
    """
    feature_accelerating = 0
    if (state[1] > 0):  # if velocity is positive
        if (action > 0):  # if action is right
            # accelerating right is good (تند شوند)
            feature_accelerating = 1*action*action
        elif(action < 0):  # if action is left
            # accelerating left is bad (کند شوند)
            feature_accelerating = -1*action*action
    elif (state[1] < 0):  # if velocity is negative
        if(action < 0):  # if action is left
            # accelerating left is good (تند شوند)
            feature_accelerating = 1*(action*action)
        elif(action > 0):  # if action is right
            # accelerating right is bad (کند شوند)
            feature_accelerating = -1*(action*action)
    return feature_accelerating


def computeQValueFromWeights(state, a, feature_weights):
    """
        Returns Q(state,a) = w * featureVector
        where * is the dotProduct operator
    """
    # features = Counter()
    # features["accelerating"] = 1 if a == 2 else 0
    # features["decelerating"] = 1 if a == 0 else 0
    # features["coasting"] = 1 if a == 1 else 0
    # features["position"] = state[0]
    # features["velocity"] = state[1]
    # return features * feature_weights

    feature_accelerating = computeFeatureAccelarating(state, a)
    # feature_position = state[0]
    # feature_velocity = state[1]
    # return feature_accelerating*feature_weights["accelerating"] + feature_position*feature_weights["position"] + feature_velocity*feature_weights["velocity"]
    # return feature_accelerating*feature_weights["accelerating"] + feature_position*feature_weights["position"]
    # return feature_accelerating*feature_weights["accelerating"] + state[0]*feature_weights["position"]
    return feature_accelerating*feature_weights["accelerating"]


def computeActionFromQValues(state, legalActions, feature_weights):
    """
        Returns the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
    """
    action = None
    maxQ = float("-inf")
    for a in legalActions:
        q = computeQValueFromWeights(state, a, feature_weights)
        if q > maxQ:
            maxQ = q
            action = a
    return action


def getAction(state, legalActions, epsilon, feature_weights):
    """
        Compute the action to take in the current state, including exploration.
        With probability self.epsilon, we should take a random action and
    """
    action = None
    if len(legalActions) == 0:  # terminal state
        return action

    epsilon_greedy = flipCoin(epsilon)
    if epsilon_greedy:  # probability of epsilon to take a random action
        action = random.choice(legalActions)
    else:
        action = computeActionFromQValues(state, legalActions, feature_weights)

    return action


def computeValueFromQValues(nextState, legalActions, feature_weights):
    """
        Returns max_action Q(nextState,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
    """
    if len(legalActions) == 0:
        return 0.0
    maxQ = float("-inf")
    for a in legalActions:
        q = computeQValueFromWeights(nextState, a, feature_weights)
        if q > maxQ:
            maxQ = q
    return maxQ


def update(feature_weights, state, action, nextState, reward, discount, alpha, legalActions):
    """
        Updates the feature weights based on transition
    """
    difference = (reward + discount * computeValueFromQValues(nextState, legalActions,
                                                              feature_weights)) - computeQValueFromWeights(state, action, feature_weights)
    if (difference > 10000000):
        difference = 10000000
    if (difference < -10000000):
        difference = -10000000
    for key in feature_weights.keys():
        if key == "accelerating":
            # feature_weights[key] += alpha * difference * \
            #     computeFeatureAccelarating(state, action) # wi = wi + alpha * (reward + discount * max_a Q(s',a) - Q(s,a)) * feature(s,a)
            feature_weights[key] -= alpha * difference * \
                computeFeatureAccelarating(
                    state, action)  # wi = wi + alpha * (reward + discount * max_a Q(s',a) - Q(s,a)) * feature(s,a)
        # elif key == "position":
        #     feature_weights[key] += alpha * difference * state[0]
        # elif key == "velocity":
        #     feature_weights[key] += alpha * difference * state[1]

        # if key == "position":
        #     feature_weights[key] += alpha * difference * state[0]
        # elif key == "velocity":
        #     feature_weights[key] += alpha * difference * state[1]
        # elif key == "accelerating":
        #     feature_weights[key] += alpha * difference * (1 if action == 2 else 0)
        # elif key == "decelerating":
        #     feature_weights[key] += alpha * difference * (1 if action == 0 else 0)
        # elif key == "coasting":
        #     feature_weights[key] += alpha * difference * (1 if action == 1 else 0)


def Approximate_Q_learning(env, num_episodes: int, discount: float, alpha: float, epsilon: float, epsilon_decay: float, alpha_decay, feature_weights):
    """
        Returns:
            Qvalues: A dictionary that maps from (state,action) pairs
            to Q values.
    """
    chart_data_time = []
    chart_data_reward = []
    chart_data_action = []
    for episode in range(num_episodes):
        observation, info = env.reset()
        state = observation
        terminated = False
        truncated = False
        i = 0
        while (not terminated) and (not truncated):
            i += 1
            # making actions discrete
            legalActions = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -
                            0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            action = getAction(state, legalActions, epsilon, feature_weights)
            observation, reward, terminated, truncated, info = env.step(
                [action])  # nextState, reward, done, _ = env.step(action)
            nextState = observation
            update(feature_weights, state, action, nextState,
                   reward, discount, alpha, legalActions)
            state = nextState

            # data for drawing chart
            if (episode == num_episodes - 1):
                chart_data_time.append(i)
                chart_data_reward.append(reward)
                chart_data_action.append(action)

            if (terminated):
                print("Episode {} finished after {} timesteps with epsilon {} and alpha {} and weights {}".format(
                    episode, i, epsilon, alpha, feature_weights))

        epsilon *= epsilon_decay  # decay epsilon
        alpha *= alpha_decay  # decay alpha

    draw_3D_chart(chart_data_time, chart_data_reward, chart_data_action)

    return feature_weights


# env = gym.make("MountainCarContinuous-v0",render_mode="human")
# env = gym.make("MountainCarContinuous-v0")


epsilon = 0.9  # could be lower as the agent learns (primitive tests = 0.9)
# epsilon = 0.1 # could be lower as the agent learns
alpha = 0.3  # learning rate (start: 0.8)
discount = 0.9  # discount factor for future rewards (gamma)
feature_weights_dictionary = {"accelerating": 1}
# feature_weights_dictionary = {"accelerating": 1, "position": 1}
# feature_weights_dictionary = {"accelerating": 1, "position": 1, "velocity": 1}
# feature_weights_dictionary = {"position": 1, "velocity": 1, "accelerating": 1, "decelerating": 1, "coasting": 1}
# comment after you remove file and it is your first try
with open(r'ApproximateQLearning_dictionary.txt', 'r') as f:
    feature_weights_dictionary = ujson.load(f)

feature_weights = Counter(feature_weights_dictionary)

env = gym.make("MountainCarContinuous-v0", render_mode='rgb_array')
env = gym.wrappers.RecordVideo(
    env, 'video', step_trigger=lambda x: x <= 1000, name_prefix='output', video_length=0)

observation, info = env.reset(seed=42)
# learned_feature_weights = Approximate_Q_learning(env, 100, discount, alpha, epsilon, 0.99, 0.999, feature_weights) # for training in 7 episodes
learned_feature_weights = Approximate_Q_learning(
    env, 5, discount, 0.01, 0.0001, 0.99, 0.999, feature_weights)  # for testing
# # save Qvalues in a file
feature_weights_dictionary = dict(learned_feature_weights)
# with open('ApproximateQLearning_dictionary.txt', 'w') as file:
#     file.write(ujson.dumps(feature_weights_dictionary))

env.close()
