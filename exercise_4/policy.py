import random

import numpy as np

from agent import Agent

K = 4
K_FACTOR = 100 // K

LEFT = 0
RIGHT = 1
NONE = 2
ACTIONS = ['left', 'right', 'none']

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

def rewards_reset(rewards):
	return all(rewards == 0)

def get_valid_actions(agent):
	if agent.position == 0:
		return [RIGHT, NONE]
	elif agent.position == len(agent.known_rewards) - 1:
		return [LEFT, NONE]
	else:
		return [LEFT, RIGHT, NONE]

def select_action(action_values, valid_actions):
	return max(filter(lambda x: x[0][0] in valid_actions, np.ndenumerate(action_values)), key=lambda x: x[1])[0][0]

def policy(agent: Agent) -> str:
	Q = policy.Q
	state_hash = str((agent.position, agent.known_rewards // K_FACTOR))
	valid_actions = get_valid_actions(agent)

	if state_hash not in Q or np.random.rand() < EPSILON:
		action = np.random.choice(valid_actions)
	else:
		action = select_action(Q[state_hash], valid_actions)

	if state_hash not in Q:
		Q[state_hash] = np.zeros(3)

	if rewards_reset(agent.known_rewards) == False and policy.prev_state != None:
		reward = agent.known_rewards[agent.position]
		Q[policy.prev_state][policy.prev_action] += ALPHA * (reward + GAMMA * np.max(Q[state_hash]) - Q[policy.prev_state][policy.prev_action])

	policy.prev_state = state_hash
	policy.prev_action = action

	return ACTIONS[action]


# https://stackoverflow.com/a/279586/12864941
policy.prev_state = None
policy.prev_action = None
policy.Q = {}
policy.i = 0
