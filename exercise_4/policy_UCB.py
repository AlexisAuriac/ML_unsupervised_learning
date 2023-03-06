import atexit
import math
import random

import numpy as np

from agent import Agent

MAX_REWARD = 100
MEAN_REWARD = (MAX_REWARD - 1) / 2
STD_REWARD = math.sqrt((MAX_REWARD**2 - 1) / 12) # https://testbook.com/question-answer/the-standard-deviation-of-the-first-n-natu--5ee75da08004d10d128bc6d3

thresholds = list(range(20, 100, 10))
nb_treshold = len(thresholds)


def normalize_reward(reward):
	return (reward - MEAN_REWARD) / STD_REWARD


def rewards_reset(rewards):
	return all(rewards == 0)


def policy(agent: Agent) -> str:
	curr_action = policy.curr_action
	threshold_used = policy.threshold_used
	nb_selections = policy.nb_selections
	reward_sum = policy.reward_sum
	total_reward = policy.total_reward
	n = policy.n
	prev_thres_i = policy.prev_thres_i

	## Apply reward
	if n > 0:
		if not rewards_reset(agent.known_rewards):
			reward = normalize_reward(agent.known_rewards[agent.position])
		else:
			reward = normalize_reward(total_reward / n)
		reward_sum[prev_thres_i] = reward_sum[prev_thres_i] + reward
		policy.total_reward = total_reward + reward

	## Select threshold
	thres_i = 0
	max_upper_bound = 0

	for i in range(nb_treshold):
		if nb_selections[i] > 0:
			average_reward = reward_sum[i] / nb_selections[i]
			delta_i = math.sqrt(3/2 * math.log(n + 1) / nb_selections[i])
			upper_bound = average_reward + delta_i
		else:
			upper_bound = 1e400
		if upper_bound > max_upper_bound:
			thres_i = i
			max_upper_bound = upper_bound

	threshold_used.append(thres_i)
	nb_selections[thres_i] = nb_selections[thres_i] + 1
	policy.prev_thres_i = thres_i
	policy.n += 1

	## Decide to move or stay put
	if agent.known_rewards[agent.position] < thresholds[thres_i]:
		if policy.curr_action == 'left':
			if agent.position == 0:
				policy.curr_action = 'right'
		elif policy.curr_action == 'right':
			if agent.position == 7:
				policy.curr_action = 'left'
		else:
			if agent.position == 0:
				policy.curr_action = 'right'
			else:
				policy.curr_action = 'left'
	else:
		return 'none'

	return policy.curr_action


# https://stackoverflow.com/a/279586/12864941
policy.curr_action = 'left'
policy.threshold_used = []
policy.nb_selections = [0] * nb_treshold
policy.reward_sum = [0] * nb_treshold
policy.total_reward = 0
policy.n = 0
policy.prev_thres_i = 0


# https://stackoverflow.com/a/3850271/12864941
def exit_handler():
	average_rewards = np.array(policy.reward_sum) / policy.nb_selections
	print(f'average rewards: {average_rewards}')
	best_thres_i = np.argmax(average_rewards)
	print(f'best threshold: {thresholds[best_thres_i]}, avg_reward={average_rewards[best_thres_i]:.2f}, selected {policy.nb_selections[best_thres_i]} times')

atexit.register(exit_handler)
