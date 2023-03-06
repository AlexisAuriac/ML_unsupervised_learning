"""
Simple policy that doesn't use any ML.
Move left or right if the reward of the current position is inferior to a threshold
"""

import random

from agent import Agent

THRESHOLD = 40


def policy(agent: Agent) -> str:
	if agent.known_rewards[agent.position] < THRESHOLD:
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
