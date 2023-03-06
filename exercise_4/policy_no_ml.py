import random

from agent import Agent

def policy(agent: Agent) -> str:
	if agent.known_rewards[agent.position] < 40:
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
