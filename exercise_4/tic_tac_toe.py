#!/bin/env python3

from typing import List

import numpy as np

"""
Example of a Reinforcment learning problem.
"""

Q = {}


def compute_reward(state: List[str]):
	winner = check_winner(state)
	if winner == 'X':
		return 1
	elif winner == 'O':
		return -1
	else:
		return 0


def q_learning(state: List[str], action_stack: List[int], alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.02):
	state_hash = str(state)
	valid_actions = get_valid_actions(state)

	if state_hash not in Q or np.random.rand() < epsilon:
		action = np.random.choice(valid_actions)
	else:
		action = max(filter(lambda x: x[0][0] in valid_actions, np.ndenumerate(Q[state_hash])), key=lambda x: x[1])[0][0]
		# action = np.argsort(Q[state_hash][valid_actions])[-1]
	
	action_stack.append([state_hash, action])

	next_state, reward = take_action(state, action)
	next_state_hash = str(next_state)

	if state_hash not in Q:
		Q[state_hash] = np.zeros(9)
	if next_state_hash not in Q:
		Q[next_state_hash] = np.zeros(9)

	Q[state_hash][action] += alpha * (reward + gamma * np.max(Q[next_state_hash]) - Q[state_hash][action])

	if reward != 0:
		# print(action_stack)
		# for i in range(len(action_stack) - 2, -1, -1):
		#     h1 = action_stack[i][0]
		#     a1 = action_stack[i][1]
		#     h2 = action_stack[i+1][0]
		#     Q[h1][a1] = (1 - alpha) * Q[h1][a1] + alpha * (reward + gamma * np.max(Q[h2]))
		action_stack = []

	return next_state, reward, action_stack


def take_action(state: List[str], action: int, player: str = 'X'):
	if state[action] != ' ':
		raise 'invalid action'

	next_state = list(state)
	next_state[action] = player
	reward = compute_reward(next_state)
	return tuple(next_state), reward


def get_valid_actions(state: List[str]):
	return [i for i, val in enumerate(state) if val == ' ']


def check_winner(state: List[str]):
	# Check rows
	for i in range(0, 9, 3):
		if state[i] == state[i+1] == state[i+2] and state[i] != ' ':
			return state[i]
	# Check columns
	for i in range(3):
		if state[i] == state[i+3] == state[i+6] and state[i] != ' ':
			return state[i]
	# Check diagonals
	if state[0] == state[4] == state[8] and state[0] != ' ':
		return state[0]
	if state[2] == state[4] == state[6] and state[2] != ' ':
		return state[2]
	if [i for i in range(9) if state[i] == ' '] == []:
		return 'D'
	# No winner
	return None


def get_opponent_action(state: List[str]):
	# for i in range(9):
	#     if state[i] == ' ':
	#         next_state, _ = take_action(state, i, player='O')
	#         if check_winner(next_state) == 'O':
	#             return i
	
	valid_moves = [i for i in range(9) if state[i] == ' ']
	return np.random.choice(valid_moves)


def play_game():
	state = [' '] * 9
	action_stack = []

	while True:
		state, r, action_stack = q_learning(state, action_stack)
		winner = check_winner(state)
		if winner is not None:
			return winner

		state, _ = take_action(state, get_opponent_action(state), player='O')
		winner = check_winner(state)
		if winner is not None:
			return winner


import matplotlib.pyplot as plt

NB_GAMES = 1000
win_history = [0]
wins = {'X': 0, 'O': 0, 'D': 0}
for i in range(NB_GAMES):
	winner = play_game()
	wins[winner] += 1

	if winner == 'X':
		win_history.append(win_history[-1] + 1)
	elif winner == 'O':
		win_history.append(win_history[-1] - 1)
	else:
		win_history.append(win_history[-1])

win_history = win_history[1:]

# plt.plot(np.array(win_history) / np.arange(1, 1001))
# plt.show()
print(wins)
print(f'win rate:   {wins["X"] / NB_GAMES * 100:.2f}% ({wins["D"] / NB_GAMES * 100:.2f}% draw)')
print(f'loose rate: {wins["O"] / NB_GAMES * 100:.2f}%')
