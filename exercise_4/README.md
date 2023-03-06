# Part 4: exploitation/exploration compromise

## Subject

### Setting

We consider a one dimensional world, with 8 possible positions, as defined in the folder ```project/exercise_4```. An agent lives in this world, and can perform one of 3 actions at each time step: stay at its position, move right or move left.

In this folder, you can find 3 files :
- ```simulation.py``` is the main file that you can run to evaluate a policy.
- ```agent.py``` defines the Agent class. This simple agent only has two attributes.
	- position: its position
	- known_rewards: represents the knowledge of the agent about the rewards in the worlds (see below)
- ```default_policy.py``` implements a default policy that consists in always going left.

Some rewards are placed in this world randomly, and are randomly updated perdiodically, at a fixed frequency. This means that a good agent should update its policy periodically as well and adapt to the new rewards. The agent knows about a reward in the world if its position has been on the same position as the reward, but each time the rewards are updated, the agents forgets all this knowledge, as implemented line 46 in ```simulation.py```.

```simulation.py``` computes the statistical amount of reward obtained by the agent and plots the evolution of this quantity in ```images/```. As you can see in the ```images/``` folder, the average accumulated reward with the default policy is around 16, with a little bit of variance.

### Objective

Write a different, stochastic policy in a separate file named ```<group_name>_policy.py``` that achieves a better performance than the default policy. ```<group_name>_``` should be the name of one of the students of your group, or any name that identifies your
group.

You will need to
- import you policy in simulation.py
- replace line 51 by a line that calls your policy instead of the default policy.

Your objective is to obtain a final average reward of at least 20.

Figure 1 (see pdf subject) - Convergence of the average reward obtained by the agent with the default policy.

## Main sources

[Simple introduction to reinforcement learning (UCB, Thomson Sampling)](https://www.kaggle.com/code/sangwookchn/reinforcement-learning-using-scikit-learn)

[Q-learning](https://en.wikipedia.org/wiki/Q-learning)

## Solution

<!-- https://stackoverflow.com/a/45508928/12864941 -->
**The main solution is [Q-learning](#q-learning) the rest is mostly there to document our process**

### No machine learning

We implemented a solution that doesn't use any machine learning, this was useful for understanding the rules of the game.

It chooses to stay if the current position has a reward superior to a certain threshold.

The code is in ```policy_no_ml.py```.

It gets an average accumulated reward of approximately 45 (with a threshold of 40).

![Policy no ML average accumulated reward](images/policy_no_ml.jpg?raw=true)

### UCB

(see policy_UCB.py)

This policy is based on the no ML one, it uses UCB to select the optimal threshold.

The best performing threshold varies but it is often 40.

Its results are basically the same as the version with a fixed threshold of 40.

![Policy UCB average accumulated reward](images/policy_UCB.jpg?raw=true)

### Q-learning

(see alexis-david_policy.py)

To understand the Q-learning algorithm we first applied it to a game of tic-tac-toe (see ```tic_tac_toe.py```).

On the simulation it does not work very well (average accumulated reward of around 21).

The algorithm learns the value of an action in a particular state, which posed a problem since each of the 8 position could have a value of 0 to 100 (it's a bit more complicated than that in reality, see ```reset_reward()``` in simulation.py), this means that there are 100^8 possible states (again, this is simplified). This is too many possible states, we would never encounter the same state twice, defeating the point of Q-learning.

To solve this issue we divide the reward into "reward levels", for number_of_levels=4 we have:
- 0-24 = 0
- 25-49 = 1
- ...

This allows us to drastically reduce the number of states.

At this point we had an average accumulated reward of around 21.

![Policy Q-learning average accumulated reward](images/policy_Q-learning.jpg?raw=true)

We then normalized the rewards to have a mean of 0 and a standard deviation of 1.

This improved our average total rewards to 25.

![Policy Q-learning with normalized rewards average accumulated reward](images/policy_Q-learning_normalized_rewards.jpg?raw=true)

Here are some other ideas of improvements that could be made:
- optimize the parameters (especially epsilon and the number of reward levels):
	- \+ easy to do
	- \- probably won't change the result significantly
- make the reward impact previous actions
	- \+ could improve exploration
	- \- actually made results worse when tried on tic-tac-toe
