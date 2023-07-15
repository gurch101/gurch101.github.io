---
title: Reinforcement Learning
date: 2022-05-29
description: Udemy course notes
category: summary
type: notes
---

Goal: train an agent to make decisions in an environment in order to maximize a reward signal (optimal policy). We are trying to learn a state -> action map.

Learns through exploration and exploitation of the environment.

The return is the sum of rewards the agent gets weighted by some discount factor. This captures temporal differences in rewards by multiplying the reward by a decreasing discount factor with gamma < 1. Higher gamma = agent is more patient.

```
Return = R1 + gammaR2 + gamma^2R3
```

A policy is a function that takes a state and returns the action to perform.

Markov decision process

- future only depends on current state, not on how you arrived at current state

agent takes actions to interact with environment -> receives feedback -> adjusts behavior.

State-action value (Q) function - Q(s,a) = the return if you start in state s, take action a, then behave optimally after that (max Q(s',a')). Goal is to pick the a that gives the largest Q(s,a).

### Bellman Equation

Q(s,a) = R(s) + gamma(maxa'(s',a'))

Q(s,a) = What you get right away + what you get later

### Stochastic Environments

need to account for random environments/probability of wrong behavior. Not looking to maximize return but maximizing the average value. Expected return = average(R1 + gammaR2 + gamma^2R3).

s' is now random in Bellman equation so take average reward of each action at s'.

### Continuous State Spaces

- can have many state variables represented as a vector that can take on a continuous range of values

### Example

Lunar lander

actions: do nothing, left thruster, main thruster, right thruster

State = [x,y,xdot,ydot,theta,theatadot,l(left leg sitting on ground),r(right leg sitting on ground)]

reward = 100 - 140 for getting to landing pad
additional reward for moving toward/away from pad
crash = -100
soft landing = +100
leg grounded = +10
fire main engine = -.3
fire side thruster = -.0.3

##### Learning the state-value function

Deep RL - 12 inputs (state + actions), 64 unit layer, 64 unit layer, 4 unit output Q(s,a) - one per state

Use the nn to compute Q(s, an) for all 4 actions. Pick the action a that maximizes Q(s,a)

To build up a dataset, try random actions and save states (x), compute reward and new state (y)

Deep Q Network Algorithm:

- initialize nn randomly as a guess of Q(s,a)
  repeat:
  - exploration step: take actions - pick action with probability X that maximizes Q(s,a) (exploitation), otherwise pick action randomly (exploration). episilon = greedy policy = 1 - exploration probability. Get (s,a,R(s),s').
    - episilon starts high (completely random) and decreases gradually with next training steps (greedy)
  - Store 10k most recent tuples (replay buffer).
  - create training set of 10k examples: x = (s,a), y = R(s) + gamma(max(Q(s',a'))). Y is just random initially.
    - experience replay: store the agent's states/actions/rewards in a memory buffer and sample mini-batch
    - since y is constantly changing, this leads to instability since MSE constantly varies since the target varies
    - use a separate nn for generating y targets
  - train Qnew such that Qnew(s,a) ~= y
  - set Q = Qnew

Mini-batch and soft updates

Every step of gradient descent requires taking the average over every training example. If the training set is large, this is slow.

Mini-batch:
take a subset of the data
for each iteration, choose a different subset
Fast but doesn't reliably compute the global minima

Soft updates:
Make a more gradual change when updating.
when setting Q = Qnew, do W = 0.01Wnew + 0.99W, B = 0.01Bnew + 0.99B
Increases reliability of convergence

```py
import time
from collections import deque, namedtuple

import gym
import numpy as np
import PIL.Image
import tensorflow as tf
import utils

from pyvirtualdisplay import Display
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# Set up a virtual display to render the Lunar Lander environment.
Display(visible=0, size=(840, 480)).start();

# Set the random seed for TensorFlow
tf.random.set_seed(utils.SEED)

# Hyperparameters
MEMORY_SIZE = 100_000     # size of memory buffer
GAMMA = 0.995             # discount factor
ALPHA = 1e-3              # learning rate
NUM_STEPS_FOR_UPDATE = 4  # perform a learning update every C time steps

# Each action has a corresponding numerical value:
# Do nothing = 0
# Fire right engine = 1
# Fire main engine = 2
# Fire left engine = 3

env = gym.make('LunarLander-v2')
initial_state = env.reset()
# show the first frame
PIL.Image.fromarray(env.render(mode='rgb_array'))

state_size = env.observation_space.shape
num_actions = env.action_space.n

# 8
print('State Shape:', state_size)
# 4
print('Number of actions:', num_actions)

# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, info = env.step(action)

with np.printoptions(formatter={'float': '{:.3f}'.format}):
    print("Initial State:", initial_state)
    print("Action:", action)
    print("Next State:", next_state)
    print("Reward Received:", reward)
    print("Episode Terminated:", done)
    print("Info:", info)

# Create the Q-Network
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
    ])

optimizer = Adam(learning_rate=ALPHA)

experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def compute_loss(experiences, gamma, q_network, target_q_network):
    """
    Calculates the loss.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.
      q_network: (tf.keras.Sequential) Keras model for predicting the q_values
      target_q_network: (tf.keras.Sequential) Karas model for predicting the targets

    Returns:
      loss: (TensorFlow Tensor(shape=(0,), dtype=int32)) the Mean-Squared Error between
            the y targets and the Q(s,a) values.
    """

    # Unpack the mini-batch of experience tuples
    states, actions, rewards, next_states, done_vals = experiences

    # Compute max Q^(s,a)
    max_qsa = tf.reduce_max(target_q_network(next_states), axis=-1)

    # Set y = R if episode terminates, otherwise set y = R + γ max Q^(s,a).
    ### START CODE HERE ###
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    ### END CODE HERE ###

    # Get the q_values
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))

    # Compute the loss
    ### START CODE HERE ###
    loss = MSE(y_targets, q_values)
    ### END CODE HERE ###

    return loss

@tf.function
def agent_learn(experiences, gamma):
    """
    Updates the weights of the Q networks.

    Args:
      experiences: (tuple) tuple of ["state", "action", "reward", "next_state", "done"] namedtuples
      gamma: (float) The discount factor.

    """

    # Calculate the loss
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)

    # Get the gradients of the loss with respect to the weights.
    gradients = tape.gradient(loss, q_network.trainable_variables)

    # Update the weights of the q_network.
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

    # update the weights of target q_network
    utils.update_target_network(q_network, target_q_network)
```

### Limitations

- Most research has been in simulations. Much harder to get working in real world
- Far fewer applications than supervised/unsupervised learning

### Usage

- game playing
- teach robots
- autonomous driving
- recommendation systems

### Questions

What happens if you don't know the terminal states? is this considered unsupervised RL?
How do you come up with reward values?

### Goals

Use RL to solve NES Punch Out
