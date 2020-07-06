import numpy as np
from openai.parameters import Epsilon
from openai.base import BaseLearner


class EpsilonGreedyTabularQLearner(BaseLearner):

    def __init__(self, action_space, state_space, discount_factor=0.99, learning_rate=0.8,
                 max_epsilon=1.0, min_epsilon=0.01, decay_rate=None, epsilon=None):

        self.action_space = action_space
        self.state_space = state_space

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = Epsilon(max_epsilon=max_epsilon, min_epsilon=min_epsilon, decay_rate=decay_rate, epsilon=epsilon)

        self.q = np.zeros([self.state_space.n, self.action_space.n])
        self.last_action = None
        self.curr_state = None

    def reset_state(self, state):
        self.curr_state = state

    def get_action(self, state):
        self.curr_state = state
        # Epsilon greedy action selection
        if np.random.uniform(0, 1) > self.epsilon():
            # Greedy
            self.last_action = np.argmax(self.q[state, :])
        else:
            # Random selection
            self.last_action = self.action_space.sample()
        return self.last_action

    def update(self, state, reward):
        curr_q = self.q[self.curr_state, self.last_action]
        next_max = np.max(self.q[state])
        learned_value = reward + self.discount_factor * next_max
        new_q = curr_q + self.learning_rate * (learned_value - curr_q)
        self.q[self.curr_state, self.last_action] = new_q
