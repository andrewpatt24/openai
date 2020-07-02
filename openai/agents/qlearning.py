import numpy as np


class RandomQLearner:

    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def get_action(self, state):
        return self.env.action_space.sample()

    def update(self, state, reward):
        pass


class EpsilonGreedyTabularQLearner:

    def __init__(self, env, discount_factor=0.99, learning_rate=0.8, epsilon=0.8):
        self.env = env
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        self.q = np.ones([self.env.observation_space.n, self.env.action_space.n])

        self.curr_state = self.env.reset()
        self.last_action = None

    def get_action(self, state):
        # Epsilon greedy action selection
        if np.random.random() < self.epsilon:
            # Greedy
            self.last_action = np.argmax(self.q[state, :])
        else:
            # Random selection
            self.last_action = self.env.action_space.sample()
        return self.last_action

    def update(self, state, reward):
        learned_value = reward + self.discount_factor * np.max(self.q[state, :])
        self.q[state, self.last_action] += \
            self.learning_rate * (learned_value - self.q[self.curr_state, self.last_action])

        self.curr_state = state
