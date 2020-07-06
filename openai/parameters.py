import numpy as np
from openai.base import BaseEpsilon


class FixedEpsilon(BaseEpsilon):

    def __init__(self, max_epsilon=None, min_epsilon=None, decay_rate=None, epsilon=None):
        super().__init__(max_epsilon, min_epsilon, decay_rate, epsilon)

    def __call__(self):
        return self.epsilon

    def reset(self):
        pass


class ExponentialEpsilon(BaseEpsilon):

    def __init__(self, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.01, epsilon=None):
        super().__init__(max_epsilon, min_epsilon, decay_rate, epsilon)

        self.counter = 0

    def __call__(self):
        new_epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate*self.counter)
        self.counter += 1
        self.epsilon = new_epsilon
        return new_epsilon

    def reset(self):
        self.counter = 0
        self.epsilon = None


def Epsilon(max_epsilon=1.0, min_epsilon=0.01, decay_rate=None, epsilon=None):

    if decay_rate is not None:
        if epsilon is not None:
            raise AssertionError('Both decay_rate and epsilon are specified. Pick one.')
    elif epsilon is None:
        raise AssertionError('Please specify either decay_rate or epsilon')

    if decay_rate is not None:
        epsilon_function = ExponentialEpsilon
    else:
        epsilon_function = FixedEpsilon

    return epsilon_function(max_epsilon, min_epsilon, decay_rate, epsilon)
