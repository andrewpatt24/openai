
class BaseParameter:
    def __init__(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class BaseLearner:

    def reset_state(self, state):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

    def update(self, state, reward):
        raise NotImplementedError


# noinspection PyMissingConstructor
class BaseEpsilon(BaseParameter):
    def __init__(self, max_epsilon=None, min_epsilon=None, decay_rate=None, epsilon=None):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.epsilon = epsilon
