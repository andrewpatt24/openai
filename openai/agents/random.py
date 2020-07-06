from openai.base import BaseLearner


class RandomLearner(BaseLearner):

    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space

        self.curr_state = None

    def reset_state(self, state):
        self.curr_state = state

    def get_action(self, state):
        return self.action_space.sample()

    def update(self, state, reward):
        pass