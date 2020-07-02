
class Simulator:

    def __init__(self, env, agent, max_timesteps=100):
        self.env = env
        self.agent = agent
        self.max_timestamps = max_timesteps

        self.curr_state = None
        self.curr_reward = None
        self.reward_list = []
        self.state_list = []

    def reset(self):
        self.curr_state = self.env.reset()
        self.agent.curr_state = self.curr_state
        self.curr_reward = 0
        self.state_list.append(self.curr_state)

    def _iteration(self):

        # Get action from agent
        action = self.agent.get_action(state=self.curr_state)

        # take steps
        new_state, reward, done, info = self.env.step(action)

        # Update Agent
        self.agent.update(new_state, reward)

        # Cache Values
        self.reward_list.append(reward)
        self.state_list.append(new_state)
        self.curr_state = new_state
        self.curr_reward = reward

        return done, info

    def simulate(self):

        self.reset()

        for t in range(self.max_timestamps):
            done, info = self._iteration()

            if done:
                break
