
class EpisodeSimulator:

    def __init__(self, env, agent, max_timesteps=100, update_agent=True):
        self.env = env
        self.agent = agent
        self.max_timestamps = max_timesteps
        self.update_agent = update_agent

        self.curr_state = None
        self.curr_reward = 0
        self.reward_list = []
        self.state_list = []

    def reset(self):
        self.curr_state = self.env.reset()
        self.curr_reward = 0

        self.reward_list = []
        self.state_list = []

        self.state_list.append(self.curr_state)
        self.reward_list.append(self.curr_reward)

        self.agent.reset_state(self.curr_state)

    def _iteration(self):

        # Get action from agent
        action = self.agent.get_action(state=self.curr_state)

        # take steps
        new_state, reward, done, info = self.env.step(action)

        # Update Agent
        if self.update_agent:
            self.agent.update(new_state, reward)

        # Cache Values
        self.reward_list.append(reward)
        self.state_list.append(new_state)
        self.curr_state = new_state
        self.curr_reward = reward

        return done, info

    def simulate_episode(self):

        self.reset()

        for t in range(self.max_timestamps):
            # self.env.render()
            done, info = self._iteration()

            if done:
                break



