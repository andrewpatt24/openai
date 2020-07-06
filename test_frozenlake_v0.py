from openai.simulator import EpisodeSimulator
from openai.agents.qlearning import RandomQLearner, EpsilonGreedyTabularQLearner
from openai.parameters import Epsilon
import gym
import tqdm
import numpy as np

env = gym.make('FrozenLake-v0', is_slippery=False)

# agent = RandomQLearner(
#     action_space=env.action_space,
#     state_space=env.observation_space
#     )

agent = EpsilonGreedyTabularQLearner(
    action_space=env.action_space,
    state_space=env.observation_space,
    decay_rate=0.0001,
    max_epsilon=0.8
)

sim = EpisodeSimulator(env, agent, max_timesteps=100)

episodes = 10000

final_reward_list = []

for i in tqdm.tqdm(range(episodes)):
    sim.simulate_episode()
    final_reward_list.append(np.sum(sim.reward_list))
    e_counter +=1

pass_percentage = round(float(sum(final_reward_list))*100. / float(episodes), 2)
print(f'Reward met {pass_percentage}% times')
# print(agent.q)

# sim.update_agent=False



exploit_episodes = 2000
exploit_reward_list = []
for i in tqdm.tqdm(range(exploit_episodes)):
    sim.simulate_episode()
    #print(agent.q)
    #print(sim.state_list)
    exploit_reward_list.append(np.sum(sim.reward_list))

# print(np.mean(exploit_reward_list[:500]), np.mean(exploit_reward_list[exploit_episodes-500:]))

pass_percentage = round(float(sum(exploit_reward_list))*100. / float(exploit_episodes), 2)
print(f'Reward met in exploitation {pass_percentage}% times')
# print(agent.q)
