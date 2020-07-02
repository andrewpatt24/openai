from openai.simulator import Simulator
from openai.agents.qlearning import RandomQLearner, EpsilonGreedyTabularQLearner
import gym
import tqdm

env = gym.make('FrozenLake-v0')

# agent = RandomQLearner(env)
agent = EpsilonGreedyTabularQLearner(env)

print(env.render())

sim = Simulator(env, agent, max_timesteps=100)

episodes = 1000

final_reward_list = []
for i in tqdm.tqdm(range(episodes)):
    sim.simulate()
    print(agent.q)
    final_reward_list.append(sim.curr_reward)

pass_percentage = round(float(sum(final_reward_list))*100. / float(episodes), 2)
print(f'Reward met {pass_percentage}% times')
# print(agent.q)
