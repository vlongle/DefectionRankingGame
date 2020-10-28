'''
This is the S-exp3 algo.
Reward should be scaled to [0, 1]


REFERENCE:
https://jeremykun.com/2013/11/08/adversarial-bandits-and-the-exp3-algorithm/
'''
import numpy as np

from code.MonteCarlo import MC_simulation
class S_exp3:
    def __init__(self, game, policies, agent, n_episodes, gamma):
        '''
        FUNCTIONALITY: keeping every other agent's policies fixed, optimize
        an AI agent using exp3 algorithm.

        For each policy state involves AI agent, we run a (correlated) exp3
        algorithm.

        :param game:
        :param policies: initial policies. Target agent's policy
        would be updated
        :param agent: AI agent to run exp3 on
        :param n_episodes: how many game runs
        :param gamma: in [0, 1]. High gamma ==> more exploration
        '''
        self.num_actions = 2
        self.weights = {PS:[1]*self.num_actions for PS in game.policyStates} # two actions: leave or stay
        self.gamma = gamma
        self.n_episodes  = n_episodes
        self.agent = agent
        self.game = game
        self.policies = policies


    def optimize_agent(self):
        for t in range(self.n_episodes):
            print('\n==== episode {} ===='.format(t))
            Gt, visited_PS, choices = MC_simulation(self.game, self.policies)
            terminal_state = Gt[-1]
            self.update_weight(visited_PS, choices, terminal_state)


    def dist(self, weights):
        # convert self.weights to a prob. dist
        normalizing = sum(weights)
        return [(1-self.gamma) * (w/normalizing) + (self.gamma/self.num_actions)\
                for w in weights]

    def update_weight(self, policy_states, choices, terminal_state):
        '''
        FUNCTIONALITY:
            Given an MC episode, we go through all policy states and pick those that involve the AI agent.
            Then from those, we get the rewards and the choices that the agent make. We update the weights
            and agent's policy by exp3 algorithm
        :param policy_states:
        :param choices:
        :return:
        '''
        for PS, actions in zip(policy_states, choices):
            if self.agent not in PS.coalition_considered:
                continue
            probs = self.policies[self.agent][PS.state_num]
            agent_choice = actions[self.agent] # 0 = leave, 1 = stay
            print('\n>> At {} >> choice {}'.format(PS, agent_choice))
            reward = self.scaled_reward(terminal_state)
            print('reward for {} is {}'.format(self.agent, reward))
            unbiased_reward = reward / probs[agent_choice]
            # update the weights
            self.weights[PS][agent_choice] *= np.exp(unbiased_reward * self.gamma / self.num_actions)
            # update policy
            print('policy changed from', self.policies[self.agent][PS.state_num], end=' ')
            self.policies[self.agent][PS.state_num] = self.dist(self.weights[PS])
            print('to ', self.policies[self.agent][PS.state_num])



    def scaled_reward(self, terminal_state):
        '''
        :param terminal_state:
        :return: scaled reward to [0, 1]. A reward is the final standing of the AI agent
        during this MC episode.
        '''
        final_ranking = terminal_state.evaluate_leaf(ret_delta=False)
        print('final_ranking:', final_ranking)
        return final_ranking[self.agent.name]/sum(final_ranking.values())



