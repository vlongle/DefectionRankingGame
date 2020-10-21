from code import *
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pprint import pprint
import numpy as np
from itertools import product



## Players
A = Player.Player('A', 3)
B = Player.Player('B', 2)
C = Player.Player('C', 1)
N = set([A, B, C])


## Game defn
CS0 = CoalitionStructure.CoalitionStructure(CS=[], Z=[])
G0 = State.State(N, CS0, t=0)
graph = nx.DiGraph()
T = 2


## build Game
game = Game.Game(G0, graph, T)
game.build_graph()


## Graph!
#plt.rcParams['figure.figsize'] = 30, 10
#pos =graphviz_layout(game.graph, prog='dot')
#nx.draw(game.graph, pos,alpha=0.5, node_size=50, arrowsize=5, arrows=True)
#plt.show()

## list policy states!
pprint(game.policyStates)
print()
# evaluate the leaf nodes!
#for state in game.nodes[T]:
#    game.valuations[state] = state.evaluate()

#Policy.constructNashGame(game, policy_state)


policies = {} # key = player, value = policy
# policy = np array
for agent in N:
    policies[agent] = 0.5 * np.ones(shape=(len(game.policyStates), 2))

n_samples = 100

def softmax(raw_policy, alpha=1):
    # policy is a 2d matrix
    # [state][action_probability]
    # e.g. num_states = 3
    # [[0.5, 0.5],
    # [0.7, 0.3],
    # [0.3, 0.8]]
    # we apply softmax transform along every row to make it a prob. distribution over
    # actions per state
    # alpha is an acceleration factor. If alpha is large then we prioritize
    # exploitation over exploration!
    raw_policy = np.array(raw_policy)
    transformed = np.exp(alpha * raw_policy)
    normalizing_const = np.sum(transformed, axis=-1, keepdims=True)
    return transformed/normalizing_const


def eval_states(game, policies):
    print('eval_states!!')
    T = game.T
    for t in range(T, -1,-1):
        print('t:', t)
        for state in game.nodes[t]:
            if t == T:
                game.valuations[state] = state.evaluate_leaf()
                continue
            # bottom down eval

def eval_state(game, policy_state, policies):
    # should only called if the resulting states are already defined
    # return the average outcome for each agent over this state
    # using the "policies"
    value = {}

    possible_outcomes = product(range(2), repeat=len(C))
    state_payout = np.array([0]*len(C))
    for outcome in possible_outcomes:
        p = 1
        outcome_payout = ...
        for k, agent in enumerate(C):
            action = outcome[i]
            p *= policies[agent][policy_state.state_num, action]
        # for each outcome, we should evaluate
    return value

def optimize_agent(agent, policies, game):
    # optimize this agent fixing everyone else's policy!
    for policy_state in game.policyStates[::-1]: # from bottom to top
        #print('\n ++ policy_state:', policy_state)
        # optimize this policy_state!
        C = policy_state.coalition_considered
        if agent not in C:
            continue
        #print('agent_id:', agent_id)
        my_policy_in_ps = policies[agent][policy_state.state_num]
        policies[agent][policy_state.state_num] = [1, 1]

        #possible_outcomes = product(range(2), repeat=len(C))

        action_values = [0, 0] # [leave, not_leave] i.e. (0, 1)

        payout = Policy.eval_policy_state(game, policy_state, policies)[agent]
        for action in [0, 1]: # leave or not leave
            np.take(payout, [action], axis=agent_id)
            pass

        #for outcome in possible_outcomes:
        #    my_action = outcome[agent_id]
        #    p = 1
        #    for i, other_agent in enumerate(C):
        #        if other_agent != agent:
        #            other_action = outcome[i]
        #            p *= policies[other_agent][policy_state.state_num, other_action]
        #    #print('PAY:', payout[tuple([agent_id] + list(outcome))])
        #    #print('my_action:', my_action)
        #    action_values[my_action] += p * payout[tuple([agent_id] + list(outcome))]
        #    #print('value assigning:', p * payout[tuple([agent_id] + list(outcome))])
        #    #print('action_values:', action_values)
        # change policy now!

        probs = softmax(action_values, 3)
        eps = 0.05
        if (abs(probs - my_policy_in_ps) > eps).any() :
            #print(probs != policies[agent][policy_state.state_num])
            #print(probs[0], policies[agent][policy_state.state_num][0], probs[0] != policies[agent][policy_state.state_num][0])
            print('=== changing', agent, 'in state', policy_state, 'to:',
                  probs, 'from action_values:', action_values)
                  #, 'from old policies:', policies[agent][policy_state.state_num])
        policies[agent][policy_state.state_num] = probs
        # change exactly proportional to action_values. I mean I could
        # have take the max but whatever...

print('\n\n >>> Optimizing agent')
for i in range(n_samples):
    eval_states(game, policies)
    for agent in N:
        optimize_agent(agent, policies, game)

#optimize_agent(C, policies, game)
