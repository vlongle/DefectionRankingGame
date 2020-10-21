from itertools import product
from code.CoalitionStructure import CoalitionStructure
from code.State import State
import numpy as np
from collections import defaultdict

def eval_states(game, policies):
    T = game.T
    for t in range(T, -1,-1):
        for state in game.nodes[t]:
            if t == T:
                game.valuations[state] = state.evaluate_leaf()
                #print('>> leaf', state, '\n \t val:', game.valuations[state])
                continue
            # bottom down eval
            state_value = defaultdict(float) # dictionary
            denom = len(state.policyStates)
            for ps in state.policyStates:
                policy_state_payoff = eval_policy_state(game, ps, policies)
                for i, agent in enumerate(game.N):
                    state_value[agent.name] += (1/denom) * np.sum(policy_state_payoff[i])
            game.valuations[state] = state_value
            #print('++ state', state, '\n \t val:', game.valuations[state], 'B VAL:', game.valuations[state]['B'])
            #print('poli_states:', state.policyStates)

            #if state == G0:
            #    print('G0!')
            #    for ps in state.policyStates:
            #        print('ps:', ps, 'val:', policy_state_payoff)



def eval_policy_state(game, policy_state, policies=None):
    '''
    Bug here! Payoff here should include everyone!!
    Let k be the number of agents in the coalitions!
    We have |A|^k=2^k possible outcomes. Each outcome leads
    to another game state.

    For the Gambit lib, we need to provide k matrices of payoff, each for
    each player. Each matrix is shape (2,2,...,2) where M[i_1, i_2, ..., i_k]
    is where each player plays the strategy i


    Input: policy_state
    Output: payout
    Payout ordering of agent is in the same ordering as policy_state.coalition_considered.

    This function only works if the resulting game states have their valuations dict
    written before.
    '''
    #print('\nConstructing Game for', policy_state, end='\n')
    c = np.array(policy_state.coalition_considered)
    k = len(c)
    n = len(game.N)
    payout = np.zeros(shape=[n] + [2] * k)  # shape=(n,2,...,2)
    #payout = np.zeros(shape= [2] * k)  # shape=(2,...,2)
    #payout = defaultdict(list)
    # 1 == stay, 0 == leave
    possible_outcomes = product(range(2), repeat=k)

    s = policy_state.parentState

    for outcome in possible_outcomes:
        outcome = np.array(outcome)
        # print('outcome:', outcome)
        z = int((outcome == 1).all())
        betrayers = set(c[np.where(outcome == 0)])  # 0 means player leaves!
        resulting_C = set(c).difference(betrayers)
        F_new = s.F.difference(resulting_C)
        if resulting_C:
            CS_new = CoalitionStructure(s.CS.CS + [resulting_C], s.CS.Z + [z])
        else:
            CS_new = CoalitionStructure(s.CS.CS, s.CS.Z)  # no plus
        new_state = State(F_new, CS_new, s.t + 1)

        p = 1
        if policies:
            for i, agent in enumerate(c):
                action = outcome[i]
                p *= policies[agent][policy_state.state_num, action]

        valuations = game.valuations[new_state]
        #print('valuation:', valuations, 'for outcome', outcome, 'new_state', new_state)
        for i, agent in enumerate(game.N):
            #print('doing something?')
            #print('p:', p, 'val:', valuations[agent.name], p * valuations[agent.name])
            payout[tuple([i] + list(outcome))] = p * valuations[agent.name] # bug here!!
            #payout[tuple([i] + list(outcome))] = p * valuations[agent] # bug here!!
            #if agent.name == 'A':
            #    print('A payoff:', payout[i])
    return payout

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

def optimize_agent(agent, policies, game):
    # optimize this agent fixing everyone else's policy!
    for policy_state in game.policyStates[::-1]: # from bottom to top
        #print('\n ++ policy_state:', policy_state)
        # optimize this policy_state!
        C = policy_state.coalition_considered
        if agent not in C:
            continue
        #print('agent_id:', agent_id)
        my_policy_in_ps = np.copy(policies[agent][policy_state.state_num])
        policies[agent][policy_state.state_num] = [1, 1]

        #possible_outcomes = product(range(2), repeat=len(C))

        agent_id = game.N.index(agent)
        action_values = [0, 0] # [leave, not_leave] i.e. (0, 1)

        payout = eval_policy_state(game, policy_state, policies)[agent_id]
        for action in [0, 1]: # leave or not leave
            action_values[action] = np.sum(np.take(payout, [action], axis=C.index(agent)))

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
        #eps = 0.05
        #if (abs(probs - my_policy_in_ps) > eps).any() :
            #print(probs != policies[agent][policy_state.state_num])
            #print(probs[0], policies[agent][policy_state.state_num][0], probs[0] != policies[agent][policy_state.state_num][0])
            #print('=== changing', agent, 'in state', policy_state, 'to:',
            #      probs, 'from action_values:', action_values, 'from old:', my_policy_in_ps)
                  #, 'from old policies:', policies[agent][policy_state.state_num])
        policies[agent][policy_state.state_num] = probs
        # change exactly proportional to action_values. I mean I could
        # have take the max but whatever...
