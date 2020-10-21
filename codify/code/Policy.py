from itertools import product
from code.CoalitionStructure import CoalitionStructure
from code.State import State
import numpy as np

def eval_policy_state(game, policy_state, policies=None):
    '''
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
    #payout = np.zeros(shape=[k] + [2] * k)  # shape=(k,2,...,2)
    #payout = np.zeros(shape= [2] * k)  # shape=(2,...,2)
    payout = {}
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

        valuations = p * game.valuations[new_state]
        # print('new_state:', new_state)
        #print('valuation:', valuations, 'for outcome', outcome)
        for i, agent in enumerate(c):
            payout[agent][tuple(outcome)] = valuations[agent.name]
    return payout