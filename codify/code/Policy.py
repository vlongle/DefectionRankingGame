from itertools import product
from code.CoalitionStructure import CoalitionStructure
from code.State import State
import numpy as np
from collections import defaultdict
from code.MonteCarlo import next_game_state

def eval_states(game, policies):
    # using backward induction, evaluate every game state
    # using the policies
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
            #print('++ state', state, '\n \t val:', game.valuations[state])
            #, 'B VAL:', game.valuations[state]['B'])
            #print('poli_states:', state.policyStates)

            #if state == G0:
            #    print('G0!')
            #    for ps in state.policyStates:
            #        print('ps:', ps, 'val:', policy_state_payoff)



def eval_policy_state(game, policy_state, policies=None):
    '''
    Let k be the number of agents in the coalitions!
    We have |A|^k=2^k possible outcomes. Each outcome leads
    to another game state.

    Input: policy_state
    Output: payout
    Payout ordering of agent is in the same ordering as policy_state.coalition_considered.

    This function only works if the resulting game states have their valuations dict
    written before.
    '''
    #print('\nConstructing Game for', policy_state, end='\n')
    c = policy_state.coalition_considered
    k = len(c)
    n = len(game.N)
    payout = np.zeros(shape=[n] + [2] * k)  # shape=(n,2,...,2)
    #payout = np.zeros(shape= [2] * k)  # shape=(2,...,2)
    #payout = defaultdict(list)
    # 1 == stay, 0 == leave
    possible_outcomes = product(range(2), repeat=k)


    for outcome in possible_outcomes:

        action_chosen = {}
        for agent, action in zip(c, outcome):
            action_chosen[agent] = action

        new_state = next_game_state(policy_state, action_chosen)

        p = 1
        if policies:
            for i, agent in enumerate(c):
                action = outcome[i]
                p *= policies[agent][policy_state.state_num, action]

        valuations = game.valuations[new_state]
        print('valuation:', valuations, 'for outcome', outcome, 'new_state', new_state)
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

def optimize_agent(agent, policies, game, policy_type="prop"):
    '''
    :param agent: agent to otpimize
    :param policies:
    :param game:
    :param policy_type: "max" or "prop" or "random"
    :return: if this agent has at least one policy state deviated
    signficantly afterwards from policies as the result of this function.
    '''
    #print("===== OPT AGENT:", agent)
    ret = False
    # optimize this agent fixing everyone else's policy!
    for policy_state in game.policyStates[::-1]: # from bottom to top
        #print('\n ++ policy_state:', policy_state)
        # optimize this policy_state!
        C = policy_state.coalition_considered
        if agent not in C:
            continue
        my_policy_in_ps = np.copy(policies[agent][policy_state.state_num])
        # change the policy temporarily here so that eval_policy_state function
        # works!
        policies[agent][policy_state.state_num] = [1, 1]

        agent_id = game.N.index(agent)
        payout = eval_policy_state(game, policy_state, policies)[agent_id]

        if policy_type == "prop":
            action_values = []
            for action in [0, 1]: # leave or stay
                action_values.append(np.sum(np.take(payout, [action], axis=C.index(agent))))
            probs = softmax(action_values, 3)
            #print('\n ++ policy_state:', policy_state)
            #print("action_values:", action_values, probs)
        elif policy_type == "max":
            leave_payoff = np.sum(np.take(payout, [0], axis=C.index(agent)))
            stay_payoff = np.sum(np.take(payout, [1], axis=C.index(agent)))
            if leave_payoff > stay_payoff:
                probs = [1, 0]
            else:
                probs = [0, 1]
        elif policy_type == "random":
            action_values = np.random.normal(0, 1, size=(2))
            probs = softmax(action_values, 3)



        eps = 0.05
        if (abs(probs - my_policy_in_ps) > eps).any() :
            ret = True
            #print('\n ++ policy_state:', policy_state)
            #print('=== changing', agent, 'in state', policy_state, 'to:',
            #      probs, 'from action_values:', action_values, 'from old:', my_policy_in_ps)
            #      #, 'from old policies:', policies[agent][policy_state.state_num])

        policies[agent][policy_state.state_num] = probs
    return ret
