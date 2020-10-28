import numpy as np
from code.CoalitionStructure import CoalitionStructure
from code.State import State

def draw_policy_state(game_state):
    # draw uniformly random from policyStates
    if not game_state.policyStates:
        #print('{} has no policy states!'.format(game_state))
        return 'NA' # the free set is empty! ==> no more players left!
    return np.random.choice(game_state.policyStates)


def agent_act(policy_state, policies):
    C = policy_state.coalition_considered
    action_chosen = {} # key = agent, value = 0/1
    for agent in C:
        # [leave, stay] | action = 0 ==> leave | action = 1 ==> stay
        action = np.random.choice(range(2), 1, p=policies[agent][policy_state.state_num])[0]
        action_chosen[agent] = action
    return action_chosen

def next_game_state(game, policy_state, action_chosen):
    c = policy_state.coalition_considered
    s = policy_state.parentState

    z = 1 # 0 if the group fail, 1 if the group succeeds (i.e. no betrayers)
    betrayers = set()
    for agent, action in action_chosen.items():
        if not action: # if leave
            z = 0
            betrayers.add(agent)

    resulting_C = set(c).difference(betrayers)
    F_new = s.F.difference(resulting_C)
    if resulting_C:
        CS_new = CoalitionStructure(s.CS.CS + [resulting_C], s.CS.Z + [z])
    else:
        CS_new = CoalitionStructure(s.CS.CS, s.CS.Z)  # no plus
    new_state = State(F_new, CS_new, s.t + 1)
    return game.states_lookup[new_state.name]

def MC_simulation(game, policies):
    '''
    Carry out a run of this game using policies
    :param game:
    :return:
        - game states visited
        - policy states visited
        - actions of each agent in those policy states
    '''
    game_states_visited = [game.G0]
    policy_states_visited = []
    agent_choices = []

    for t in range(game.T):
        cur_game_state = game_states_visited[t]
        PS = draw_policy_state(cur_game_state)
        if PS == 'NA':
            break # no more free players. The game effectively terminates!
        action_chosen = agent_act(PS, policies)

        next_state = next_game_state(game, PS, action_chosen)

        policy_states_visited.append(PS)
        game_states_visited.append(next_state)
        agent_choices.append(action_chosen)


    return game_states_visited, policy_states_visited, agent_choices
