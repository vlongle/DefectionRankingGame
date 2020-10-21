from code.Game import init_game
from code.Policy import eval_states
from code.Policy import optimize_agent

def alternating_opt(game, policies, n_samples):
    res = []
    G0 = game.G0
    print('\n\n >>> Optimizing agents')
    for i in range(n_samples):
        eval_states(game, policies)
        for agent in game.N:
            optimize_agent(agent, policies, game)
        d = game.valuations[G0]
        sorted_values = [d[agent_name] for agent_name in sorted(d.keys())]
        res.append(sorted_values)
    return res


def alternating_opt_T(N, T):
    game, policies = init_game(N, T=T)
    G0 = game.G0
    print('\n\n >>> Optimizing agents')
    n_samples = 100
    alternating_opt(game, policies, n_samples)
    # store result!
    d = game.valuations[G0]
    sorted_values = [d[agent_name] for agent_name in sorted(d.keys())]
    return sorted_values
    #for agent in game.N:
    #    delta_ranks[agent.name].append(dynamic[agent.name])



#def alternating_opt_n_samples(N, n_samples):
#    T = 10
#    game, policies = init_game(N, T=T)
#    G0 = game.G0
#    print('\n\n >>> Optimizing agents')
#
#    alternating_opt(game, policies, n_samples)
#    # store result!
#    d = game.valuations[G0]
#    sorted_values = [d[agent_name] for agent_name in sorted(d.keys())]
#    return sorted_values
