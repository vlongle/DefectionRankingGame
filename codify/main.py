from code import *
import networkx as nx
from pprint import pprint
import numpy as np
from itertools import product
from collections import defaultdict
import time
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')


def plot_progress(N, result, put_label=False):
    for i, agent_name in enumerate(sorted([agent.name for agent in N])):
        plt.plot(result[:, i], '-^', label=agent_name)

        # https://queirozf.com/entries/add-labels-and-text-to-matplotlib-plots-annotation-examples
        if put_label:
            for x, y in zip(len(result[:, i]), result[:, i]):
                label = "{:.2f}".format(y)
                plt.annotate(label,  # this is the text
                             (x, y),  # this is the point to label
                             textcoords="offset points",  # how to position the text
                             xytext=(0, 10),  # distance from text to points (x,y)
                             ha='center')  # horizontal alignment can be left, right or center
    plt.legend()
    plt.ylabel('Delta ranking')
    #plt.xlabel('Game depth (T)')
    plt.xlabel('number of runs')
    plt.title('Alternating Optimization Trajectories')
    plt.show()



def print_policies(policies, game):
    for PS in game.policyStates:
        print(PS)
        for agent in game.N:
            if agent not in PS.coalition_considered:
                continue
            print(agent, policies[agent][PS.state_num])

if __name__ == '__main__':
    ## Players
    A = Player.Player('A', 3)
    B = Player.Player('B', 2)
    C = Player.Player('C', 1)
    D = Player.Player('D', -1)
    E = Player.Player('E', 0)
    #N = set([A, B, C, D, E])
    N = set([A, B, C])



    ## Paralell experiment stuff
    start = time.time()

    #delta_ranks = {agent.name:[] for agent in N}
    #a_pool = multiprocessing.Pool()


    #T_min = 1
    #T_max = 20
    #result = a_pool.map(alternating_opt_T, range(5, 10))
    #result = a_pool.map(alternating_opt_T, range(T_min,T_max))
    #result = np.array(result)

    #n_s_min = 1
    #n_s_max = 100
    #result = a_pool.map(partial(AlternatingPolicyOpt.alternating_opt_n_samples,
    #                            N), range(n_s_min,n_s_max))
    #result = np.array(result)


    n_samples = 20
    T = 2
    game, policies = Game.init_game(N, T)
    print(game.G0.policyStates)

    #print_policies(policies, game)
    #game.draw()

    #Policy.eval_states(game, policies)
    #print(game.valuations[game.G0])


    #for t in range(T+1):
    #    print(t, len(game.nodes[t]))
    #    pprint(game.nodes[t])

    result = AlternatingPolicyOpt.alternating_opt(game, policies, n_samples)
    result = np.array(result)
    #print("result:", result)
    #print("\n\n==== OPT POLICIES! ===")
    #print_policies(policies, game)
    plot_progress(N, result)
    end = time.time()
    print('take:', end-start)




