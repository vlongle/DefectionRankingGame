from code import *
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
from pprint import pprint
import numpy as np
from itertools import product
from collections import defaultdict
import time
from functools import partial


plt.style.use('fivethirtyeight')





#N = set([A, B])



## Graph!
#plt.rcParams['figure.figsize'] = 30, 10
#pos =graphviz_layout(game.graph, prog='dot')
#nx.draw(game.graph, pos,alpha=0.5, node_size=50, arrowsize=5, arrows=True)
#plt.show()

## list policy states!
#pprint(game.policyStates)
#print()
# evaluate the leaf nodes!
#for state in game.nodes[T]:
#    game.valuations[state] = state.evaluate()

#Policy.constructNashGame(game, policy_state)


import multiprocessing



if __name__ == '__main__':



    ## Players
    A = Player.Player('A', 3)
    B = Player.Player('B', 2)
    C = Player.Player('C', 1)
    D = Player.Player('C', -1)
    E = Player.Player('C', 0)
    N = set([A, B, C, D, E])



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


    n_samples = 100
    T = 5
    game, policies = Game.init_game(N, T)
    result = AlternatingPolicyOpt.alternating_opt(game, policies, n_samples)
    result = np.array(result)
    #for T in range(5, 8):
    #    alternating_opt(T)

    end = time.time()
    print('take:', end-start)
    for i, agent_name in enumerate(sorted([agent.name for agent in N])):
        print(agent_name)
        plt.plot(result[:, i], '-^', label=agent_name)

        # https://queirozf.com/entries/add-labels-and-text-to-matplotlib-plots-annotation-examples
        #for x, y in zip( range(T_max-T_min+1), result[:, i]):
        #    label = "{:.2f}".format(y)
        #    plt.annotate(label,  # this is the text
        #                 (x, y),  # this is the point to label
        #                 textcoords="offset points",  # how to position the text
       #                 xytext=(0, 10),  # distance from text to points (x,y)
       #                 ha='center')  # horizontal alignment can be left, right or center
    plt.legend()
    plt.ylabel('Delta ranking')
    plt.xlabel('Game depth (T)')
    plt.title('Alternating Optimization Trajectories')
    plt.show()

#optimize_agent(C, policies, game)
#print(game.valuations[G0])
#for agent_name, delta_r in delta_ranks.items():
#    plt.plot(delta_r, label=agent_name)
#
#plt.legend()
#plt.show()



