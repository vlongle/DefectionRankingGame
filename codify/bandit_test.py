from code import *
from pprint import pprint
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

A = Player.Player('A', 3)
B = Player.Player('B', 2)
C = Player.Player('C', 1)
N = set([A, B, C])




T = 2
game, policies = Game.init_game(N, T)

target_agent = C
n_episodes = 100
gamma = 0.08
sExp3 = Bandit.S_exp3(game, policies, target_agent, n_episodes, gamma)
sExp3.optimize_agent()


#pprint(game.policyStates)

#print('>> NO PS???')
#for game_T1 in list(game.nodes[1]):
#    print(game_T1)
#    print(game_T1.policyStates)
#    print('====')

#nodes_visited = defaultdict(int) # key=nodes, val=number of times visited
#
#n_episodes = 1000
#for _ in range(n_episodes):
#    Gt, PS, choices = MonteCarlo.MC_simulation(game, policies)
#    for node in Gt:
#        nodes_visited[node] += 1

#colors = [nodes_visited[node] for node in game.graph.nodes()]

#print(">> visit freq")
#pprint(nodes_visited)

#game.draw(node_color=colors, cmap=plt.cm.Reds)
#game.fancy_draw(node_color=colors, cmap=plt.cm.Reds)


#game.draw()
#pos =graphviz_layout(game.graph, prog='dot')
#nx.draw(game.graph, pos, node_color=colors, cmap=plt.cm.Reds)
#plt.show()

#game.draw(node_color=colors, cmap=plt.cm.Blues)
#game.draw(node_color=colors, cmap=plt.cm.jet)

#print('>>Gts:')
#pprint(Gt)
#print('>>PS:')
#pprint(PS)
#print('>>choices:')
#pprint(choices)

#G0 = game.G0
#pprint(G0.policyStates)

#PS = G0.policyStates[0]
#print(PS.coalition_considered)

#num_episodes = 20
#for i in range(num_episodes):
#    print(MonteCarlo.draw_policy_state(G0))



# https://stackoverflow.com/questions/47380865/json-serialization-error-using-matplotlib-mpld3-with-linkedbrush


#import matplotlib.pyplot as plt
#import numpy as np
#import mpld3
#
#import networkx as nx
#
#plt.style.use('fivethirtyeight')
#
#G = nx.path_graph(4)
#pos = nx.spring_layout(G)
#
##fig, ax = plt.subplots(subplot_kw=dict(facecolor='#EEEEEE'))
#fig, ax = plt.subplots()
#scatter = nx.draw_networkx_nodes(G, pos, ax=ax)
#nx.draw_networkx_edges(G, pos, ax=ax)
#
#labels = G.nodes()
#tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
#mpld3.plugins.connect(fig, tooltip)
#plt.axis("off")
#mpld3.show()



