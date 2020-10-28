from code import *
from pprint import pprint
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

A = Player.Player('A', 3)
B = Player.Player('B', 2)
C = Player.Player('C', 1)
N = set([A, B])




T = 2
game, policies = Game.init_game(N, T)
#pprint(game.policyStates)

#print('>> NO PS???')
#for game_T1 in list(game.nodes[1]):
#    print(game_T1)
#    print(game_T1.policyStates)
#    print('====')

nodes_visited = defaultdict(int) # key=nodes, val=number of times visited

n_episodes = 1000
for _ in range(n_episodes):
    Gt, PS, choices = MonteCarlo.MC_simulation(game, policies)
    for node in Gt:
        nodes_visited[node] += 1

colors = [nodes_visited[node] for node in game.graph.nodes()]
#print(">> visit freq")
#pprint(nodes_visited)

game.draw(node_color=colors, cmap=plt.cm.Reds)
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