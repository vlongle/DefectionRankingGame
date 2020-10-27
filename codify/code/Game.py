from collections import defaultdict
from code.PolicyState import PolicyState
from code.utils import *
from code.CoalitionStructure import CoalitionStructure
from code.State import State
from code.Policy import softmax
import networkx as nx
import numpy as np
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

class Game:
    def __init__(self, G0, N, graph, T):
        self.G0 = G0  # initial game state!
        self.N = list(N)
        self.graph = graph
        self.T = T  # termination time
        self.nodes = defaultdict(set)
        self.nodes[0] = set([self.G0])  # key = time, value = list of nodes
        self.policyStates = []
        self.create_policy_states(self.G0)
        self.valuations = defaultdict(dict)  # key = (leaf) state, value = delta ranking
        self.build_graph()

    def build_graph(self):
        for t in range(self.T):
            nodes = self.nodes[t]
            for node in nodes:
                node.expand_states(self)

    def draw(self):
        plt.rcParams['figure.figsize'] = 30, 10
        pos =graphviz_layout(self.graph, prog='dot')
        nx.draw(self.graph, pos,alpha=0.5, node_size=50, arrowsize=5, arrows=True)
        plt.show()

    def add_state(self, new_state):
        if new_state in self.nodes[new_state.t]:
            return

        self.nodes[new_state.t].add(new_state)
        if new_state.t < self.T:
            self.create_policy_states(new_state)

    def create_policy_states(self, state):
        p = powerset(state.F)
        for C in p:  # C is a proposed new coalition!
            if not C:
                continue
            policy_state = PolicyState(state, C, len(self.policyStates))
            self.policyStates.append(policy_state)
            state.policyStates.append(policy_state)



## Game defn
def init_game(N, T):
    CS0 = CoalitionStructure(CS=[], Z=[])
    G0 = State(N, CS0, t=0)
    graph = nx.DiGraph()

    ## build Game
    game = Game(G0, N, graph, T)

    policies = {} # key = player, value = policy
    # policy = np array
    for agent in game.N:
        policies[agent] = 0.5 * np.ones(shape=(len(game.policyStates), 2))
        #policies[agent] = np.random.normal(0, 2, size=(len(game.policyStates), 2))
        #policies[agent] = softmax(policies[agent])

        # remember to softmax these stuff!
        #policies[agent] = np.random.normal(loc=0, scale=1, size=(len(game.policyStates), 2))
    return game, policies
