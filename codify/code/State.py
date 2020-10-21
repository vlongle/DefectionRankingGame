from code.CoalitionStructure import CoalitionStructure
from code.utils import *


def v(s):
    # valuation of a coalition
    return 5 * int(len(s) >= 2)

class State:
    def __init__(self, F, CS, t):
        self.F = set(F)  # FREE is a set
        self.CS = CS  # coalition structure
        self.t = t
        self.name = 'G(' + 'F=' + str(self.F) + ',' + str(self.CS) + ',t=' + str(self.t) + ')'
        self.value = None
        self.policyStates = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, State):
            raise Exception('Cannot compare!')
            # don't attempt to compare against unrelated types
        return self.name == other.name

    def __repr__(self):
        return self.name

    def evaluate_leaf(self):
        # this function should only be called for the very bottom horizon leaf
        # nodes!
        # for leaf node, return dict {'A': delta_ranking, 'B':...}
        abs_weights_term = {}
        abs_weights_init = {}

        # free
        for agent in self.F:
            abs_weights_term[agent.name] = agent.score
            abs_weights_init[agent.name] = agent.score

        # coalition structure
        for cs, z in zip(self.CS.CS, self.CS.Z):
            for agent in cs:
                abs_weights_term[agent.name] = agent.score + z * v(cs)
                abs_weights_init[agent.name] = agent.score

        ranking_init = get_ranking(abs_weights_init)
        ranking_term = get_ranking(abs_weights_term)
        self.value = delta_ranking(ranking_init, ranking_term)
        # e.g. {'C': 1, 'B': -1, 'A': 0}
        return self.value

    def expand_states(self, game):
        p = powerset(self.F)
        for C in p:  # C is a aftermath new coalition!
            if not C:
                # new coalition proposed but everyone leave!
                new_state = State(self.F, self.CS, self.t + 1)
                # if new_state in game.nodes[self.t+1]:
                #    new_state = game.nodes[self.t+1][new_state]
                game.graph.add_edge(self, new_state)
                game.add_state(new_state)
                game.nodes[self.t + 1].add(new_state)
                continue
            # else, C is no longer free!
            F_new = self.F.difference(C)
            # C can either be successful or lost
            for z in [0, 1]:
                if not F_new and not z:
                    continue  # to avoid the case empty set F but Z = 0 indicating failure still
                CS_new = CoalitionStructure(self.CS.CS + [set(C)], self.CS.Z + [z])
                new_state = State(F_new, CS_new, self.t + 1)
                game.graph.add_edge(self, new_state)
                # if new_state in game.nodes[self.t+1]:
                #    new_state = game.nodes[self.t+1][new_state]
                game.add_state(new_state)

