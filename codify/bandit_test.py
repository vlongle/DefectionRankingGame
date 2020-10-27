from code import *
from pprint import pprint


A = Player.Player('A', 3)
B = Player.Player('B', 2)
C = Player.Player('C', 1)
N = set([A, B])




T = 2
game, policies = Game.init_game(N, T)

G0 = game.G0
#pprint(G0.policyStates)

PS = G0.policyStates[0]
print(PS.coalition_considered)

#num_episodes = 20
#for i in range(num_episodes):
#    print(MonteCarlo.draw_policy_state(G0))