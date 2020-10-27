'''
This is the S-exp3 algo.
Reward should be scaled to [0, 1]
'''



class S_exp3:
    def __init__(self, gamma):
        self.t = 0
        self.num_actions = 2
        self.weights = [0]*self.num_actions # two actions: leave or stay
        self.gamma = gamma


    def dist(self):
        # convert self.weights to a prob. dist
        normalizing = sum(self.weights)
        return [(1-self.gamma) * (w/normalizing) + (self.gamma/len(self.num_actions))\
                for w in self.weights]

    def update_weight(self):
        pass


    def scale_reward(self):
        pass



