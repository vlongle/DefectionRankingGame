class PolicyState:
    def __init__(self, parentState, coalition_considered, state_num):
        self.parentState = parentState
        self.coalition_considered = coalition_considered
        self.state_num = state_num
        self.name = 'PS(' + str(parentState) + ', || CS=' + str(coalition_considered) + ')'
    def __repr__(self):
        return self.name
