class CoalitionStructure:
    def __init__(self, CS, Z):
        self.CS = CS # e.g. [{1}, {2,3}]
        self.Z = Z # e.g. (0, 1)
        #self.name = 'CS(' + str(self.CS) + ',Z=' + str(self.Z) + ')'
        self.anti_dup = set()
        for coalition, outcome in zip(self.CS, self.Z):
            self.anti_dup.add((frozenset(coalition), outcome))
        self.name = 'CS(' + str(self.anti_dup) + ')'
    def __repr__(self):
        return self.name
