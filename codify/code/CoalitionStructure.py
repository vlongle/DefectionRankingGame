class CoalitionStructure:
    def __init__(self, CS, Z):
        self.CS = CS # e.g. [{1}, {2,3}]
        self.Z = Z # e.g. (0, 1)
        self.name = 'CS(' + str(self.CS) + ',Z=' + str(self.Z) + ')'
    def __repr__(self):
        return self.name
