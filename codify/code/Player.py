class Player:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def init_policy(self, policy):
        self.policy = policy  # policy is a 2d matrix
        # [state][action_probability]
        # e.g. num_states = 3
        # [[0.5, 0.5],
        # [0.7, 0.3],
        # [0.3, 0.8]]

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        # <= operator
        return self.score < other.score

    def __le__(self, other):
        # <= operator
        return self.score <= other.score