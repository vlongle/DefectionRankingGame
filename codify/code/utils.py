from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_ranking(d):
    # https://stackoverflow.com/questions/2299696/positional-rankings-and-dealing-with-ties-in-python
    # d = {A:absolute_weight_A, ...}
    sorted_scores = sorted(d.items(),\
            key=lambda player: d[player[0]])
    res = {}
    prev = None
    for i,(k,v) in enumerate(sorted_scores):
        if v!=prev:
            place,prev = i+1,v
        res[k] = place
    return res

def delta_ranking(ranking_init, ranking_term):
    '''
    ranking_init = {'A': 1, 'B': 1, 'C': 3, 'D': 4}
    '''
    res = {}
    for player, rank_init in ranking_init.items():
        res[player] = ranking_term[player] - rank_init
    return res