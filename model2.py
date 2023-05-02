from math import factorial
from random import random, seed, shuffle
from itertools import permutations

def solve(count, alternatives, W):
    voter_locations = list(count.keys())
    points = voter_locations + [str(i) for i in alternatives]
    import docplex.mp.model as cpx

    opt_model = cpx.Model(name="MIP Model")

    d = {(i, j): opt_model.continuous_var(lb=0, name="d_{0}_{1}".format(i, j)) for i in points for j in points}

    constraints_triangle = {
    (i, j, k): opt_model.add_constraint(ct=d[i, k] <= d[i, j] + d[j, k], ctname="triangle_{0}_{1}_{2}".format(i, j, k))
    for i in points for j in points for k in points}
    constraints_sym = {(i, j): opt_model.add_constraint(ct=d[i, j] == d[j, i], ctname="sym_{0}_{1}".format(i, j)) for i
                       in points for j in points}
    constraints_ident = {(i): opt_model.add_constraint(ct=d[i, i] == 0, ctname="ident_{0}".format(i)) for i in points}

    # preferences

    constraints_prefs = {}
    for p in count:
        for i in range(len(p) - 1):
            constraints_prefs[p, i] = opt_model.add_constraint(ct=d[p, p[i]] <= d[p, p[i + 1]],
                                                               ctname="prefs_{0}_{1}".format(p, i))

    # fix the optimal solution to have cost of 1

    optimal_cost = opt_model.add_constraint(ct=opt_model.sum(count[i] * d[optimal, i] for i in count) == 1,
                                            ctname="fix_cost_opt")

    opt_model.maximize(opt_model.sum(opt_model.sum(count[i] * d[W, i] for i in count)))

    opt_model.solve()

    return opt_model.objective_value


def run_test(alternatives_c, alternatives, sample):
    count = {}
    perms = list(permutations(alternatives))
    shuffle(perms)
    perms = perms[:int(len(perms)*sample+1)]
    for p in perms:
            count[p] = random()

    best = None
    best_dist = 10 ** 10

    for A in alternatives:
        worse = 1
        for B in alternatives:
            if A is B: continue
            worse = max(solve(count, alternatives, A, B), worse)
        if worse < best_dist:
            best_dist = worse
            best = A
    return best_dist

seed(0)

alternatives_c = 4
alternatives = ["A" + str(i) for i in range(alternatives_c)]

prefererence_profiles = factorial(alternatives_c)


for i in range(100):
    print(run_test(alternatives_c, alternatives, .2))