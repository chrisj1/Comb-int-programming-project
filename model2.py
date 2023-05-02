from math import factorial
from random import random, seed, shuffle
from itertools import permutations
from time import time

def solve(count, alternatives, W):
    M = 10**8
    voter_locations = list(count.keys())
    points = voter_locations + [str(i) for i in alternatives]
    import docplex.mp.model as cpx
    import docplex.cp.parameters as params

    params.LogVerbosity = "Verbose"

    opt_model = cpx.Model(name="MIP Model")

    d = {(i, j): opt_model.continuous_var(lb=0, name="d_{0}_{1}".format(i, j)) for i in points for j in points}

    x = {i : opt_model.binary_var(name="x_{0}".format(i)) for i in alternatives}

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

    optimal_cost_1 = {p : opt_model.add_constraint(ct=opt_model.sum(count[i] * d[i, a] for i in count) >= x[a],
                                            ctname="fix_cost_opt") for a in alternatives}

    optimal_cost_2 = {p : opt_model.add_constraint(ct=opt_model.sum(count[i] * d[i, a] for i in count) <= M*(1-x[a]) + 1,
                                            ctname="fix_cost_opt") for a in alternatives}

     # ensure the optimal solution is optimal

    optimal_best = {(i, Y) : opt_model.add_constraint(ct=opt_model.sum(count[v] * d[v, Y] for v in count) - opt_model.sum(count[v] * d[v, i] for v in count) >= (x[i]-1) * M,
                                            ctname="fix_cost_opt") for i in alternatives for Y in alternatives}

    #sum of x
    single_x = opt_model.add_constraint(ct=opt_model.sum(x[i] for i in alternatives) == 1, ctname="sum_of_x")
    opt_model.maximize(opt_model.sum(opt_model.sum(count[i] * d[W, i] for i in count)))

    t = time()
    opt_model.solve()

    return opt_model.objective_value, (time() - t)*1000


def run_test(alternatives_c, alternatives, sample):
    count = {}
    perms = list(permutations(alternatives))
    shuffle(perms)
    perms = perms[:int(len(perms)*sample+1)]
    for p in perms:
            count[p] = random()

    best = None
    best_dist = 10 ** 10
    total_time = 0

    for A in alternatives:
        value, time = solve(count, alternatives, A)
        total_time+=time
        if value < best_dist:
            best_dist = value
            best = A
    return best_dist, total_time

seed(0)

def run_test_alternatives(alternatives_c, runs, samples=1):
    alternatives = ["A" + str(i) for i in range(alternatives_c)]

    prefererence_profiles = factorial(alternatives_c)

    times = []
    distortions = []

    for i in range(runs):
        d, t = run_test(alternatives_c, alternatives, samples)
        print(alternatives_c, d,t)
        times.append(t)
        distortions.append(d)
    return distortions, times


import matplotlib.pyplot as plt
import numpy as np
import pickle

times = []
distortions = []

ks = [2,3,4,5,6]
samples = [1,1,1,.25,.05]
runs = 40
for i in range(len(ks)):
    d,t = run_test_alternatives(ks[i], runs, samples[i])

    times.append(t)
    distortions.append(d)


fig, axs = plt.subplots(2, 1)

axs[0].boxplot(times, labels=ks)
axs[0].set_title('Time')
axs[0].set_yscale('log')

print(distortions)

axs[1].boxplot(distortions, labels=ks)
axs[1].set_title('Distortion')




with open("model2.dat", "wb") as f:
    pickle.dump((times, distortions, ks), f)

plt.show()
