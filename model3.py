from math import factorial
from random import random, seed, shuffle
from itertools import permutations
from time import time
import networkx as nx
from networkx.algorithms import bipartite

import docplex.mp.model as cpx

def mtch_cost(mtch, dists, V1):
    c = 0
    for v in V1:
        pair = mtch[v]
        c+=dists[v,pair]
    return c

def minMatching(V1, V2, dists):
    B = nx.Graph()
    B.add_nodes_from(V1, bipartite=0)
    B.add_nodes_from(V2, bipartite=1)

    for i in V1:
        for j in V2:
            B.add_edge(i,j, weight=dists[i,j])
    return bipartite.matching.minimum_weight_full_matching(B, V1, "weight")

def all_matchings(V1, V2, n):
    l = []
    for p in permutations([i for i in range(n)]):
        m = {V1[i] : V2[p[i]] for i in range(n)}
        l.append(m)
    return l

def solve(V1, V2, prefs, W, optimal):
    points = V1 + V2

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
    for v1 in V1:
        for i in range(len(V2) - 1):
            constraints_prefs[v1, i] = opt_model.add_constraint(ct=d[v1, prefs[v1][i]] <= d[v1, prefs[v1][i+1]],
                                                               ctname="prefs_{0}_{1}".format(v1, i))

    for v2 in V2:
        for i in range(len(V1) - 1):
            constraints_prefs[v2, i] = opt_model.add_constraint(ct=d[v2, prefs[v2][i]] <= d[v2, prefs[v2][i+1]],
                                                               ctname="prefs_{0}_{1}".format(v2, i))
    # fix the optimal solution to have cost of 1

    optimal_cost = opt_model.add_constraint(ct=opt_model.sum(d[i, optimal[i]] for i in V1) == 1,
                                            ctname="fix_cost_opt")

    opt_model.maximize(opt_model.sum(opt_model.sum(d[i, W[i]] for i in V1)))

    t = time()
    min_cost = 0

    while min_cost<1-.0001:
        opt_model.solve()

        dists = {(v1, v2): opt_model.solution.get_values([d[v1,v2]])[0] for v1 in V1 for v2 in V2 }
        mtch = minMatching(V1, V2, dists)
        min_cost = mtch_cost(mtch, dists, V1)
        opt_model.add_constraint(ct=opt_model.sum(d[i, mtch[i]] for i in V1) >= 1)
        if min_cost < 1-.0001: 
            print("found better matching adding constrraint", min_cost)

    return opt_model.objective_value, (time() - t)*1000


def run_test(n):

    V1 = ["v1_{}".format(i) for i in range(n)]
    V2 = ["v2_{}".format(i) for i in range(n)]

    prefs = {}
    for v in V1:
        p = V2
        shuffle(p)
        prefs[v] = p

    for v in V2:
        p = V1
        shuffle(p)
        prefs[v] = p
    
    ws = all_matchings(V1, V2, n)
    os = all_matchings(V1, V2, n)
    best_matching_cost = 10**10
    tt = 0
    for W in ws:
        worse = 0
        for O in os:
            d,t = solve(V1, V2, prefs, W, O)
            worse = max(worse, d)
            tt+=t
        best_matching_cost = min(worse, best_matching_cost)
    return best_matching_cost, tt
seed(0)

def run_test_alternatives(runs, n):

    times = []
    distortions = []

    for i in range(runs):
        d, t = run_test(n)
        print(n,d,t)
        times.append(t)
        distortions.append(d)
    return distortions, times


import matplotlib.pyplot as plt
import numpy as np
import pickle

times = []
distortions = []

ks = [2,3,4,5,6]
runs = 40
for i in range(len(ks)):
    d,t = run_test_alternatives(runs,ks[i])

    times.append(t)
    distortions.append(d)
    print(distortions)


fig, axs = plt.subplots(2, 1)

axs[0].boxplot(times, labels=ks)
axs[0].set_title('Time')
axs[0].set_yscale('log')

print(distortions)

axs[1].boxplot(distortions, labels=ks)
axs[1].set_title('Distortion')




with open("model3.dat", "wb") as f:
    pickle.dump((times, distortions, ks), f)

plt.show()
