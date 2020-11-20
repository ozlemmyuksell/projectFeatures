import inline as inline
import networkx as nx
import math
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import csv
from itertools import groupby
import __future__
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import heapq

try:
    from itertools import imap
except ImportError:
    # Python 3...
    imap=map
from operator import itemgetter
from numpy import array

def split_on_timestamps(*args, start, end):

    newData = open('file.csv','w')
    dw = csv.writer(newData)
    for row in groupby(args):
        time = row['timestamp']
        print(time)
        if int(time) > int(start) and int(time) < int(end):
           dw.writerow(row)
    return newData

def take_integer(start, end):
    print(start)
    print(end)

def graphData():
    delimiter = ','
    with open('sample_edges.csv', 'rb') as input_file:
        reader = csv.reader(input_file, delimiter=delimiter)
        with open('output.csv', 'wb') as output_file:
            writer = csv.writer(output_file, delimiter=delimiter)
            writer.writerows(imap(itemgetter(1, 2), reader))


Data = open('data.csv', "r")
read = csv.reader(Data)
G = nx.read_edgelist('data.csv', delimiter=",")
print(nx.info(G))

#SIM RANK
print("SIM RANK")
sim = nx.simrank_similarity(G)
lol = [[sim[u][v] for v in sorted(sim[u])] for u in sorted(sim)]
sim_array = array(lol)

for i in sim_array:
    print(i)

#KATZ MEASURE
print("KATZ MEASUREMENT")
def katz_centrality(
    G,
    alpha=0.1,
    beta=1.0,
    max_iter=1000,
    tol=1.0e-6,
    nstart=None,
    normalized=True,
    weight=None,
):
    if len(G) == 0:
        return {}

    nnodes = G.number_of_nodes()

    if nstart is None:
        # choose starting vector with entries of 0
        x = {n: 0 for n in G}
    else:
        x = nstart

    try:
        b = dict.fromkeys(G, float(beta))
    except (TypeError, ValueError, AttributeError) as e:
        b = beta
        if set(beta) != set(G):
            raise nx.NetworkXError(
                "beta dictionary " "must have a value for every node"
            ) from e
# make up to max_iter iterations
    for i in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast, 0)
        # do the multiplication y^T = Alpha * x^T A - Beta
        for n in x:
            for nbr in G[n]:
                x[nbr] += xlast[n] * G[n][nbr].get(weight, 1)
        for n in x:
            x[n] = alpha * x[n] + b[n]

        # check convergence
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < nnodes * tol:
            if normalized:
                # normalize vector
                try:
                    s = 1.0 / sqrt(sum(v ** 2 for v in x.values()))
                # this should never be zero?
                except ZeroDivisionError:
                    s = 1.0
            else:
                s = 1
            for n in x:
                x[n] *= s
            return x
    raise nx.PowerIterationFailedConvergence(max_iter)

def katz_centrality_numpy(G, alpha=0.1, beta=1.0, normalized=True, weight=None):
    if len(G) == 0:
        return {}
    try:
        nodelist = beta.keys()
        if set(nodelist) != set(G):
            raise nx.NetworkXError(
                "beta dictionary " "must have a value for every node"
            )
        b = np.array(list(beta.values()), dtype=float)
    except AttributeError:
        nodelist = list(G)
        try:
            b = np.ones((len(nodelist), 1)) * float(beta)
        except (TypeError, ValueError, AttributeError) as e:
            raise nx.NetworkXError("beta must be a number") from e

    A = nx.adj_matrix(G, nodelist=nodelist, weight=weight).todense().T
    n = A.shape[0]
    centrality = np.linalg.solve(np.eye(n, n) - (alpha * A), b)
    if normalized:
        norm = np.sign(sum(centrality)) * np.linalg.norm(centrality)
    else:
        norm = 1.0
    centrality = dict(zip(nodelist, map(float, centrality / norm)))
    print(centrality)

    return centrality

katz_centrality_numpy(G, alpha=0.1, beta=1.0, normalized=True, weight=None)

#PAGERANK
print("PAGERANK")
pr = nx.pagerank(G, alpha=0.9)
print(pr)

#HITTING TIME
print("HITTING TIME")
h,a = nx.hits(G)
print(h,a)

#ADAMIC ADAR
print("ADAMIC ADAR")
def nonedges(G,u):  #a generator with (u,v) for every non neighbor v
    for v in nx.non_neighbors(G, u):
        yield (u, v)

for u in G:# you may want to check that there will be at least 10 choices.
    preds = nx.adamic_adar_index(G,nonedges(G,u))
    tenlargest = heapq.nlargest(10, preds, key = lambda x: x[2])
    print(tenlargest)

#JACCARD

#preds = nx.jaccard_coefficient(G, [(19, 1), (2, 3)])
#for u, v, p in preds:
#    print(f"({u}, {v}) -> {p:.8f}")

#PREFERENTIAL ATTACHMENT
preds = nx.preferential_attachment(G, [(0, 1), (2, 3)])
print(preds)
