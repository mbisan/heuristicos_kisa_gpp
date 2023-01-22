import os
import numpy as np
import math

def read_graph(file):
    with open(file, 'r') as f:
        n, k = f.readline().split()
        n, k = int(n), int(k)

        grafo = np.zeros((n, n))

        for i, line in enumerate(f):
            grafo[i, [int(j) - 1 for j in line.split()]] = 1

    return grafo

test = np.array([[0,0,1,1,0,0],
                 [0,0,1,1,0,1],
                 [1,1,0,0,1,0],
                 [1,1,0,0,1,0],
                 [0,0,1,1,0,0],
                 [0,1,0,0,0,0]])

testp = np.array([1,1,1,0,0,0])

def cost(graph, p):
    pos_mat = np.zeros_like(graph)
    pos_mat[p==1,] = 1-p
    pos_mat[p==0,] = p

    return np.sum(graph * pos_mat)

def localsearch(graph, p_, n):
    cost_v = cost(graph, p_)
    recall = cost_v
    better_cost = recall

    p = np.copy(p_)
    new_p = np.copy(p_)
    better_sols = np.copy(p_)

    for _ in range(n):
        # go throug all possible changes in order without
        # computing the solutions and/or costs from scratch
        # self-loops are not allowed

        for i in np.nonzero(p==1)[0]:
            sub_cause_i = np.dot(graph[i,:],(1-p)) + np.dot(graph[:,i],(1-p))

            new_p[i] ^= 1

            for j in np.nonzero(p==0)[0]:

                new_p[j] ^= 1

                cost_v -= np.dot(graph[j,:],p) + np.dot(graph[:,j],p) + sub_cause_i

                # substracting graph[i, j] twice (also graph[i, i] and [j, j])

                cost_v += np.dot(graph[j,:],(1-new_p)) + np.dot(graph[:,j],(1-new_p)) + np.dot(graph[i,:],new_p) +np.dot(graph[:,i],new_p)

                # summing graph[i, j] twice

                if cost_v < better_cost:
                    better_sols = new_p.copy()
                    better_cost = cost_v

                cost_v = recall
                new_p[j] ^= 1

            new_p[i] ^= 1

        p = better_sols.copy()
        new_p = p.copy()
        recall = better_cost
        cost_v = recall

    return (better_cost, better_sols)

localsearch(test, testp, 1)

import time

n=1000
a=np.random.randint(0, 2, (n, n))
b=np.random.randint(0, 2, n)

print(a.dtype, b.dtype)

d=time.time()
h=localsearch(a, b, 1)
print(time.time()-d)