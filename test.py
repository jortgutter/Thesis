import math
import matplotlib.pyplot as plt
import numpy as np


def dist():
    p1 = [np.random.rand(),np.random.rand()]
    p2 = [np.random.rand(), np.random.rand()]
    distance = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return distance


def pigs(n):
    roundtotal = 0
    roundthrows = []
    for i in range(n):
        throw = int(np.ceil(np.random.rand()*6))
        roundthrows.append(throw)
        if throw == 1:
            return 0, roundthrows
        roundtotal += throw
    return roundtotal, roundthrows


def try_pigs(n):
    ts = []
    total = 0
    for i in range(10000):
        score, throws = pigs(n)
        total += score
        ts = ts + throws
    print("total:", total)
    #plt.hist(ts)
    #plt.show()
    return total


scores = []
best_score = 0
best_n = 0
for i in range(1, 51):
    print("n =", i, ":")
    score = try_pigs(i)
    scores.append(score)
    if score > best_score:
        best_score = score
        best_n = i
print("Best score:", best_score)
print("rolled with a strategy of", best_n, "rolls")
plt.plot(scores)
plt.show()

# print([i for i in range(3)])
# n = 1000000
# dists = [int(dist()*1000)/1000.0 for i in range(n)]
# print(np.sum(dists)/n)
# plt.hist(dists, bins=100)
# plt.show()
