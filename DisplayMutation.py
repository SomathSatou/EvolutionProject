import heapq

import numpy as np
import matplotlib.pyplot as plt
from random import *

listFile = ["twoBest, monoPoint, bitflips, bestoflower, .txt",
            "twoBest, monoPoint, fiveflips, bestoflower, .txt",
            "twoBest, monoPoint, oneflip, bestoflower, .txt",
            "twoBest, monoPoint, threeflip, bestoflower, .txt",
            "twoBest, monoPoint, flipsAdaptatifPursuit, bestoflower, .txt",
            "twoBest, monoPoint, UCB_mutation, bestoflower, .txt"]
"""
listFile = ["twoBest, monoPoint, oneflip, elder, .txt",
            "twoBest, monoPoint, threeflip, bestoflower, .txt",
            "twoBest, uniform, oneflip, bestoflower, .txt",
            "twoBest, monoPoint, oneflip, bestoflower, .txt",
            "twoRandom, monoPoint, oneflip, bestoflower, .txt"]

listFile = ["twoBest, monoPoint, oneflip, bestoflower, .txt",
            "twoRandom, monoPoint, oneflip, bestoflower, .txt",
            "twoBestIn5Random, monoPoint, oneflip, bestoflower, .txt",
            "wheel, monoPoint, oneflip, bestoflower, .txt" ]

listFile = ["twoBest, monoPoint, oneflip, bestoflower, .txt",
            "twoBest, uniform, oneflip, bestoflower, .txt",
            "twoBest, multiPoint, oneflip, bestoflower, .txt"]

listFile = ["twoBest, monoPoint, oneflip, bestoflower, .txt",
            "twoBest, monoPoint, oneflip, elder, .txt"]"""

for filename in listFile:
    fichier = open(filename, "r")

    x = []

    out = []


    index = 0
    lignes = fichier.readlines()
    nbrExe = int(len(lignes))

    y = [ []*nbrExe for i in range(12000)]

    for l in lignes:
        print(str(index))
        data = l.split(";")
        for i in range(0, len(data)-1):
            y[i].append(int(data[i]))
        index = index + 1


    for o in range(0, len(y)):
        if y[o] != []:
            out.append(np.mean(y[o]))

    for j in range(0, len(out)):
        x.append(j+1)


    plt.plot(x, out, label=filename[:-4])
    x = []

    fichier.close()
    print("fichier")

plt.legend()
plt.show()

