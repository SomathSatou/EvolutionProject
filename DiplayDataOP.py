import heapq

import numpy as np
import matplotlib.pyplot as plt
from random import *

fichier = open("dataOPAPW.txt", "r")

lignes = fichier.readlines()

nbrExe = int(len(lignes)/5)

one = [[]*nbrExe for i in range(12000)]
three = [[]*nbrExe for i in range(12000)]
five = [[]*nbrExe for i in range(12000)]
bit = [[]*nbrExe for i in range(12000)]

x = []

index = 0

# oneflip data

#print(data)
for longueur in range(0, nbrExe):
    data = lignes[longueur*5].split(";")
    for i in range(0, len(data)-1):
        one[i].append(float(data[i]))

    # threeflip data
    data = lignes[longueur*5+1].split(";")
    for i in range(0, len(data)-1):
        three[i].append(float(data[i]))

    # bitflip data
    data = lignes[longueur*5+2].split(";")
    for i in range(0, len(data)-1):
        bit[i].append(float(data[i]))

    # fiveflip data
    data = lignes[longueur*5+3].split(";")
    for i in range(0, len(data)-1):
        five[i].append(float(data[i]))



out = []
for o in range(0, len(one)):
    if one[o] != []:
        out.append(np.mean(one[o]))
for j in range(0, len(out)):
    x.append(j + 1)
plt.plot(x, out, label="oneflip")

x = []
out = []
for o in range(0, len(three)):
    if three[o] != []:
        out.append(np.mean(three[o]))
for j in range(0, len(out)):
    x.append(j + 1)
plt.plot(x, out, label="threeflip")

x = []
out = []
for o in range(0, len(five)):
    if five[o] != []:
        out.append(np.mean(five[o]))
for j in range(0, len(out)):
    x.append(j + 1)
plt.plot(x, out, label="fiveflip")

x =[]
out = []
for o in range(0, len(bit)):
    if bit[o] != []:
        out.append(np.mean(bit[o]))
for j in range(0, len(out)):
    x.append(j + 1)
plt.plot(x, out, label="bitflip")

fichier.close()
print("fichier")

plt.legend()
plt.show()
