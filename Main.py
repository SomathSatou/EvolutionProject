import heapq

import numpy as np
import matplotlib.pyplot as plt
from random import *
import math

# region Selection
def twoBest():
    global CurrentEval, Population
    ret = []
    best = heapq.nlargest(2, CurrentEval)
    ret.append(Population[CurrentEval.index(best[0])].copy())
    ret.append(Population[CurrentEval.index(best[1])].copy())
    return ret


def twoRandom():
    global Population, SizePop
    ret = []
    first = 0
    second = 0
    while 1:
        if first != second:
            break
        first = randint(0, SizePop-1)
        second = randint(0, SizePop-1)
    ret.append(Population[first].copy())
    ret.append(Population[second].copy())
    return ret


def twoBestIn5Random():
    global Population, SizePop, CurrentEval
    ret = []
    indexRand = []
    evalRand = []
    while len(indexRand) < 5:
        tmp = randint(0, SizePop - 1)
        if not indexRand.__contains__(tmp):
            indexRand.append(tmp)
            evalRand.append(CurrentEval[tmp])
    best = heapq.nlargest(2, evalRand)
    ret.append(Population[indexRand[evalRand.index(best[0])]].copy())
    ret.append(Population[indexRand[evalRand.index(best[1])]].copy())
    return ret


def wheel():
    global Population, SizePop, CurrentEval, Size
    ret = []
    wheel = []
    perCent = 0
    for child in range(0, len(Population)):
        perCent += (CurrentEval[child]/(sum(CurrentEval)+1))*100
        wheel.append([child, perCent])
    while len(ret) < 2:
        prop = randint(0, 100)
        for elt in wheel:
            if elt[1] < prop:
                ret.append(Population[elt[0]].copy())
                break;
    return ret


# endregion Selection

# region Recombination
def monoPoint(parents):
    childs = parents.copy()
    pivot = randint(1, len(childs[0]))
    if randint(0,100) < RecombinationProp:
        for i in range(0,pivot):
            tmp = childs[0][i]
            childs[0][i] = childs[1][i]
            childs[1][i] = tmp
    return childs


def uniform(parents):
    childs = parents.copy()
    for elt in range(0, len(childs[0])):
        if randint(0, 100) < 50:
            tmp = childs[0][elt]
            childs[0][elt] = childs[1][elt]
            childs[1][elt] = tmp
    return childs


def multiPoint(parents):
    childs = parents.copy()
    pivot = randint(1, len(childs[0]))
    pivot2 = randint(pivot, len(childs[0]))
    if randint(0, 100) < RecombinationProp:
        for i in range(0,pivot):
            tmp = childs[0][i]
            childs[0][i] = childs[1][i]
            childs[1][i] = tmp
        for i in range(pivot2, len(childs[0])):
            tmp = childs[0][i]
            childs[0][i] = childs[1][i]
            childs[1][i] = tmp
    return childs


# endregion Recombination


# region Mutation
def oneflip():
    global Childrens
    childs = Childrens
    pivot = randint(0, len(childs[0])-1)
    if randint(0,100)<= MutationProb:
        if childs[0][pivot] == 1:
            childs[0][pivot] = 0
        else:
            childs[0][pivot] = 1
    return


def threeflip():
    global Childrens
    childs = Childrens
    if randint(0, 100) <= MutationProb:
        indexRand = []
        while len(indexRand) < 3:
            tmp = randint(0, Size- 1)
            if not indexRand.__contains__(tmp):
                indexRand.append(tmp)
        for elt in indexRand:
            if childs[0][elt] == 1:
                childs[0][elt] = 0
            else:
                childs[0][elt] = 1
    return


def bitflips():
    global Childrens
    for index in range(0, len(Childrens[0])):
        if randint(0, len(Childrens[0])) == 0:
            if Childrens[0][index] == 1:
                Childrens[0][index] = 0
            else:
                Childrens[0][index] = 1

    return


def fiveflips():
    global Childrens
    childs = Childrens
    if randint(0, 100) <= MutationProb:
        indexRand = []
        while len(indexRand) < 5:
            tmp = randint(0, Size- 1)
            if not indexRand.__contains__(tmp):
                indexRand.append(tmp)
        for elt in indexRand:
            if childs[0][elt] == 1:
                childs[0][elt] = 0
            else:
                childs[0][elt] = 1
    return

def wheel():
    global Childrens, CurrentEval, ChildrenEval, deltaEval, probOP
    initializeOP()
    tirage = randint(0, 100)
    sum = 0

    for elt in range(0, len(probOP)):
        sum += probOP[elt]
        if tirage < sum * 100:
            tmp = mutSwitch.get(elt + 1, lambda: oneflip)
            print(tmp)

            tmp()
            evaluatechildren()
    return
5
def adaptatifWhell():
    global Childrens, CurrentEval, ChildrenEval, deltaEval, probOP
    Pmin = (1/len(probOP)) * 0.5 # 0.625
    return


def flipsAdaptatifPursuit():
    global Childrens, CurrentEval, ChildrenEval, deltaEval, probOP, DataOP
    beta = 0.8 # a paramétrer
    Pmin = (1/len(probOP)) *0.2#* 0.001 # diminuer 0.005 si jamais
    Pmax = 1 - (len(probOP)-1)*Pmin

#    print(Pmin)

    evaluatechildren()
    OldChildrenEval = ChildrenEval.copy()
    OldChildren = Childrens.copy()

    tirage = randint(0, 10000)
    sum = 0

    for elt in range(0, len(probOP)):
        sum += probOP[elt]
        if (tirage < sum*10000):

            tmp = mutSwitch.get(elt, lambda: oneflip)
            tmp()
            evaluatechildren()
            if ( 0 < (ChildrenEval[0] - OldChildrenEval[0])): #inverse old and children eval
                for ind in range(0, len(probOP)):
                    if ind == elt:
                        probOP[ind] = probOP[ind]*(1 - beta) + beta * Pmax
                    else:
                        probOP[ind] = probOP[ind]*(1 - beta) + beta * Pmin
            # ajout
            else:
                Childrens = OldChildren
            # end ajout
            #if nbCycle % 500 == 0 :
                #initializeOP()
            DataOP.append(probOP)
            break

#    print(probOP)
#   print(deltaEval)
    return


def UCB_mutation():
    global numbers_of_mutation,sums_of_reward
    evaluatechildren()
    OldChildrenEval = ChildrenEval.copy()

    mutation_selected = 1
    max_upper_bound = 0

    for i in range(0, NbrOP):
        if (numbers_of_mutation[i] > 0):
            average_reward = sums_of_reward[i] / numbers_of_mutation[i]
            delta_i = math.sqrt(2 * math.log(nbCycle + 1) / numbers_of_mutation[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            mutation_selected = i
    numbers_of_mutation[mutation_selected] += 1
    tmp = mutSwitch.get(mutation_selected+1, lambda: oneflip)

    tmp()
    evaluatechildren()
    reward = ChildrenEval[0] - OldChildrenEval[0]
    sums_of_reward[mutation_selected] += reward
    return


# endregion Mutation

# region Insertion
def bestoflower():
    global Childrens, ChildrenEval, Population, CurrentEval
    best = ChildrenEval.index(max(ChildrenEval))
    for j in range(0, SizePop-1):
        if CurrentEval[j] < ChildrenEval[best]:
            CurrentEval[j] = ChildrenEval[best]
            Population[j] = Childrens[best].copy()
            ChildrenEval = []
            Childrens = []
            return
    return


def elder():
    global Childrens, ChildrenEval, Population, CurrentEval
    best = ChildrenEval.index(max(ChildrenEval))
    Population.append(Childrens[best].copy())
    Population.pop(0)
    return


# endregion Insertion

# region function of algo
def evaluate():
    global Population, CurrentEval
    CurrentEval = []
    for elt in Population:
        CurrentEval.append(fitness(elt))
    return


def objectif(elt):
    global Size
    if fitness(elt) == Size:
        return True
    return False


def fitness(elt):
    ret = 0
    for value in elt:
        if value == 1:
            ret = ret + 1
    return ret


def terminaison():
    global nbCycleMax
    if nbCycle >= nbCycleMax:
        return False
    if max(CurrentEval) == Size:
        return False
    return True


def evaluatechildren():
    global Childrens, ChildrenEval
    ChildrenEval = []
    for elt in Childrens:
        ChildrenEval.append(fitness(elt))
    return


def CurrentBest():
    return max(CurrentEval)


def CurrentLow():
    return min(CurrentEval)

def initializeOP():
    global NbrOP, deltaEval, probOP
    NbrOP = 4
    deltaEval = [0] * NbrOP
    probOP = [(1 / NbrOP)] * NbrOP
    return
# endregion function of algo

# region Parameters
initializeOP()

numbers_of_mutation = [0] * NbrOP
sums_of_reward = [0] * NbrOP

x = []
y = []

moyY = []
worstY = []

mutationType = 1
mutSwitch = {
    1: oneflip,
    2: threeflip,
    3: bitflips,
    4: fiveflips,
    5: wheel,
    6: adaptatifWhell,
    7: flipsAdaptatifPursuit,
    8: UCB_mutation,
}

selectionType = 4
selSwitch = {
    1: twoBest,
    2: twoRandom,
    3: twoBestIn5Random,
    4: wheel
}

recombinationType = 1
recSwitch = {
    1: monoPoint,
    2: uniform,
    3: multiPoint
}

reinsertionType = 1
reiSwitch = {
    1: bestoflower,
    2: elder
}

def initialisation(taille, pop, mut, rec, nbcMax):
    global Size, SizePop, MutationProb, RecombinationProp, nbCycleMax, x, y, moyY, Population, nbCycle
    x = []
    moyY = []
    y = []
    Size = taille
    SizePop = pop
    MutationProb = mut
    RecombinationProp = rec
    nbCycleMax = nbcMax
    tmpIndividu = [0]
    tmpIndividu *= Size
    Population = []
    nbCycle = 0
    for i in range(0, SizePop):
        Population.append(tmpIndividu.copy())
    return


# endregion Parameters

# region Main
# [mutation, selection,recombinaison, reinsertion]
#methodList = [[1, 1, 1, 2], [2, 1, 1, 1], [1, 1, 2, 1], [1 ,1 ,1 ,1], [1, 2, 1, 1]]

# Mutation set of parameter
#methodList = [[1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1], [4, 1, 1, 1], [5, 1, 1, 1], [6, 1, 1, 1],[7, 1, 1, 1], [8, 1, 1, 1]]

# selection set of parameter
# methodList = [[1, 1, 1, 1], [1, 2, 1, 1], [1, 3, 1, 1], [1, 4, 1, 1]]

# recombinaison set of parameter
#methodList = [[1, 1, 1, 1], [1, 1, 2, 1], [1, 1, 3, 1]]

# reinsertion set of parameter
# methodList = [[1, 1, 1, 1], [1, 1, 1, 2]]

# test
methodList = [[7, 1, 1, 1]]

displayMoy = False
Size = 100
nbCycleMax = 2000
np.random.seed(9001)


# 4 tab to keep probability of operator
DataOP = []

for methodElt in methodList:

    mutationType = methodElt[0]
    selectionType = methodElt[1]
    recombinationType = methodElt[2]
    reinsertionType = methodElt[3]
    Label = ""

    select = selSwitch.get(selectionType, lambda: twoBest)
    Label += select.__name__ + ", "

    recombine = recSwitch.get(recombinationType, lambda: monoPoint)
    Label += recombine.__name__ + ", "

    mutation = mutSwitch.get(mutationType, lambda: oneflip)
    Label += mutation.__name__ + ", "

    reinsertion = reiSwitch.get(reinsertionType, lambda: bestoflower())
    Label += reinsertion.__name__ + ", "

    initialisation(Size, 100, 40, 100, nbCycleMax)
    evaluate()
    while terminaison():
        #selection

        Parents = select()

        #Recombine

        Childrens = recombine(Parents)

        #mutation

        mutation()

        #evaluate children
        evaluatechildren()

        #Reinsertion

        reinsertion()

        #graph Value
        evaluate()
        nbCycle = nbCycle +1
        affichage = "nombre de tour effectuer : "+str(nbCycle)+"/"+str(nbCycleMax)
        #print(affichage)
        x.append(nbCycle)
        y.append(CurrentBest())
        moyY.append(np.mean(CurrentEval))
    print(Label)
 #   plt.plot(x, y, label=Label)

    fichier = open(Label+".txt", "a")

    for elt in y:
        fichier.write(str(elt)+";")

    fichier.write("\n")
    fichier.close()

    if displayMoy:
        Label = "moy : " + Label
        plt.plot(x, moyY, label=Label)

    if mutation == flipsAdaptatifPursuit:
        fichierOP = open("dataOPAPW.txt", "a")
        for iop in range(0,NbrOP):
            for elt in DataOP:
                fichierOP.write(str(elt[iop]) + ";")
            fichierOP.write("\n")
        fichierOP.write("\n")
        fichierOP.close()
#plt.legend()
#plt.show()
# endregion Main


# for i in {1..20}; do python3 Main.py $i; done