import numpy as np
import torch as th
from pyscipopt import Model, quicksum, multidict
import sys
import os
import copy
import time

import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor
from typing import List
import random
from env.env import MaxcutEnv
from learn_to_anneal_2 import MaxCutEnv2

import copy
import time
from typing import List, Union
import numpy as np
from typing import List
import networkx as nx
from utils import read_txt_as_networkx_graph
from utils import obj_maxcut
from utils import write_result
from utils import plot_fig

def write_result_of_scip(model, filename='result/result'):
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    file = filename + '.txt'
    with open(file, 'w', encoding="UTF-8") as file:
        file.write(f"obj: {model.getObjVal()}\n")
        vars = model.getVars()
        file.write('values of vars: \n')
        for var in vars:
            file.write(f'{var.name}: {model.getVal(var)}\n')
    model.writeLP(f"{filename}.lp")
    model.writeStatistics(f"{filename}.sta")
    model.writeBestSol(f"{filename}.sol")
    # model.writeSol(f"{filename}.sol")

def run_using_scip(filename: str):
    model = Model("maxcut")

    graph = read_txt_as_networkx_graph(filename)

    adjacency_matrix = nx.adjacency_matrix(graph)
    num_nodes = nx.number_of_nodes(graph)
    nodes = list(range(num_nodes))

    x = {}
    y = {}
    for i in range(num_nodes):
        x[i] = model.addVar(vtype='B', name=f"x[{i}]")
    for i in range(num_nodes):
        for j in range(num_nodes):
            y[(i, j)] = model.addVar(vtype='B', name=f"y[{i}][{j}]")
    model.setObjective(quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                    'maximize')
    #model.setParam('TimeLimit', 10)

    # constrs
    for j in nodes:
        for i in range(0, j):
            model.addCons(y[(i, j)] - x[i] - x[j] <= 0, name='C0a_' + str(i) + '_' + str(j))
            model.addCons(y[(i, j)] + x[i] + x[j] <= 2, name='C0b_' + str(i) + '_' + str(j))

    model.optimize()


    if model.getStatus() == "optimal":
        write_result_of_scip(model)


    print('obj:', model.getObjVal())




    scores = [model.getObjVal()]
    alg_name = 'Scip'
    plot_fig(scores, alg_name)
    print()



if __name__ == '__main__':
    import sys
    filename = 'data/syn_30_110.txt'
    run_using_scip(filename)

    pass

