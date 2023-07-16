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
from utils import calc_txt_files_with_prefix
from utils import calc_result_file_name

# If running_duration (seconds) is not None, the new file name should include it.
def write_result_of_scip(model, filename: str = 'result/result', running_duration: int = None):
    if filename.split('/')[0] == 'data':
        filename = calc_result_file_name(filename)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    if running_duration is None:
        file = filename + '.txt'
    else:
        file = filename + '_' + str(int(running_duration)) + 's.txt'
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

def run_using_scip(filename: str, time_limit: int):
    start_time = time.time()
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

    # constrs
    for j in nodes:
        for i in range(0, j):
            model.addCons(y[(i, j)] - x[i] - x[j] <= 0, name='C0a_' + str(i) + '_' + str(j))
            model.addCons(y[(i, j)] + x[i] + x[j] <= 2, name='C0b_' + str(i) + '_' + str(j))

    model.setRealParam("limits/time", time_limit)
    model.optimize()


    # if model.getStatus() == "optimal":
    running_duration = time.time() - start_time
    write_result_of_scip(model, filename, time_limit)


    print('obj:', model.getObjVal())




    scores = [model.getObjVal()]
    alg_name = 'Scip'
    plot_fig(scores, alg_name)
    print()



if __name__ == '__main__':
    import sys
    select_single_file = False
    if select_single_file:
        filename = 'data/syn_30_110.txt'
        run_using_scip(filename)
    else:
        directory = 'data'
        prefix = 'syn_300_'
        files = calc_txt_files_with_prefix(directory, prefix)
        for i in range(len(files)):
            print(f'The {i}-th file: {files[i]}')
            time_limits = [6, 6 * 5, 6 * 10]
            for j in range(len(time_limits)):
                run_using_scip(files[i], time_limits[j])

    pass

