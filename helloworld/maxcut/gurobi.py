import numpy as np
import torch as th
from gurobipy import *
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
from utils import calc_avg_std_of_obj
# If running_duration (seconds) is not None, the new file name should include it.
def write_result_gurobi(model, filename: str = 'result/result', running_duration: int = None):
    if filename.split('/')[0] == 'data':
        filename = calc_result_file_name(filename)
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    if running_duration is None:
        new_filename = filename + '.txt'
    else:
        new_filename = filename + '_' + str(int(running_duration))
    with open(f"{new_filename}.txt", 'w', encoding="UTF-8") as new_file:
        new_file.write(f"obj: {model.objVal}\n")
        # new_file.write(f"time_limit: {time_limit}\n")
        time_limit = model.getParamInfo("TIME_LIMIT")
        new_file.write(f"time_limit: {time_limit}\n")

        vars = model.getVars()
        new_file.write('values of vars: \n')
        for var in vars:
            new_file.write(f'{var.VarName}: {var.x}\n')
    model.write(f"{new_filename}.mst")
    model.write(f"{new_filename}.lp")
    model.write(f"{new_filename}.mps")
    model.write(f"{new_filename}.sol")

def run_using_gurobi(filename: str, time_limit: int, plot_fig: bool = False):
    model = Model("maxcut")

    graph = read_txt_as_networkx_graph(filename)

    adjacency_matrix = nx.adjacency_matrix(graph)
    num_nodes = nx.number_of_nodes(graph)
    nodes = list(range(num_nodes))

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="y")
    model.setObjective(quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                    GRB.MAXIMIZE)


    # constrs
    for j in nodes:
        for i in range(0, j):
            model.addConstr(y[(i, j)] - x[i] - x[j] <= 0, name='C0a_' + str(i) + '_' + str(j))
            model.addConstr(y[(i, j)] + x[i] + x[j] <= 2, name='C0b_' + str(i) + '_' + str(j))
    model.setParam('TimeLimit', time_limit)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if
                                c.getAttr(GRB.Attr.IISConstr) > 0]
        print('infeasibleConstrName: {}'.format(infeasibleConstrName))
        model.write('result/model.ilp')
        sys.exit()

    elif model.getAttr('SolCount') >= 1:  # get the SolCount:
        write_result_gurobi(model, filename, time_limit)

    num_vars = model.getAttr(GRB.Attr.NumVars)
    num_constrs = model.getAttr(GRB.Attr.NumConstrs)
    print(f'num_vars: {num_vars}, num_constrs: {num_constrs}')
    print('obj:', model.getObjective().getValue())
    vars = model.getVars()


    if model.getAttr('SolCount') == 0:  # model.getAttr(GRB.Attr.SolCount)
        print("No solution.")
    print("SolCount: ", model.getAttr('SolCount'))
    # except Exception as e:
    #     print("Exception!")

    scores = [model.getObjective().getValue()]
    alg_name = 'Gurobi'
    if plot_fig:
        plot_fig(scores, alg_name)
    print()

def run_gurobi_over_multiple_files(prefix: str, time_limits: List[int], directory_data: str = 'data', directory_result: str = 'result'):
    files = calc_txt_files_with_prefix(directory_data, prefix)
    for i in range(len(files)):
        print(f'The {i}-th file: {files[i]}')
        for j in range(len(time_limits)):
            run_using_gurobi(files[i], time_limits[j])
    calc_avg_std_of_obj(directory_result, prefix)

if __name__ == '__main__':
    import sys

    select_single_file = False
    if select_single_file:
        filename = 'data/syn_30_110.txt'
        run_using_gurobi(filename)
    else:
        prefix = 'syn_100_'
        time_limits = [3600, 3600 * 5, 3600 * 10]
        run_gurobi_over_multiple_files(prefix, time_limits)

    pass

