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
from env.maxcut_env import MaxcutEnv
from MaxCut_H2O import MaxCutEnv2

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

def write_result(model, filename='result/result'):
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    file = filename + '.txt'
    with open(file, 'w', encoding="UTF-8") as file:
        file.write(f"obj: {model.objVal}\n")
        vars = model.getVars()
        file.write('values of vars: \n')
        for var in vars:
            file.write(f'{var.VarName}: {var.x}\n')
    model.write(f"{filename}.mst")
    model.write(f"{filename}.lp")
    model.write(f"{filename}.mps")
    model.write(f"{filename}.sol")

def run_using_gurobi(filename: str):
    model = Model("maxcut")

    graph = read_txt_as_networkx_graph(filename)

    adjacency_matrix = nx.adj_matrix(graph)
    num_nodes = nx.number_of_nodes(graph)
    nodes = list(range(num_nodes))

    x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
    y = model.addVars(num_nodes, num_nodes, vtype=GRB.BINARY, name="y")
    model.setObjective(quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                    GRB.MAXIMIZE)
    #model.setParam('TimeLimit', 10)

    # constrs
    for j in nodes:
        for i in range(0, j):
            model.addConstr(y[(i, j)] - x[i] - x[j] <= 0, name='C0a_' + str(i) + '_' + str(j))
            model.addConstr(y[(i, j)] + x[i] + x[j] <= 2, name='C0b_' + str(i) + '_' + str(j))

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if
                                c.getAttr(GRB.Attr.IISConstr) > 0]
        print('infeasibleConstrName: {}'.format(infeasibleConstrName))
        model.write('result/model.ilp')
        sys.exit()

    elif model.getAttr('SolCount') >= 1:  # get the SolCount:
        write_result(model)

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
    alg_name = 'gurobi'
    plot_fig(scores, alg_name)
    print()



if __name__ == '__main__':
    import sys
    filename = 'data/syn_30_110.txt'
    run_using_gurobi(filename)

    pass

