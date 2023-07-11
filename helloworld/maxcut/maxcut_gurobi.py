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
from MaxCut_H2O import GraphMaxCutEnv



RESULT_GUROBI_DIR = 'gurobi_result'
RESULT_DIR = 'gurobi_result'

def write_result_gurobi(model, env: GraphMaxCutEnv):
    file_name = RESULT_GUROBI_DIR + "_NUM_NODES=" + str(env.num_nodes) + "_" + str(env.num_edges) + ".txt"
    with open(file_name, 'w', encoding="UTF-8") as file:
        file.write(f"obj when NUM_NODES={env.num_nodes} NUM_EDGES={env.num_edges}: {model.objVal}")


def run_using_gurobi(env: GraphMaxCutEnv):
    model = Model("maxcut")
    node_indices = list(range(env.num_nodes))
    import pickle as pkl
    sparsity = float(sys.argv[2])

    adjacency_matrix_list = np.load(f'N{env.num_nodes}Sparsity{sparsity}.npy')
    # print(adjacency_matrix_list)
    for adj_id in range(1):
    # try:
        # print(adjacency_matrix_list)
        adjacency_matrix = adjacency_matrix_list
        x = model.addVars(env.num_nodes, vtype=GRB.BINARY, name="x")
        y = model.addVars(env.num_nodes, env.num_nodes, vtype=GRB.BINARY, name="y")
        model.setObjective(quicksum(quicksum(adjacency_matrix[i][j] * y[(i, j)] for i in range(0, j)) for j in node_indices),
                        GRB.MAXIMIZE)
        #model.setParam('TimeLimit', 10)

        # constrs
        for j in node_indices:
            for i in range(0, j):
                model.addConstr(y[(i, j)] - x[i] - x[j] <= 0, name='C0a_' + str(i) + '_' + str(j))
                model.addConstr(y[(i, j)] + x[i] + x[j] <= 2, name='C0b_' + str(i) + '_' + str(j))

        model.optimize()

        if model.status == GRB.INFEASIBLE:
            model.computeIIS()
            infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if c.getAttr(GRB.Attr.IISConstr) > 0]
            print('infeasibleConstrName: {}'.format(infeasibleConstrName))
            model.write(RESULT_DIR + '/model.ilp')
            sys.exit()
            
        elif model.getAttr('SolCount') >= 1:  # get the SolCount:
            model.write(RESULT_DIR + '/model.sol')
            write_result_gurobi(model, env.num_nodes)

        num_vars = model.getAttr(GRB.Attr.NumVars)
        num_constrs = model.getAttr(GRB.Attr.NumConstrs)
        print(model.getObjective().getValue())
        # print('numVars in model: {}'.format(num_vars))
        # print('numConstrs in model: {}'.format(num_constrs))
        
        if model.getAttr('SolCount') == 0:  # model.getAttr(GRB.Attr.SolCount)
            print("No solution.")
        print("SolCount: ", model.getAttr('SolCount'))
    # except Exception as e:
    #     print("Exception!")



if __name__ == '__main__':
    import sys
    env = GraphMaxCutEnv(graph_key='gset_14')

    run_using_gurobi(env)
    pass

