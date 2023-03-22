import numpy as np
import torch as th
from gurobipy import *

import os



NUMS_NODES = [20]
EXP_ID = 0

RESULT_GUROBI_DIR = 'gurobi_result'
RESULT_DIR = 'gurobi_result'

def write_result_gurobi(model, n :int):
    file_name = RESULT_GUROBI_DIR + "_NUM_NODES=" + str(n) + "_" + EXP_ID + ".txt"
    with open(file_name, 'w', encoding="UTF-8") as file:
        file.write(f"obj when NUM_NODES={n}: {model.objVal}")


def run_using_gurobi(nums:list = NUMS_NODES):
    for n in nums:
        run_using_gurobi_fixed_num_nodes(n)

def run_using_gurobi_fixed_num_nodes(n: int):
    model = Model("maxcut")
    node_indices = list(range(n))
    import pickle as pkl
    sparsity = 0.5

    adjacency_matrix_list = np.load(f'N{n}Sparsity{sparsity}.npy')
    print(adjacency_matrix_list)
    for adj_id in range(1):
    # try:
        print(adjacency_matrix_list)
        adjacency_matrix = adjacency_matrix_list
        x = model.addVars(n, vtype=GRB.BINARY, name="x")
        y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
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
            write_result_gurobi(model, n)

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
    EXP_ID = "0"
    run_using_gurobi(NUMS_NODES)
    pass

