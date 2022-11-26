# problem formulation: TEAMS OF GLOBAL EQUILIBRIUM SEARCH ALGORITHMS FOR SOLVING WEIGHTED MAXIMUM CUT PROBLEM IN PARALLEL

import numpy as np
from gurobipy import *

import os


from config import (
NUMS_NODES,
ADJACENCY_MATRIX_DIR,
RESULT_DIR,
RESULT_GUROBI_DIR,
)
EXP_ID = 0

def calc_file(n: int):
    global EXP_ID
    return ADJACENCY_MATRIX_DIR + "/" + EXP_ID + "/adjacency.npy"
    #return ADJACENCY_MATRIX_DIR + "/NUM_NODES=" + str(n) + ".npy"

def generate_adjacency_matrix(n: int):
    adjacency_matrix = np.random.randint(0, 2, (n, n))
    for j in range(n):
        for i in range(0, j):
            adjacency_matrix[i][j] = adjacency_matrix[j][i]
    adjacency_matrix[adjacency_matrix == 0] = -10000
    file = calc_file(n)
    np.save(file, adjacency_matrix)
    pass


def read_adjacency_matrix(n: int):
    file = calc_file(n)
    adjacency_matrix = np.load(file)
    adjacency_matrix[adjacency_matrix == 0] = -10000
    return adjacency_matrix

def write_result_gurobi(model, n :int):
    file_name = RESULT_GUROBI_DIR + "_NUM_NODES=" + str(n) + "_" + EXP_ID + ".txt"
    with open(file_name, 'w', encoding="UTF-8") as file:
        file.write(f"obj when NUM_NODES={n}: {model.objVal}")


def run_using_gurobi(nums:list = NUMS_NODES):
    for n in nums:
        run_using_gurobi_fixed_num_nodes(n)

def run_using_gurobi_fixed_num_nodes(n: int):
    model = Model("maxcut")
    file = calc_file(n)
    node_indices = list(range(n))
    if os.patorch.exists(file):
        adjacency_matrix = read_adjacency_matrix(n)
    else:
        generate_adjacency_matrix(n)
        adjacency_matrix = read_adjacency_matrix(n)

    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(n, n, vtype=GRB.BINARY, name="y")
    model.setObjective(quicksum(quicksum(adjacency_matrix[i][j] * y[(i, j)] for i in range(0, j)) for j in node_indices),
                       GRB.MAXIMIZE)

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
    print('numVars in model: {}'.format(num_vars))
    print('numConstrs in model: {}'.format(num_constrs))
    
    if model.getAttr('SolCount') == 0:  # model.getAttr(GRB.Attr.SolCount)
        print("No solution.")
    print("SolCount: ", model.getAttr('SolCount'))



if __name__ == '__main__':
    import sys
    EXP_ID = sys.argv[1]
    run_using_gurobi(NUMS_NODES)
    pass

