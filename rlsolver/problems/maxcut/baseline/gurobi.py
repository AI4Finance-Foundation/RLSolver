import sys
sys.path.append('../')
from gurobipy import *
import os
import copy
from typing import List
import networkx as nx
import time
import sys
import matplotlib.pyplot as plt

from rlsolver.problems.maxcut.util import read_nxgraph
from rlsolver.problems.maxcut.util import calc_txt_files_with_prefix
from rlsolver.problems.maxcut.util import calc_result_file_name
from rlsolver.problems.maxcut.util import calc_avg_std_of_objs
from rlsolver.problems.maxcut.util import plot_fig
from rlsolver.problems.maxcut.util import fetch_node
from rlsolver.problems.maxcut.util import (transfer_float_to_binary,
                  transfer_nxgraph_to_adjacencymatrix,
                  obtain_first_number)
from rlsolver.problems.maxcut.config import *

# 定义回调函数，每隔一段时间将当前找到的最佳可行解输出到当前目录下以 solution 开头
# 的文件中。同时，将当前进展输出到 report.txt 报告中。
def mycallback(model, where):
    if where == GRB.Callback.MIPSOL:
        # MIP solution callback
        currentTime = time.time()
        running_duation = int((currentTime - model._startTime) / model._interval) * model._interval

        # Statistics
        objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        # gap = abs(obj - objbnd) / (obj + 1e-6)

        # Export solution to a file
        # solutionfile = open("solution_" + str(running_duation) + ".txt", 'w')

        # filename = copy.deepcopy(model._reportFile.name) # ok, successful
        filename = copy.deepcopy(model._attribute['result_filename'])
        filename = filename.replace('.txt', '')
        filename = filename + '_' + str(running_duation) + '.txt'

        # vars = model.getVars()
        # nodes: List[int] = []
        # values: List[int] = []
        # for var in vars:
        #     node = fetch_node(var.VarName)
        #     if node is None:
        #         break
        #     value = transfer_float_to_binary(var.x)
        #     nodes.append(node)
        #     values.append(value)

        if GUROBI_INTERVAL is not None and running_duation < GUROBI_INTERVAL:
            return
        with open(filename, 'w', encoding="UTF-8") as new_file:
            write_statistics_in_mycallback(model, new_file, add_slash=True)
            new_file.write(f"// num_nodes: {model._attribute['num_nodes']}\n")
            # for i in range(len(nodes)):
            #     new_file.write(f"{nodes[i] + 1} {values[i] + 1}\n")
            varlist = [v for v in model.getVars() if 'x' in v.VarName]
            soln = model.cbGetSolution(varlist)
            for i in range(len(varlist)):
                value = int(round(soln[i]) + 1) if not GUROBI_VAR_CONTINUOUS else soln[i]
                new_file.write(f"{i + 1}  {value}\n")
            # for var, soln in zip(varlist, soln):
            #     solutionfile.write('%s %d\n' % (var.VarName, soln))


        # varlist = model.getVars()
        # soln = model.cbGetSolution(varlist)
        # solutionfile.write('Objective %e\n' % obj)
        # for var, soln in zip(varlist, soln):
        #     solutionfile.write('%s %.16e\n' % (var.VarName, soln))
        # solutionfile.close()
        #
        # # Export statistics
        # msg = str(currentTime - model._startTime) + " : " + "Solution Obj: " + str(obj) + " Solution Gap: " + str(
        #     gap) + "\n"
        # model._reportFile.write(msg)
        # model._reportFile.flush()

# the file has been open
def write_statistics(model, new_file, add_slash = False):
    prefix = '// ' if add_slash else ''
    if PROBLEM == Problem.maximum_independent_set:
        from rlsolver.problems.maxcut.util import obj_maximum_independent_set
        solution = model._attribute['solution']
        graph = model._attribute['graph']
        obj = obj_maximum_independent_set(solution, graph)
        new_file.write(f"{prefix}obj: {obj}\n")
    else:
        new_file.write(f"{prefix}obj: {model.objVal}\n")
    new_file.write(f"{prefix}running_duration: {model.Runtime}\n")
    if not GUROBI_VAR_CONTINUOUS:
        new_file.write(f"{prefix}gap: {model.MIPGap}\n")
    new_file.write(f"{prefix}obj_bound: {model.ObjBound}\n")
    # new_file.write(f"time_limit: {time_limit}\n")
    time_limit = model.getParamInfo("TIME_LIMIT")
    new_file.write(f"{prefix}time_limit: {time_limit}\n")

def write_statistics_in_mycallback(model, new_file, add_slash = False):
    if model.getAttr('SolCount') == 0:
        return

    currentTime = time.time()
    running_duation = int((currentTime - model._startTime) / model._interval) * model._interval

    # Statistics
    objbnd = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
    obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
    gap = abs(obj - objbnd) / (obj + 1e-6)

    # varlist = model.getVars()
    # soln = model.cbGetSolution(varlist)
    # new_file.write('Objective %e\n' % obj)
    # for var, soln in zip(varlist, soln):
    #     new_file.write('%s %.16e\n' % (var.VarName, soln))
    # new_file.close()

    # # Export statistics
    # msg = str(currentTime - model._startTime) + " : " + "Solution Obj: " + str(obj) + " Solution Gap: " + str(
    #     gap) + "\n"
    # model._reportFile.write(msg)
    # model._reportFile.flush()

    prefix = '// ' if add_slash else ''
    new_file.write(f"{prefix}obj: {obj}\n")
    new_file.write(f"{prefix}running_duration: {running_duation}\n")
    new_file.write(f"{prefix}gap: {gap}\n")
    new_file.write(f"{prefix}obj_bound: {objbnd}\n")
    # new_file.write(f"time_limit: {time_limit}\n")
    time_limit = model.getParamInfo("TIME_LIMIT")
    # time_limit2 = model.params['TimeLimit']
    new_file.write(f"{prefix}time_limit: {time_limit}\n")

# if filename = '../result/barabasi_albert_100_ID0.txt', running_duration = 100,
#
def write_result_gurobi(model, filename: str = './result/result', running_duration: int = None):
    directory = filename.split('/')[0]
    if not os.path.exists(directory):
        os.mkdir(directory)
    add_tail = '_' + str(int(running_duration)) if running_duration is not None else None
    new_filename = calc_result_file_name(filename, add_tail)

    vars = model.getVars()
    nodes: List[int] = []
    values: List[int] = []
    for var in vars:
        if "x" not in var.VarName:
            continue
        node = fetch_node(var.VarName)
        if node is None:
            break
        if GUROBI_VAR_CONTINUOUS:
            value = var.x
        else:
            value = transfer_float_to_binary(var.x)
        nodes.append(node)
        values.append(value)
    with open(new_filename, 'w', encoding="UTF-8") as new_file:
        model._attribute['solution'] = values
        write_statistics(model, new_file, True)
        new_file.write(f"// num_nodes: {len(nodes)}\n")
        for i in range(len(nodes)):
            if GUROBI_VAR_CONTINUOUS or PROBLEM == Problem.minimum_vertex_cover:
                new_file.write(f"{nodes[i] + 1} {values[i]}\n")
            else:
                new_file.write(f"{nodes[i] + 1} {values[i] + 1}\n")

    if_write_others = False
    if if_write_others:
        with open(f"{new_filename}.sta", 'w', encoding="UTF-8") as new_file:
            write_statistics(model, new_file, False)
        with open(f"{new_filename}.sov", 'w', encoding="UTF-8") as new_file:
            new_file.write('values of vars: \n')
            vars = model.getVars()
            for var in vars:
                new_file.write(f'{var.VarName}: {var.solution}\n')
        model.write(f"{new_filename}.mst")
        model.write(f"{new_filename}.lp")
        model.write(f"{new_filename}.mps")
        model.write(f"{new_filename}.sol")

def run_using_gurobi(filename: str, time_limit: int = None, plot_fig_: bool = False):
    model = Model("maxcut")

    graph = read_nxgraph(filename)
    edges = list(graph.edges)
    subax1 = plt.subplot(111)
    nx.draw_networkx(graph, with_labels=True)
    # plt.show() # if show the fig, remove this comment
    if plot_fig_:
        plt.show()

    adjacency_matrix = transfer_nxgraph_to_adjacencymatrix(graph)
    num_nodes = nx.number_of_nodes(graph)
    nodes = list(range(num_nodes))

    if PROBLEM == Problem.maxcut:
        y_lb = adjacency_matrix.min()
        y_ub = adjacency_matrix.max()
        x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
        if GUROBI_MILP_QUBO == 0:
            y = model.addVars(num_nodes, num_nodes, vtype=GRB.CONTINUOUS, lb=y_lb, ub=y_ub, name="y")
            model.setObjective(quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                            GRB.MAXIMIZE)
        else:
            model.setObjective(
                quicksum(quicksum(adjacency_matrix[(i, j)] * (0.5 - 2 * (x[i] - 0.5) * (x[j] - 0.5)) for i in range(0, j)) for j in nodes),
                GRB.MAXIMIZE)
    elif PROBLEM == Problem.graph_partitioning:
        if GUROBI_MILP_QUBO == 0:
            y_lb = adjacency_matrix.min()
            y_ub = adjacency_matrix.max()
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            y = model.addVars(num_nodes, num_nodes, vtype=GRB.CONTINUOUS, lb=y_lb, ub=y_ub, name="y")
            model.setObjective(quicksum(quicksum(adjacency_matrix[(i, j)] * y[(i, j)] for i in range(0, j)) for j in nodes),
                               GRB.MINIMIZE)
        else:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            coef_A = len(edges) + 10
            model.setObjective(coef_A * quicksum(2 * (x[k] - 0.5) for k in nodes) * quicksum(2 * (x[k] - 0.5) for k in nodes)
                +quicksum(quicksum(adjacency_matrix[(i, j)] * (0.5 - 2 * (x[i] - 0.5) * (x[j] - 0.5)) for i in range(0, j)) for j in nodes),
                GRB.MINIMIZE)
    elif PROBLEM == Problem.minimum_vertex_cover:
        if GUROBI_MILP_QUBO == 0:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            model.setObjective(quicksum(x[j] for j in nodes),
                               GRB.MINIMIZE)
        else:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            coef_A = len(nodes) + 10
            model.setObjective(coef_A * quicksum(quicksum(adjacency_matrix[(i, j)] * (1 - x[i]) * (1 - x[j]) for i in range(0, j)) for j in nodes)
                               + quicksum(x[j] for j in nodes),
                               GRB.MINIMIZE)
    elif PROBLEM == Problem.maximum_independent_set:
        if GUROBI_MILP_QUBO == 0:
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            model.setObjective(quicksum(x[j] for j in nodes),
                               GRB.MAXIMIZE)
        else:
            coef_B1 = -1
            coef_B2 = 3
            x = model.addVars(num_nodes, vtype=GRB.BINARY, name="x")
            model.setObjective(-quicksum(x[j] for j in nodes) + coef_B1 * quicksum((2 - x[i] - x[j]) * (2 - x[i] - x[j]) for (i, j) in edges)
                               + coef_B2 * quicksum((1 - x[i] - x[j]) * (1 - x[i] - x[j]) for (i, j) in edges),
                               GRB.MINIMIZE)

    # constrs if using MILP
    if GUROBI_MILP_QUBO == 0:
        if PROBLEM == Problem.maxcut:
            # y_{i, j} = x_i XOR x_j
            for j in nodes:
                for i in range(0, j):
                    model.addConstr(y[(i, j)] <= x[i] + x[j], name='C0b_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] <= 2 - x[i] - x[j], name='C0a_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= x[i] - x[j], name='C0c_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= -x[i] + x[j], name='C0d_' + str(i) + '_' + str(j))
        elif PROBLEM == Problem.graph_partitioning:
            # y_{i, j} = x_i XOR x_j
            for j in nodes:
                for i in range(0, j):
                    model.addConstr(y[(i, j)] <= x[i] + x[j], name='C0b_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] <= 2 - x[i] - x[j], name='C0a_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= x[i] - x[j], name='C0c_' + str(i) + '_' + str(j))
                    model.addConstr(y[(i, j)] >= -x[i] + x[j], name='C0d_' + str(i) + '_' + str(j))
            model.addConstr(quicksum(x[j] for j in nodes) == num_nodes / 2, name='C1')
        elif PROBLEM == Problem.minimum_vertex_cover:
            for i in range(len(edges)):
                node1, node2 = edges[i]
                model.addConstr(x[node1] + x[node2] >= 1, name=f'C0_{node1}_{node2}')
        elif PROBLEM == Problem.maximum_independent_set:
            for i in range(len(edges)):
                node1, node2 = edges[i]
                model.addConstr(x[node1] + x[node2] <= 1, name=f'C0_{node1}_{node2}')


    if time_limit is not None:
        model.setParam('TimeLimit', time_limit)

    # reportFile = open('../result/', 'w')

    result_filename = calc_result_file_name(filename)

    model._startTime = time.time()
    # model._reportFile = open(result_filename, 'w')
    model._interval = GUROBI_INTERVAL  # 每隔一段时间输出当前可行解，单位秒
    model._attribute = {'data_filename': filename, 'result_filename': result_filename, 'num_nodes': num_nodes}

    if GUROBI_VAR_CONTINUOUS:
        # for v in model.getVars():
        #     v.setAttr('vtype', GRB.CONTINUOUS)
        model.update()
        r = model.relax()
        r.update()
        if GUROBI_INTERVAL is None:
            r.optimize()
        else:
            r.optimize(mycallback)

        if_write_others = False
        if if_write_others:
            r.write("../result/result.lp")
            r.write("../result/result.mps")
            r.write("../result/result.sol")
        x_values = []
        # for i in range(num_nodes):
        #     var = r.getVarByName(x[i].VarName)
        #     x_values.append(var.x)
        # var = r.getVarByName(x.VarName)
        vars_in_model = [var for var in model.getVars() if "x" in var.VarName]
        name = "x"
        names_to_retrieve = [f"{name}[{i}]" for i in range(num_nodes)]

        for i in range(num_nodes):
            var = r.getVarByName(names_to_retrieve[i])
            x_values.append(var.solution)
        print(f'values of x: {x_values}')
        return x_values

    if GUROBI_INTERVAL is None:
        model.optimize()
    else:
        model.optimize(mycallback)

    if model.status == GRB.INFEASIBLE:
        model.computeIIS()
        infeasibleConstrName = [c.getAttr('ConstrName') for c in model.getConstrs() if
                                c.getAttr(GRB.Attr.IISConstr) > 0]
        print('infeasibleConstrName: {}'.format(infeasibleConstrName))
        model.write('../result/model.ilp')
        sys.exit()

    elif model.getAttr('SolCount') >= 1:  # get the SolCount:
        # result_filename = '../result/result'
        model._attribute['graph'] = graph
        write_result_gurobi(model, result_filename, time_limit)

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
    if plot_fig_:
        plot_fig(scores, alg_name)

    print(f"model.Runtime: {model.Runtime}")
    print()

    x_values = []
    for i in range(num_nodes):
        x_values.append(x[i].x)
    print(f'values of x: {x_values}')
    return x_values

def run_gurobi_over_multiple_files(prefixes: List[str], time_limits: List[int], directory_data: str = 'data', directory_result: str = 'result'):
    for prefix in prefixes:
        files = calc_txt_files_with_prefix(directory_data, prefix)
        files.sort()
        for i in range(len(files)):
            print(f'The {i}-th file: {files[i]}')
            for j in range(len(time_limits)):
                run_using_gurobi(files[i], time_limits[j])
    avg_std = calc_avg_std_of_objs(directory_result, prefixes, time_limits)

if __name__ == '__main__':
    select_single_file = False
    if select_single_file:
        filename = '../data/syn/syn_10_21.txt'
        time_limits = GUROBI_TIME_LIMITS
        run_using_gurobi(filename, time_limit=time_limits[0], plot_fig_=True)
        directory = '../result'
        prefixes = ['syn_10_']
        avg_std = calc_avg_std_of_objs(directory, prefixes, time_limits)
    else:
        if_use_syn = True
        # time_limits = GUROBI_TIME_LIMITS
        # time_limits = [10 * 60, 20 * 60, 30 * 60, 40 * 60, 50 * 60, 60 * 60]
        if if_use_syn:
            # prefixes = ['syn_10_', 'syn_50_', 'syn_100_', 'syn_300_', 'syn_500_', 'syn_700_', 'syn_900_', 'syn_1000_', 'syn_3000_', 'syn_5000_', 'syn_7000_', 'syn_9000_', 'syn_10000_']
            directory_data = '../data/syn'
            prefixes = ['syn_10_']

        if_use_syndistri = False
        if if_use_syndistri:
            directory_data = '../data/syn_BA'
            prefixes = ['barabasi_albert_100_']
            # prefixes = ['syn_100_']
            # directory_data = '../data/syn'

        directory_result = '../result'
        run_gurobi_over_multiple_files(prefixes, GUROBI_TIME_LIMITS, directory_data, directory_result)
        avg_std = calc_avg_std_of_objs(directory_result, prefixes, GUROBI_TIME_LIMITS)

    pass

