import sys
sys.path.append('../')
import copy
import time
from typing import List, Union
import numpy as np
import random
import networkx as nx
from rlsolver.problems.maxcut.util import read_nxgraph
from rlsolver.problems.maxcut.util import obj_maxcut, obj_graph_partitioning, obj_minimum_vertex_cover, cover_all_edges
from rlsolver.problems.maxcut.util import write_result
from rlsolver.problems.maxcut.util import plot_fig
from rlsolver.problems.maxcut.util import run_simulated_annealing_over_multiple_files
from rlsolver.problems.maxcut.config import *
def simulated_annealing(init_temperature: int, num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('simulated_annealing')
    if PROBLEM in [Problem.maxcut, Problem.graph_partitioning]:
        init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
    if PROBLEM == Problem.minimum_vertex_cover:
        from greedy import greedy_minimum_vertex_cover
        _, init_solution, _ = greedy_minimum_vertex_cover([0] * int(graph.number_of_nodes()), int(graph.number_of_nodes()), graph)
        assert cover_all_edges(init_solution, graph)
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    if PROBLEM == Problem.maxcut:
        curr_score = obj_maxcut(curr_solution, graph)
    elif PROBLEM == Problem.graph_partitioning:
        curr_score = obj_graph_partitioning(curr_solution, graph)
    elif PROBLEM == Problem.minimum_vertex_cover:
        curr_score = obj_minimum_vertex_cover(curr_solution, graph)
    init_score = curr_score
    num_nodes = len(init_solution)
    scores = []
    for k in range(num_steps):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        new_solution = copy.deepcopy(curr_solution)
        if PROBLEM == Problem.maxcut:
            idx = np.random.randint(0, num_nodes)
            new_solution[idx] = (new_solution[idx] + 1) % 2
            new_score = obj_maxcut(new_solution, graph)
        elif PROBLEM == Problem.graph_partitioning:
            while True:
                idx = np.random.randint(0, num_nodes)
                index2 = np.random.randint(0, num_nodes)
                if new_solution[idx] != new_solution[index2]:
                    break
            print(f"new_solution[index]: {new_solution[idx]}, new_solution[index2]: {new_solution[index2]}")
            tmp = new_solution[idx]
            new_solution[idx] = new_solution[index2]
            new_solution[index2] = tmp
            new_score = obj_graph_partitioning(new_solution, graph)
        elif PROBLEM == Problem.minimum_vertex_cover:
            iter = 0
            max_iter = 3 * graph.number_of_nodes()
            index = None
            while True:
                iter += 1
                if iter >= max_iter:
                    break
                indices_eq_1 = []
                for i in range(len(new_solution)):
                    if new_solution[i] == 1:
                        indices_eq_1.append(i)
                idx = np.random.randint(0, len(indices_eq_1))
                new_solution2 = copy.deepcopy(new_solution)
                new_solution2[indices_eq_1[idx]] = 0
                if cover_all_edges(new_solution2, graph):
                    index = indices_eq_1[idx]
                    break
            if index is not None:
                new_solution[index] = 0
            new_score = obj_minimum_vertex_cover(new_solution, graph, False)
        scores.append(new_score)
        delta_e = curr_score - new_score
        if delta_e < 0:
            curr_solution = new_solution
            curr_score = new_score
        else:
            prob = np.exp(- delta_e / (temperature + 1e-6))
            if prob > random.random():
                curr_solution = new_solution
                curr_score = new_score
    print("score, init_score of simulated_annealing", curr_score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':


    # run alg
    # init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))

    if_run_one_case = False
    if if_run_one_case:
        # read data
        graph = read_nxgraph('../data/syn/syn_50_176.txt')
        init_temperature = 4
        num_steps = 2000
        sa_score, sa_solution, sa_scores = simulated_annealing(init_temperature, num_steps, graph)
        # write result
        write_result(sa_solution, '../result/result.txt')
        # plot fig
        alg_name = 'SA'
        plot_fig(sa_scores, alg_name)


    alg = simulated_annealing
    alg_name = 'simulated_annealing'
    init_temperature = 4
    num_steps = 30
    directory_data = '../data/syn_BA'
    prefixes = ['barabasi_albert_100_ID0']
    run_simulated_annealing_over_multiple_files(alg, alg_name, init_temperature, num_steps, directory_data, prefixes)





