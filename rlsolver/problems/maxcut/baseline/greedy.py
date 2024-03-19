import sys
sys.path.append('../')

import copy
import time
from typing import List, Union
import numpy as np
import multiprocessing as mp
import networkx as nx
from rlsolver.problems.maxcut.util import read_nxgraph
from rlsolver.problems.maxcut.util import obj_maxcut, obj_graph_partitioning, obj_minimum_vertex_cover, obj_maximum_independent_set
from rlsolver.problems.maxcut.util import plot_fig
from rlsolver.problems.maxcut.util import transfer_nxgraph_to_weightmatrix
from rlsolver.problems.maxcut.util import cover_all_edges
from rlsolver.problems.maxcut.util import run_greedy_over_multiple_files
from rlsolver.problems.maxcut.config import *

def split_list(my_list: List[int], chunk_size: int):
    res = []
    for i in range(0, len(my_list), chunk_size):
        res.append(my_list[i: i + chunk_size])
    return res

# the len of result may not be mp.cpu_count()
def split_list_equally_by_cpus(my_list: List[int]):
    res = []
    num_cpus = mp.cpu_count()
    chunk_size = int(np.ceil(len(my_list) / num_cpus))
    for i in range(0, len(my_list), chunk_size):
        res.append(my_list[i: i + chunk_size])
    return res

def split_list_equally(my_list: List[int], chunk_size: int):
    res = []
    for i in range(0, len(my_list), chunk_size):
        res.append(my_list[i: i + chunk_size])
    return res

def traverse_in_greedy_maxcut(curr_solution, selected_nodes, graph):
    traversal_scores = []
    traversal_solutions = []
    # calc the new solution when moving to a new node. Then store the scores and solutions.
    for node in selected_nodes:
        new_solution = copy.deepcopy(curr_solution)
        # search a new solution and calc obj
        new_solution[node] = (new_solution[node] + 1) % 2
        new_score = obj_maxcut(new_solution, graph)
        traversal_scores.append(new_score)
        traversal_solutions.append(new_solution)
    return traversal_scores, traversal_solutions


# init_solution is useless
def greedy_maxcut(init_solution, num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    assert sum(init_solution) == 0
    if num_steps is None:
        num_steps = num_nodes
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []
    for iteration in range(num_nodes):
        if iteration >= num_steps:
            break
        score = obj_maxcut(curr_solution, graph)
        print(f"iteration: {iteration}, score: {score}")
        traversal_scores = []
        traversal_solutions = []
        # calc the new solution when moving to a new node. Then store the scores and solutions.
        use_multiprocessing = False
        if use_multiprocessing:
            pass
            split_nodess = split_list_equally_by_cpus(nodes)
            pool = mp.Pool(len(split_nodess))
            # print(f'len split_nodess: {len(split_nodess)}')
            results = []
            for split_nodes in split_nodess:
                results.append(pool.apply_async(traverse_in_greedy_maxcut, (curr_solution, split_nodes, graph)))
            for result in results:
                tmp_traversal_scores, tmp_traversal_solutions = result.get()
                # print(f'tmp_traversal_scores: {tmp_traversal_scores}, tmp_traversal_solutions: {tmp_traversal_solutions}')
                traversal_scores.extend(tmp_traversal_scores)
                traversal_solutions.extend(tmp_traversal_solutions)
        else:
            for node in nodes:
                new_solution = copy.deepcopy(curr_solution)
                # search a new solution and calc obj
                new_solution[node] = (new_solution[node] + 1) % 2
                new_score = obj_maxcut(new_solution, graph)
                traversal_scores.append(new_score)
                traversal_solutions.append(new_solution)
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score > curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
        else:
            break
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_graph_partitioning(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    assert num_steps is None
    start_time = time.time()
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_graph_partitioning(curr_solution, graph)
    init_score = curr_score
    scores = []
    for i in range(num_nodes):
        node1 = nodes[i]
        traversal_scores = []
        traversal_solutions = []
        for j in range(i + 1, num_nodes):
            node2 = nodes[j]
            if curr_solution[node1] == curr_solution[node2]:
                continue
            new_solution = copy.deepcopy(curr_solution)
            tmp = new_solution[node1]
            new_solution[node1] = new_solution[node2]
            new_solution[node2] = tmp
            new_score = obj_graph_partitioning(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        if len(traversal_scores) == 0:
            continue
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score > curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
        else:
            break
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores


def greedy_minimum_vertex_cover(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    assert num_steps is None
    start_time = time.time()
    assert sum(init_solution) == 0
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_minimum_vertex_cover(curr_solution, graph)
    init_score = curr_score
    scores = []
    iter = 0
    unselected_nodes = list(graph.nodes())
    while True:
        cover_all = cover_all_edges(curr_solution, graph)
        if cover_all:
            break
        max_degree = 0
        best_node = -INF
        for node in unselected_nodes:
            degree = graph.degree(node)
            if degree > max_degree:
                max_degree = degree
                best_node = node
        if max_degree > 0:
            curr_solution[best_node] = 1
            unselected_nodes.remove(best_node)
        iter += 1
        if iter > num_nodes:
            break
    curr_score = obj_minimum_vertex_cover(curr_solution, graph)
    print("score, init_score", curr_score, init_score)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

def greedy_maximum_independent_set(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    num_nodes = int(graph.number_of_nodes())
    nodes = list(range(num_nodes))
    if num_steps is None:
        num_steps = num_nodes
    if init_solution is None:
        init_solution = [0] * num_nodes
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maximum_independent_set(curr_solution, graph)
    init_score = curr_score
    scores = []
    for i in range(num_steps):
        traversal_scores = []
        traversal_solutions = []
        for node in range(num_nodes):
            if curr_solution[node] == 1:
                continue
            new_solution = copy.deepcopy(curr_solution)
            new_solution[node] = (new_solution[node] + 1) % 2
            new_score = obj_maximum_independent_set(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        if len(traversal_scores) == 0:
            continue
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score > curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
        else:
            break
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':
    # read data
    graph = read_nxgraph('../data/syn/syn_10_21.txt')
    weightmatrix = transfer_nxgraph_to_weightmatrix(graph)
    # run alg
    num_steps = 30
    alg_name = 'GR'

    if_run_one_case = False
    if if_run_one_case:
        # maxcut
        if PROBLEM == Problem.maxcut:
            # init_solution = None
            init_solution = [0] * graph.number_of_nodes()
            gr_score, gr_solution, gr_scores = greedy_maxcut(init_solution, num_steps, graph)

        # graph_partitioning
        if PROBLEM == Problem.graph_partitioning:
            init_solution = [0] * int(graph.number_of_nodes() / 2) + [1] * int(graph.number_of_nodes() / 2)
            num_steps = 100
            gr_score, gr_solution, gr_scores = greedy_graph_partitioning(init_solution, num_steps, graph)

        if PROBLEM == Problem.minimum_vertex_cover:
            init_solution = [0] * graph.number_of_nodes()
            gr_score, gr_solution, gr_scores = greedy_minimum_vertex_cover(init_solution, num_steps, graph)
            obj = obj_minimum_vertex_cover(gr_solution, graph)
            print('obj: ', obj)

        if PROBLEM == Problem.maximum_independent_set:
            init_solution = None
            num_steps = None
            gr_score, gr_solution, gr_scores = greedy_maximum_independent_set(init_solution, num_steps, graph)
            obj = obj_maximum_independent_set(gr_solution, graph)
            print('obj: ', obj)

    else:
        if PROBLEM == Problem.maxcut:
            alg = greedy_maxcut
        elif PROBLEM == Problem.graph_partitioning:
            alg = greedy_graph_partitioning
        elif PROBLEM == Problem.minimum_vertex_cover:
            alg = greedy_minimum_vertex_cover
        elif PROBLEM == Problem.maximum_independent_set:
            alg = greedy_maximum_independent_set

        alg_name = "greedy"
        num_steps = None
        directory_data = '../data/syn_BA'
        prefixes = ['barabasi_albert_100_']
        set_init_0 = True
        scoress = run_greedy_over_multiple_files(alg, alg_name, num_steps, set_init_0, directory_data, prefixes)
        print(f"scoress: {scoress}")

        # plot fig
        plot_fig_ = False
        if plot_fig_:
            for scores in scoress:
                plot_fig(scores, alg_name)




