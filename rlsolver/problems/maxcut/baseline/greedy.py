# comparison methods for maxcut: random walk, greedy, epsilon greedy, simulated annealing
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


def greedy(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('greedy')
    start_time = time.time()
    num_nodes = len(init_solution)
    nodes = list(range(num_nodes))
    curr_solution = copy.deepcopy(init_solution)
    curr_score: int = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    scores = []
    for iteration in range(num_nodes):
        if iteration >= num_steps:
            break
        print("iteration in greedy: ", iteration)
        traversal_scores = []
        traversal_solutions = []
        # calc the new solution when moving to a new node. Then store the scores and solutions.
        for node in nodes:
            new_solution = copy.deepcopy(curr_solution)
            new_solution[node] = (new_solution[node] + 1) % 2
            # calc the obj
            new_score = obj_maxcut(new_solution, graph)
            traversal_scores.append(new_score)
            traversal_solutions.append(new_solution)
        best_score = max(traversal_scores)
        index = traversal_scores.index(best_score)
        best_solution = traversal_solutions[index]
        if best_score >= curr_score:
            scores.append(best_score)
            curr_score = best_score
            curr_solution = best_solution
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", traversal_scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores


if __name__ == '__main__':
    # read data
    graph = read_txt_as_networkx_graph('../data/syn/syn_50_176.txt')

    # run alg
    init_solution = [0] * graph.number_of_nodes()
    num_steps = 30
    alg_name = 'GR'
    gr_score, gr_solution, gr_scores = greedy(init_solution, num_steps, graph)

    # write result
    write_result(gr_solution, '../result/result.txt')
    obj = obj_maxcut(gr_solution, graph)
    print('obj: ', obj)
    
    # plot fig
    plot_fig(gr_scores, alg_name)




