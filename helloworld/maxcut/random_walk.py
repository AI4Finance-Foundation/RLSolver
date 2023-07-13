# comparison methods for maxcut: random walk, greedy, epsilon greedy, simulated annealing
import copy
import time
import networkx as nx
import numpy as np
from typing import List, Union
import random
import matplotlib.pyplot as plt
from utils import read_as_networkx_graph
from utils import obj_maxcut
from utils import write_result
from utils import generate_write_symmetric_adjacency_matrix_and_networkx_graph

def plot_fig(scores: List[int], label: str):
    x = list(range(len(scores)))
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    plt.plot(x, scores, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig('result/' + label + '.png')
    plt.show()


def random_walk(init_solution: Union[List[int], np.array], num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('random_walk')
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    init_score = obj_maxcut(init_solution, graph)
    num_nodes = len(curr_solution)
    scores = []
    for i in range(num_steps):
        # select a node randomly
        node = random.randint(0, num_nodes - 1)
        curr_solution[node] = (curr_solution[node] + 1) % 2
        # calc the obj
        score = obj_maxcut(curr_solution, graph)
        scores.append(score)
    print("score, init_score of random_walk", score, init_score)
    print("scores: ", scores)
    print("solution: ", curr_solution)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return score, curr_solution, scores


if __name__ == '__main__':
    # graph1 = read_as_networkx_graph('data/gset_14.txt')
    graph = read_as_networkx_graph('data/syn_5_5.txt')
    init_solution = [1, 0, 1, 0, 1]
    adj_matrix, graph = generate_write_symmetric_adjacency_matrix_and_networkx_graph(11, 0.9)
    rw_score, rw_solution, rw_scores = random_walk(init_solution=init_solution, num_steps=1000, graph=graph)
    write_result(rw_solution)
    obj = obj_maxcut(rw_solution, graph)
    alg_name = 'RW'
    plot_fig(rw_scores, alg_name)


