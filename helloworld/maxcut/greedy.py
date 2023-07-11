# comparison methods for maxcut: random walk, greedy, epsilon greedy, simulated annealing
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
from env.maxcut_env2 import MaxCutEnv2
from utils import Opt_net
import pickle as pkl
from utils import calc_file_name
import matplotlib.pyplot as plt

# graph_node = {"14":800, "15":800, "22":2000, "49":3000, "50":3000, "55":5000, "70":10000  }

def plot_fig(scores: List[int], label: str):
    # fig = plt.figure()
    x = list(range(len(scores)))
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    plt.plot(x, scores, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig(label + '.png')
    plt.show()


def greedy(init_solution: Tensor, num_steps: int, env: MaxCutEnv2) -> (int, Tensor):
    print('greedy')
    start_time = time.time()
    nodes = list(range(env.num_nodes))
    curr_solution: Tensor = copy.deepcopy(init_solution)
    curr_score: int = int(-env.get_objective(curr_solution)[0])
    init_score = curr_score
    for iteration in range(env.num_nodes):
        if iteration >= num_steps:
            break
        print("iteration in greedy: ", iteration)
        scores = []
        solutions = []
        for node in nodes:
            new_solution = copy.deepcopy(curr_solution)
            # Here, 0 denotes the 0-th env, since the dim is 1 * env.num_nodes, where 1 dentoes the num of envs.
            new_solution[0, node] = (new_solution[0, node] + 1) % 2
            # calc the obj
            new_score = int(-env.get_objective(new_solution)[0])
            scores.append(new_score)
            solutions.append(new_solution)
        best_score = max(scores)
        index = scores.index(best_score)
        best_solution = solutions[index]
        if best_score >= curr_score:
            curr_score = best_score
            curr_solution = best_solution
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", scores)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores


if __name__ == '__main__':
    # RW: random walk
    # GR: greedy
    # SA: simulated_annealing
    # alg_names = ['RW', 'GR']
    # alg_names = ['RW', 'GR', 'SA']


    env = MaxCutEnv2(graph_key='gset_14')

    # Initialize the solution, 0 or 1 for each node, with the size env.num_nodes
    # The dim is 1 * env.num_nodes, where 1 dentoes the num of envs. In maxcut_env, we use this format for Massively Parallel Environments
    init_solution = th.randint(0, 1 + 1, (1, env.num_nodes))

    alg_name = 'GR'
    gr_score, gr_solution, gr_scores = greedy(init_solution=init_solution, num_steps=10, env=env)
    plot_fig(gr_scores, alg_name)




