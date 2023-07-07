# comparison methods for maxcut: random walk, greedy, epsilon greedy, simulated annealing
import copy

import torch as th
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch import Tensor
from typing import List
import random
from envs.maxcut_env import MaxcutEnv
from MaxCut_H2O import GraphMaxCutEnv
from utils import Opt_net
import pickle as pkl
from utils import calc_file_name
graph_node = {"14":800, "15":800, "22":2000, "49":3000, "50":3000, "55":5000, "70":10000  }


def random_walk(init_solution: Tensor, num_steps: int, env: GraphMaxCutEnv) -> (int, Tensor):
    curr_solution: Tensor = copy.deepcopy(init_solution)
    length = len(curr_solution)
    for i in range(num_steps):
        index = random.randint(0, length - 1)
        curr_solution[index] = (curr_solution[index] + 1) % 2
    score = int(-env.get_objective(curr_solution)[0])
    return score, curr_solution



def greedy(start_node: int, env: GraphMaxCutEnv) -> (int, Tensor):
    nodes = list(range(env.num_nodes))
    visited = [start_node]
    curr_solution: Tensor = th.zeros((1, env.num_nodes))
    curr_score: int = int(-env.get_objective(curr_solution)[0])
    for iteration in range(env.num_nodes - 1):
        print("iteration in greedy: ", iteration)
        tmp_solutions = []
        tmp_scores = []
        tmp_nodes = []
        nodes_to_visit = set()
        for k in range(len(visited)):
            will_visit = env.get_neighbor_nodes(k)
            to_visit = [item for item in will_visit if item not in visited]
            nodes_to_visit = nodes_to_visit | set(to_visit)
        nodes_to_visit = list(nodes_to_visit)
        for j in range(len(nodes_to_visit)):
            node = int(nodes_to_visit[j])
            tmp_solution = copy.deepcopy(curr_solution)
            tmp_solution[0, node] = (tmp_solution[0, node] + 1) % 2
            tmp_score = int(-env.get_objective(tmp_solution)[0])
            tmp_scores.append(tmp_score)
            tmp_solutions.append(tmp_solution)
            tmp_nodes.append(node)
        if len(tmp_scores) >= 1:
            best_tmp_score = max(tmp_scores)
            index = tmp_scores.index(best_tmp_score)
            best_tmp_solution = tmp_solutions[index]
            best_tmp_node = tmp_nodes[index]
            if best_tmp_score > curr_score:
                curr_score = best_tmp_score
                curr_solution = best_tmp_solution
                visited.append(best_tmp_node)
        else:
            return curr_score, curr_solution

    return curr_score, curr_solution


if __name__ == '__main__':
    env = GraphMaxCutEnv(graph_key='gset_70')
    init_solution = th.randint(0, 1+1, (1, env.num_nodes))
    # rw_score, rw_solution = random_walk(init_solution=init_solution, num_steps=10000, env=env)

    start_node = random.randint(0, env.num_nodes - 1)
    gr_score, gr_solution = greedy(start_node=start_node, env=env)



