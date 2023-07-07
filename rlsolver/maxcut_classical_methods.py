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
    init_score = int(-env.get_objective(curr_solution)[0])
    length = len(curr_solution[0])
    scores = []
    for i in range(num_steps):
        index = random.randint(0, length - 1)
        curr_solution[0, index] = (curr_solution[0, index] + 1) % 2
        score = int(-env.get_objective(curr_solution)[0])
        scores.append(score)

    print("score, init_score of random_walk", score, init_score)
    print("scores: ", scores)
    return score, curr_solution



def greedy(start_node: int, init_solution: Tensor, num_steps: int, env: GraphMaxCutEnv) -> (int, Tensor):
    nodes = list(range(env.num_nodes))
    visited = [start_node]
    # curr_solution: Tensor = th.zeros((1, env.num_nodes))
    curr_solution: Tensor = init_solution
    curr_score: int = int(-env.get_objective(curr_solution)[0])
    init_score = curr_score
    scores = []
    for iteration in range(env.num_nodes - 1):
        if iteration >= num_steps:
            break
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
            scores.append(best_tmp_score)
            index = tmp_scores.index(best_tmp_score)
            best_tmp_solution = tmp_solutions[index]
            best_tmp_node = tmp_nodes[index]
            if best_tmp_score > curr_score:
                curr_score = best_tmp_score
                curr_solution = best_tmp_solution
                visited.append(best_tmp_node)
        else:
            return curr_score, curr_solution
    print("score, init_score of greedy", curr_score, init_score)
    print("scores: ", scores)
    return curr_score, curr_solution


def simulated_annealing(init_solution: Tensor, init_temperature: int, num_steps: int, env: GraphMaxCutEnv) -> (int, Tensor):
    curr_solution: Tensor = copy.deepcopy(init_solution)
    curr_score = int(-env.get_objective(curr_solution)[0])
    init_score = curr_score
    length = len(curr_solution[0])
    scores = []
    for k in range(num_steps):
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        index = random.randint(0, length - 1)
        new_solution = copy.deepcopy(curr_solution)
        new_solution[0, index] = (new_solution[0, index] + 1) % 2
        new_score = int(-env.get_objective(new_solution)[0])
        scores.append(new_score)
        delta_e = curr_score - new_score
        if delta_e < 0:
            curr_solution = new_solution
            curr_score = new_score
        else:
            if temperature == 0:
                print()
            prob = np.exp(- delta_e / (temperature + 1e-6))
            if prob > random.random():
                curr_solution = new_solution
                curr_score = new_score
    print("score, init_score of simulated_annealing", curr_score, init_score)
    print("scores: ", scores)
    return curr_score, curr_solution

if __name__ == '__main__':
    env = GraphMaxCutEnv(graph_key='gset_14')

    init_solution = th.randint(0, 1+1, (1, env.num_nodes))

    rw_score, rw_solution = random_walk(init_solution=init_solution, num_steps=1000, env=env)

    start_node = random.randint(0, env.num_nodes - 1)
    gr_score, gr_solution = greedy(start_node=start_node, init_solution=init_solution, num_steps=10, env=env)

    init_temperature = 2000
    num_steps = 1000
    sa_score, sa_solution = simulated_annealing(init_solution=init_solution, init_temperature=init_temperature, num_steps=num_steps, env=env)



