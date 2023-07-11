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
from env import GraphMaxCutEnv
from utils import Opt_net
import pickle as pkl
from utils import calc_file_name
import matplotlib.pyplot as plt

# graph_node = {"14":800, "15":800, "22":2000, "49":3000, "50":3000, "55":5000, "70":10000  }

def plot_fig(scores: List[int], num_steps: int, label: str):
    # fig = plt.figure()
    x = list(range(num_steps))
    dic = {'0': 'ro-', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    plt.plot(x, scores, dic['0'])
    plt.legend([label], loc=0)
    plt.savefig(label + '.png')
    plt.show()

def plot_figs(scoress: List[List[int]], num_steps: int, labels: List[str]):
    num = len(scoress)
    x = list(range(num_steps))
    dic = {'0': 'ro', '1': 'gs', '2': 'b^', '3': 'c>', '4': 'm<', '5': 'yp'}
    for i in range(num):
        plt(x, scoress[i], dic[str(i)], labels[i])
    plt.legend(labels, loc=0)
    plt.show()
def random_walk(init_solution: Tensor, num_steps: int, env: GraphMaxCutEnv) -> (int, Tensor):
    print('random_walk')
    start_time = time.time()
    curr_solution: Tensor = copy.deepcopy(init_solution)
    init_score = int(-env.get_objective(curr_solution)[0])
    length = len(curr_solution[0])
    scores = []
    for i in range(num_steps):
        index = random.randint(0, length - 1)
        # Here, 0 denotes the 0-th env, since the dim is 1 * env.num_nodes, where 1 dentoes the num of envs.
        curr_solution[0, index] = (curr_solution[0, index] + 1) % 2
        # calc the obj
        score = int(-env.get_objective(curr_solution)[0])
        scores.append(score)

    print("score, init_score of random_walk", score, init_score)
    print("scores: ", scores)
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return score, curr_solution, scores



def greedy(init_solution: Tensor, num_steps: int, env: GraphMaxCutEnv) -> (int, Tensor):
    print('greedy')
    start_time = time.time()
    nodes = list(range(env.num_nodes))
    # curr_solution: Tensor = th.zeros((1, env.num_nodes))
    curr_solution: Tensor = init_solution
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


def simulated_annealing(init_solution: Tensor, init_temperature: int, num_steps: int, env: GraphMaxCutEnv) -> (int, Tensor):
    print('simulated_annealing')
    start_time = time.time()
    curr_solution: Tensor = copy.deepcopy(init_solution)
    curr_score = int(-env.get_objective(curr_solution)[0])
    init_score = curr_score
    length = len(curr_solution[0])
    scores = []
    for k in range(num_steps):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        index = random.randint(0, length - 1)
        new_solution = copy.deepcopy(curr_solution)
        # Here, 0 denotes the 0-th env, since the dim is 1 * env.num_nodes, where 1 dentoes the num of envs.
        new_solution[0, index] = (new_solution[0, index] + 1) % 2
        # calc the obj
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
    running_duration = time.time() - start_time
    print('running_duration: ', running_duration)
    return curr_score, curr_solution, scores

if __name__ == '__main__':
    # RW: random walk
    # GR: greedy
    # SA: simulated_annealing
    # alg_names = ['RW', 'GR']
    alg_names = ['RW', 'GR', 'SA']


    # plot_fig(scores=[1, 2,3], num_steps=3, label='sa')

    env = GraphMaxCutEnv(graph_key='gset_14')


    # Initialize the solution, 0 or 1 for each node, with the size env.num_nodes
    # The dim is 1 * env.num_nodes, where 1 dentoes the num of envs. In maxcut_env, we use this format for Massively Parallel Environments
    init_solution = th.randint(0, 1+1, (1, env.num_nodes))

    # 1) random walk
    alg_name = 'RW'
    if alg_name in alg_names:
        rw_score, rw_solution, rw_scores = random_walk(init_solution=init_solution, num_steps=1000, env=env)

    # 2) greedy
    alg_name = 'GR'
    if alg_name in alg_names:
        gr_score, gr_solution, gr_scores = greedy(init_solution=init_solution, num_steps=10, env=env)

    # 3) simulated annealing
    alg_name = 'SA'
    if alg_name in alg_names:
        init_temperature = 40
        num_steps = 320000
        sa_score, sa_solution, sa_scores = simulated_annealing(init_solution=init_solution, init_temperature=init_temperature, num_steps=num_steps, env=env)
        plot_fig(sa_scores, num_steps, alg_name)

    # scoress = []
    # # scoress.append(rw_scores)
    # # scoress.append(gr_scores)
    # scoress.append(sa_scores)
    # # labels = ['rw', 'gr', 'sa']
    # labels = ['RW', 'GR', 'SA']
    # plot(scoress, num_steps: int, labels: List[str])


