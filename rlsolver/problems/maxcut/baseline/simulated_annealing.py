import copy
import time
from typing import List, Union
import numpy as np
from typing import List
import random
import networkx as nx
from utils import read_txt_as_networkx_graph
from utils import obj_maxcut
from utils import write_result
from utils import plot_fig

def simulated_annealing(init_solution: Union[List[int], np.array], init_temperature: int, num_steps: int, graph: nx.Graph) -> (int, Union[List[int], np.array], List[int]):
    print('simulated_annealing')
    start_time = time.time()
    curr_solution = copy.deepcopy(init_solution)
    curr_score = obj_maxcut(curr_solution, graph)
    init_score = curr_score
    num_nodes = len(init_solution)
    scores = []
    for k in range(num_steps):
        # The temperature decreases
        temperature = init_temperature * (1 - (k + 1) / num_steps)
        index = random.randint(0, num_nodes - 1)
        new_solution = copy.deepcopy(curr_solution)
        new_solution[index] = (new_solution[index] + 1) % 2
        # calc the obj
        new_score = obj_maxcut(new_solution, graph)
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
    # read data
    graph = read_txt_as_networkx_graph('../data/syn/syn_50_176.txt')

    # run alg
    init_solution = list(np.random.randint(0, 2, graph.number_of_nodes()))
    init_temperature = 4
    num_steps = 8000
    sa_score, sa_solution, sa_scores = simulated_annealing(init_solution, init_temperature, num_steps, graph)

    # write result
    write_result(sa_solution, '../result/result.txt')

    # plot fig
    alg_name = 'SA'
    plot_fig(sa_scores, alg_name)



