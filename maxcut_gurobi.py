import numpy as np


def generate_adjacency_matrix(n: int):
    matrix = np.random.randint(0, 2, (n, n))
    pass



if __name__ == '__main__':
    generate_adjacency_matrix(4)