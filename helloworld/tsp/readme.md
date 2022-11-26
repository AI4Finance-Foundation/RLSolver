# Traveling Salesman Problem (TSP)
## Formulation

## Massive parallel environment details:
+ Parameters initialization: number of cities $N$, number of environments $M$, device to run the envionment $device$.
+ Vector environment initialization: wrapping the `generate_distance_matrix()` function, `get_reward()` function, `generate_tour_rand` function into vector versions with `functorch.vmap()`.
```python
class TSPEnv():
    def __init__(self, N=4, episode_length=6, num_env=4096, device=torch.device("cuda:0")):
        self.N = N # number of cities
        self.basis_vectors, _ = torch.linalg.qr(torch.rand(self.N * 2, self.N * 2, dtype=torch.float))
        self.subspace_dim = 1
        self.num_env = num_env
        self.device = device
        self.episode_length = episode_length
        self.parallel_tensor_shape = self.zeros(self.num_env, 1)
        self.diag = 1e6 * torch.eye(self.N)
        # Use vmap to wrap the following functions
        self.get_reward_vec = vmap(self.get_reward, in_dims = (0, 0), out_dims = (0))
        self.generate_distance_matrix_cl_vec = vmap(self.generate_distance_matrix_cl, in_dims = 0, out_dims = 0)
        self.generate_distance_matrix_rand_vec = vmap(self.generate_distance_matrix_rand, in_dims = 0, out_dims = 0)
        self.generate_tour_rand_vec = vmap(self.generate_tour_rand, in_dims=0, out_dims=0)
```
## Baseline: 
  - SOTA algorithms: Christofides Algorithm
  - Commercial solver: Gurobi 10.0
## Performance Metrics:
  - Approximation Ratio" [1, $+\infty$ )
  - Training time
  - Average time cost
## Generate problem instances
* Run the following command to generate problem instances:
```
python generate_data.py --problem tsp --name test
```
## Obtain tours by Gurobi
* Requirements: Please install Gurobi and gurobipy
* Run the following command to obtain tours on the problem instances *tsp20_test_seed1234.pkl*:
```
python tsp_baseline.py gurobi data/tsp/tsp20_test_seed1234.pkl -f
```
+ Implementation details:
```python
def solve_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinates 
    :return: m.objVal, tour: tour length of solution, tour solution
    """
```