# Traveling Salesman Problem (TSP)
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