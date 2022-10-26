# Traveling Salesman Problem (TSP)
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
