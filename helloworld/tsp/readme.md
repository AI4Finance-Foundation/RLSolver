# TSP
## Generate TSP instances
* Run the following command to generate random TSP instances:
```
python generate_data.py --problem tsp --name test
```
## Get optimal tours with Gurobi
* Requirements: Please install Gurobi and gurobipy
* Run the following command to obtain optimal tours on the TSP instances *tsp20_test_seed1234.pkl*:
```
python tsp_baseline.py gurobi data/tsp/tsp20_test_seed1234.pkl -f
```