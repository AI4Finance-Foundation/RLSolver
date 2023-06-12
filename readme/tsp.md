# Hamiltonian Cycle
## File structure
```python
tsp.py  #main functions: main, train_opt_net
tsp_gurobi.py # run Gurobi for reference performance, Gurobi should be installed and its license is required
```
## Run our method with command 

Format:
```
python tsp.py #gpu_id (-1: cpu, >=0: gpu) #choice (0: Synthetic data, 1: Gset) ...
```

If using synthetic data:
```
python tsp.py #gpu_id (-1: cpu, >=0: gpu) 0 #N #Sparsity 
```

If using TSPLIB:
```
python tsp.py #gpu_id (-1: cpu, >=0: gpu) 1 #DatasetId
```
## Download data using the following link:

```
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/
```


## Run Gurobi with command 

```
python tsp_gurobi.py #N #Sparsity #gpu_id (-1: cpu, >=0: gpu) #choice (0: Synthetic data, 1: Gset)
```


## Experiment Results

In the following experiments, we use GPU during training by default. 




## Workflow
 
