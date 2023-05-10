# Hamiltonian Cycle
## File structure
```python
hamiltonian.py  #main functions: main, train_opt_net
hamiltonian_gurobi.py # run Gurobi for reference performance, Gurobi should be installed and its license is required
```
## Run our method with command 

Format:
```
python hamiltonian.py #gpu_id (-1: cpu, >=0: gpu) #choice (0: Synthetic data, 1: Gset) ...
```

If using synthetic data:
```
python hamiltonian.py #gpu_id (-1: cpu, >=0: gpu) 0 #N #Sparsity 
```

If using TSPLIB:
```
python hamiltonian.py #gpu_id (-1: cpu, >=0: gpu) 1 #DatasetId
```
## Download data using the following link:

```
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/hcp/
```


## Run Gurobi with command 

```
python hamiltonian_gurobi.py #N #Sparsity #gpu_id (-1: cpu, >=0: gpu) #choice (0: Synthetic data, 1: Gset)
```


## Experiment Results

In the following experiments, we use GPU during training by default. 




## Workflow
 
