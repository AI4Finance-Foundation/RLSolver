# Graph Maxcut
## File structure
```python
main.py  #main functions: main, train_opt_net, roll_out
opt_gurobi.py # run Gurobi for reference performance, Gurobi license is required
utils.py # utils file, including opt_net, opt_variable, obj_fun, etc.
```
## Run our method with command 

```
python main.py #N #Sparsity
```

## Run Gurobi with command 

```
python opt_gurobi.py #N #Sparsity
```


## Experiment Results
Sprsity=0.5
|Maxcut |Gurobi (Running Time)|Ours (Running Time)|
|-------|------|----|
|N=20   | 67 (5s)  | 67 (20s)|
|N=30   | 132 (10s)  | 132 (30s)|
|N=100   | 1408 (2000s)  | 1409 (800s)|
|N=1000   |  128508 (4400s)  |  129900 (1000s)|
## Workflow
 ![pipeline](pipeline.jpg)
