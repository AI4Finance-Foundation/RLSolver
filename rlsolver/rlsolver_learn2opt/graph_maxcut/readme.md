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
|Maxcut |Gurobi|Ours|
|-------|------|----|
|N=20   | 67   | 67 |
|N=30   | 132   | 132 |
|N=100   | 1408   | 1409 |
|N=1000   |  128508  |  129900|
## Workflow
 ![pipeline](pipeline.jpg)
