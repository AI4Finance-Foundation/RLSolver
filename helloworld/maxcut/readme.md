# Maxcut
## File structure
```python
data # the dataset (.txt), and the figure of the graph (.png)
env # the environment for training
result # store the result, including the solution (.txt), and the figure (.png) drawn by matplotlib
maxcut.py  #main functions: main, train_opt_net
maxcut_gurobi.py # run Gurobi for reference performance, Gurobi should be installed and its license is required
utils.py # utils file, including opt_net, obj, read, write, etc.
```
## Read data

We use the function read_txt_as_networkx_graph(filename) in utils.py to read the data, which returns a networkx graph. 

With respect to gset, the first row includes the number of nodes and edges, and the other rows indicate the two nodes together with the weight of the edge.

For example, gest_14, 

800 4694 # the number of nodes is 800, and the number of edges is 4694

1 7 1 # the edge connects node 1 and 7, and its weight is 1

1 10 1 # the edge connects node 1 and 10, and its weight is 1

...

## Run algorithms with command 

Format:
```
python xxx.py
```


| Algorithms | File| Command | 
|---|----------|----|
|random walk | random_walk.py | python random_walk.py|
| greedy | greedy.py | python greedy.py|
| simulated annealing| simulated_annealing.py | python simulated_annealing.py|


## Run using Gurobi

We can use a state-of-the-art solver Gurobi to solve the graph maxcut problem.

```
python maxcut_gurobi.py 
```


## Experiment Results

In the following experiments, we use GPU during training by default. 

Synthetic data at sparsity = 0.5

Average over 30 runs.
 
|Maxcut |Gurobi, (1 h)| Gurobi, (5 h) | Gurobi, (10 h) | Ours|improvement |
|-------|------|----| ---- |---- |--|
|N=20   | 67 (5s) $\pm$  | || 71, (36s)  | +5.97%, (0.139x) |
|N=30   | 132 (10s) $\pm$  | || 135, (93s)  | +2.27%, (0.108x) |
|N=100   | 1408 $\pm$  | || 1415, (33s)  | +0.49%, (60.6x) |
|N=1000   |  128508 $\pm$  || | 129714, (119s) | +0.94%, (36.97x) |
|N=2000   | 503890   |507628 | |  | | 
|N=3000   |  1125568 | 1129810| |  | |
|N=4000   | | | |  | |
|N=5000 | |  |  | 3175813 $\pm$, (202s)| |

Inference time of our method is less than 0.001 second.


[Gset dataset at Stanford](https://web.stanford.edu/~yyye/yyye/Gset/)

The gap is calculated by $\frac{obj - obj*} { obj*}$, where $obj$ is the objective value of our method, and $obj*$ is the best objective value over all comparison methods.  

| graph | #nodes| #edges | BLS | DSDP | KHLWG | RUN-CSP | PI-GNN | Gurobi (1 h) | Gurobi (5 h) | Gurobi (10 h) | Ours | Gap | 
|---|----------|----|---|-----|-----|--------|----------|------| ---| ---| ----|----|
|G14 | 800 | 4694 | __3064__| | 2922 | 3061 | 2943  |3056 (24h) | ---| ---| 3025 | -1.27\%|
|G15 | 800 | 4661 | __3050__ | 2938 | __3050__ | 2928 | 2990  | ---| ---| | 2965 | -2.78\% | 
|G22 | 2000 | 19990 |__13359__ | 12960 | __13359__ | 13028 | 13181  | |---| ---| 12991 |  -2.75\% | 
|G49 | 3000 | 6000 | __6000__ | __6000__ | __6000__ | __6000__ | 5918  | ---| --- | --- | 5790|  -3.50\% | 
|G50 | 3000 | 6000 | __5880__ | __5880__ | __5880__ | __5880__ | 5820  | ---| --- | --- | 5720|  -2.72\% | 
|G55 | 5000 | 12468 | __10294__ | 9960 | 10236 | 10116 | 10138  | ---| --- | ---  |9890 |  -3.92\% | 
|G70 | 10000 | 9999 |__9541__ | 9456 | 9458 | - | 9421  | ---| --- | --- |9163 | -3.96\% | 



## Workflow
 
