# Maxcut
## File structure
```python
data # the dataset (.txt), and the figure of the graph (.png)
env # the environment for training
result # store the result, including the solution (.txt), and the figure (.png) drawn by matplotlib
utils.py # utils file, including opt_net, obj, read, write, etc.
learn_to_anneal.py  # our algorithm
gurobi.py # run Gurobi for reference performance, Gurobi should be installed and its license is required
scip.py # run scip for reference performance, scip should be installed 
random_walk.py
greedy.py
simulated_annealing.py
```

## Dataset

With respect to datasest (.txt), the first row includes the number of nodes and edges, and the other rows indicate the two nodes together with the weight of the edge. We have two types of datasets, __[Gset](https://web.stanford.edu/~yyye/yyye/Gset/)__ and __synthetic__ (prefix is syn, e.g., syn_n_m.txt with n nodes and m edges)

Take gset_14 as an example,

800 4694 # the number of nodes is 800, and the number of edges is 4694

1 7 1 # the edge connects node 1 and 7, and its weight is 1

1 10 1 # the edge connects node 1 and 10, and its weight is 1

...

## Generate synthetic data

If the above dataset is not satisfied, you can generate a graph with n nodes and m edges, i.e., using the function __generate_write_symmetric_adjacency_matrix_and_networkx_graph(n, m)__ in utils.py. It returns an adjacency_matrix and a [networkx](https://networkx.org/documentation/stable/reference/introduction.html) graph, and the graph will be written to a file "syn_n_m.txt" of the folder "data". 

## Read data

We use the function __read_txt_as_networkx_graph(filename)__ in utils.py to read the data, which returns a [networkx](https://networkx.org/documentation/stable/reference/introduction.html) graph. We can access the nodes and edges by graph.nodes and graph.edges, respectively. 

## Run algorithms

Format:
```
python alg_xxx.py  # alg_xxx.py is the file name of the algorithm
```


| Algorithm | File| Command | 
|---|----------|----|
|learn_to_anneal (ours) | learn_to_anneal_1.py <br/> learn_to_anneal_2.py | python learn_to_anneal_1.py, <br/> python learn_to_anneal_2.py|
|random walk | random_walk.py | python random_walk.py|
| greedy | greedy.py | python greedy.py|
| simulated annealing| simulated_annealing.py | python simulated_annealing.py|


## Run using solvers

We can use a state-of-the-art solver [Gurobi](https://www.gurobi.com/) or a well-known open-source solver [Scip](https://scipopt.org/) to solve the graph maxcut problem. Gurobi should be installed and its license is required. Scip should also be installed if you choose it.

```
python gurobi.py

python scip.py 
```

## Store results

Results will be written to a file result.txt in the folder 'result'. The first column is the node, and the second column is the label of classified set. For example, 

1 2  # node 1 in set 2

2 1  # node 2 in set 1

3 2  # node 3 in set 2

4 1  # node 4 in set 1

5 2  # node 5 in set 2

If using Gurobi or Scip, more files will be generated (e.g., result.lp and result.mps) for easy check. 

## Experiment Results

In the following experiments, we use GPU during training by default. 

Synthetic data at density = 0.5

Average over 30 runs.
 
|Maxcut |Gurobi, (1 h)| Gurobi, (5 h) | Gurobi, (10 h) | Ours|improvement |
|-------|------|----| ---- |---- |--|
|syn_20_50   | 67 (5s) $\pm$  | || 71, (36s)  | +5.97%, (0.139x) |
|syn_30_110   | 132 (10s) $\pm$  | || 135, (93s)  | +2.27%, (0.108x) |
|syn_50_190   |   | ||   |  |
|syn_100_460  | 1408 $\pm$  | || 1415, (33s)  | +0.49%, (60.6x) |
|syn_200_1004   |   | ||   |  |
|syn_400_2109   |   | ||   |  |
|syn_800_4078   |   | ||   |  |
|syn_1000_6368   |  128508 $\pm$  || | 129714, (119s) | +0.94%, (36.97x) |
|syn_2000_13386   | 503890   |507628 | |  | | 
|syn_3000_25695   |  1125568 | 1129810| |  | |
|syn_4000_38654   | | | |  | |
|syn_5000_50543 | |  |  | 3175813 $\pm$, (202s)| |
|syn_6000_73251   |   | ||   |  |
|syn_7000_79325   |   | ||   |  |
|syn_8000_83647   |   | ||   |  |
|syn_9000_96324   |   | ||   |  |
|   |   | ||   |  |

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
 
