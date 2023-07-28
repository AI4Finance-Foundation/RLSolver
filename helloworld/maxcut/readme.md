# Maxcut
## File structure
```python
data # the dataset (.txt), and the figure of the graph (.png)
mcmc_sim # the Markov chain Monte Carlo simulation.
result # store the result, including the solution (.txt), and the figure (.png) drawn by matplotlib.
utils.py # utils file, including opt_net, obj, read, write, etc.
learn_to_anneal_x.py  # our algorithm. x denotes the version. 
gurobi.py # solve max by utilizing Gurobi, which should be installed and the license is required.
scip.py # solve max by utilizing SCIP, which should be installed .
random_walk.py # the random walk method
greedy.py # the greedy method
simulated_annealing.py # the simulated annealing method
```

# Work flow
<a target="\_blank">
	<div align="center">
		<img src=fig/work_flow.png width="90%"/>
	</div>
</a>


## Dataset

With respect to datasest (.txt), the first row includes the number of nodes and edges, and the other rows indicate the two nodes together with the weight of the edge. There are two datasets: __[Gset](https://web.stanford.edu/~yyye/yyye/Gset/)__ and __Syn__, both of which are in the "data" folder. In the Syn dataset, syn_n_m.txt is with n nodes and m edges. If users need more synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL) or [Baidu Wangpan](https://pan.baidu.com/s/1QUAAd5rs93fpc2Ixgtm8lw) (CODE 2fw9 for Chinese users). 

Take gset_14.txt as an example,

800 4694 # the number of nodes is 800, and the number of edges is 4694

1 7 1 # the edge connects node 1 and 7, and its weight is 1

1 10 1 # the edge connects node 1 and 10, and its weight is 1

...

## Generate synthetic data

If users want to generate a graph with n nodes and m edges, please use the function __generate_write__ in utils.py. It returns an adjacency matrix and a [networkx](https://networkx.org/documentation/stable/reference/introduction.html) graph, and the graph will be written to a file "syn_n_m.txt" in the folder "data". 

## Read data

We use the function __read_txt__ in utils.py to read the data, which returns a [networkx](https://networkx.org/documentation/stable/reference/introduction.html) graph. We can access the nodes and edges by graph.nodes and graph.edges, respectively. 

## Run algorithms

Format:
```
python xxx.py  # xxx.py is the file name of the algorithm
```

| Algorithm | File| 
|---|----------|
|learn_to_anneal (ours) | learn_to_anneal_1.py <br/> learn_to_anneal_2.py | 
|random walk | random_walk.py | 
| greedy | greedy.py | 
| simulated annealing| simulated_annealing.py | 


## Run using solvers

We can use a state-of-the-art solver [Gurobi](https://www.gurobi.com/) or a well-known open-source solver [SCIP](https://scipopt.org/) to solve the graph maxcut problem. Gurobi should be installed and its license is required. SCIP should also be installed if you choose it. We recommend to use Gurobi since its performance is best except that you do not have a license. 

```
python gurobi.py

python scip.py 
```

## Store results

Results will be written to a file result.txt in the folder "result". The first column is the node, and the second column is the label of classified set. For example, 

1 2  # node 1 in set 2

2 1  # node 2 in set 1

3 2  # node 3 in set 2

4 1  # node 4 in set 1

5 2  # node 5 in set 2

If using Gurobi or SCIP, the generated files have their own formats (e.g., result.lp and result.mps) for easy check, which are very different from that by running algorithms. 

The partial results are stored in the folder "result" in this repo. The whole results are stored in [Google Drive](https://drive.google.com/drive/folders/1eO6QHnN8FdyD2eJIsVTQQ3Ch863whFWp) or [Baidu Wangpan](https://pan.baidu.com/s/1aXbHCB-Kxop4gGHeu2Am0A) (CODE: 3mw0 for China users). 

## Experiment Results

In the following experiments, we use GPU during training by default. 


1) __Gset__

[Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Stanford university. In the following table, the gap is calculated by $\frac{obj - obj*} { obj*}$, where $obj$ is the objective value of our method, and $obj*$ is the best objective value over all comparison methods.  

| graph | #nodes| #edges | BLS | DSDP | KHLWG | RUN-CSP | PI-GNN | Gurobi (1 h) | Gurobi (5 h) | Gurobi (10 h) | Ours | Gap | 
|---|----------|----|---|-----|-----|--------|----------|------| ---| ---| ----|----|
|G14 | 800 | 4694 | __3064__| | 2922 | 3061 | 2943  |3056 (24h) | ---| ---| 3029 | -1.14\%|
|G15 | 800 | 4661 | __3050__ | 2938 | __3050__ | 2928 | 2990  | ---| ---| | 2995 | -1.80\% | 
|G22 | 2000 | 19990 |__13359__ | 12960 | __13359__ | 13028 | 13181  | |---| ---| 13167 |  -1.44\% | 
|G49 | 3000 | 6000 | __6000__ | __6000__ | __6000__ | __6000__ | 5918  | ---| --- | --- | 5790|  -3.50\% | 
|G50 | 3000 | 6000 | __5880__ | __5880__ | __5880__ | __5880__ | 5820  | ---| --- | --- | 5720|  -2.72\% | 
|G55 | 5000 | 12468 | __10294__ | 9960 | 10236 | 10116 | 10138  | ---| --- | ---  |10017 |  -2.69\% | 
|G70 | 10000 | 9999 |__9541__ | 9456 | 9458 | - | 9421  | ---| --- | --- |9358 | -1.92\% | 


2) __Syn__ 

We use the whole synthetic data. For graphs with n nodes, there are 20 datasets, and we run once for each dataset, and finally calcualte the average and standard deviation for the objective values. 

In the following table, the first row illustrates the limited time for solvers. The average running duration is exactly the limited time if we do not write the average duration. We see that, when the number of nodes is not larger than 100, the optimal solutions are obtained, and the average running duraton is much less than 0.5 hour. The improvement is calculated by $\frac{obj - obj'} { obj'}$, where $obj$ is the average objective value of our method, and $obj'$ is the average objective value of solvers. The inference time of our method is less than 0.001 second.
 

|Datasets |Gurobi (0.5 h)| Gap |Gurobi (1 h) | Gap |Gurobi (10 h) |Gap | SCIP (0.5 h)| Gap |SCIP (1 h) |Gap | SCIP (10 h) |Gap |Ours|Improvement |
|-------|------|----| ---- |------|----| ---- |---- |--|---- |---- |--|---- |---- |--|
|syn_10   | 20.80 $\pm$ 2.71 (0.01s) | 0| $\pm$ |  | $\pm$ | | 20.80 $\pm$ 2.71 (0.16s)  | 0| $\pm$  |  | $\pm$ | |$\pm$ | |  
|syn_50   | 138.75 $\pm$ 4.28 (0.14s)  | 0| $\pm$ |  | $\pm$ | | 138.75 $\pm$ 4.28 (10.34s)  | 0| $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_100  |  342.20 $\pm$ 4.33 (131.07s)|0 | $\pm$ |  | $\pm$ | | 342.20 $\pm$ 4.33 (597.42s) |0 | $\pm$  || $\pm$ || $\pm$ ||  $\pm$    |  
|syn_300   |  1407.05 $\pm$ 8.55 | 9.30\%| $\pm$ |  | $\pm$ | |   1344.35 $\pm$ 17.82  | 38.95\%| $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_500   | 2480.95 $\pm$ 15.22| 13.80\%| $\pm$ |  | $\pm$ | |   2327.55 $\pm$ 16.84   | 46.30\%| $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_700   |  2857.65 $\pm$ 11.91| 13.27\%| $\pm$ |  | $\pm$ | |  2667.55 $\pm$ 23.31   | 44.01\%| $\pm$  || $\pm$ || $\pm$ ||  $\pm$   | |  
|syn_900   |  3631.60 $\pm$ 12.72| 14.15\% | $\pm$ | \% | $\pm$ | \%|  3393.7 $\pm$ 32.30   | 44.48\%| $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_1000  |  4436.55 $\pm$ 16.10 | 15.92\%| $\pm$ |  | $\pm$ | |  4152.55 $\pm$ 32.43 | 47.20\% | $\pm$  | | $\pm$ || $\pm$ ||  $\pm$   | |  
|syn_3000  | 17108.45 $\pm$ 24.31 | 37.13\% | $\pm$ |  | $\pm$ | |   16192.9 $\pm$ 40.91   | 55.44\% | $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_5000  | 30334.45 $\pm$ 355.00 | 55.11\% | $\pm$ |  | $\pm$ | |   -    | -| $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_7000  | 47325.15 $\pm$ 555.25 | 59.90\% | $\pm$ |  | $\pm$ | |   -    |- | $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_9000  | 57692.75 $\pm$ 594.53 | 60.68\% | $\pm$ |  | $\pm$ | |   -    |- | $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  |  
|syn_10000 | 60689.75 $\pm$ 483.62 | 60.22\% | $\pm$ |  | $\pm$ | |   -    | -| $\pm$  || $\pm$ || $\pm$ ||  $\pm$   |  | 



 
