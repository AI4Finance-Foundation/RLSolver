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

With respect to dataset (.txt), the first row includes the number of nodes and edges, and the other rows indicate the two nodes together with the weight of the edge. There are two datasets: __[Gset](https://web.stanford.edu/~yyye/yyye/Gset/)__ and __Syn__, both of which are in the "data" folder. In the Syn dataset, syn_n_m.txt is with n nodes and m edges. If users need more synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE hojh for China users). 

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

The filename of the results follows the principle: the last number denotes the running duration (seconds). Take syn_10_21_1800.txt as an example, it is solution of syn_10_21.txt, and the running duration is 1800 seconds. Take gset_14_1800.txt as an example, it is solution of gset_14.txt, and the running duration is 1800 seconds. 

If using Gurobi or SCIP, the generated files have their own formats (e.g., result.lp and result.sol) for easy check, which are very different from that by running algorithms. 

The partial results are stored in the folder "result" in this repo. All the results are stored in [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE: hojh for China users). 



## Performance

In the following experiments, we use GPU during training by default. 

When use solvers, "gap" is calculated based on the objective of its solution and the best bound. When we use our method, "gap_best" is calculated based on the objective of our solution and the best one over other methods. To distinguish them, we use "gap_best" for our method. gap_best = $\frac{obj - obj*} { obj*}$, where $obj$ is the objective value of our method, and $obj*$ is the best objective value over all comparison methods. Therefore, we see that the solution of solvers may be better than ours, but the "gap" of solvers is larger than "gap_best" of our method, which is caused by different calculations.

1) __Gset__

[Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Stanford university. 

| graph | #nodes| #edges | BLS | DSDP | KHLWG | RUN-CSP | PI-GNN | Gurobi (0.5 h) | Gap | Gurobi (1 h) |Gap | Gurobi (10 h) |Gap | Ours | Gap_best | 
|---|----------|----|---|-----|-----|--------|----------|------| ---| ---| ----|----| ---| ----|----|
|G14 | 800 | 4694 | __3064__| | 2922 | 3061 | 2943  |3034 | 4.15%|3042| 3.61\%|3046|3.22\%| 3029 | 1.14\%|
|G15 | 800 | 4661 | __3050__ | 2938 | __3050__ | 2928 | 2990  | 3016| 4.31%|3033|3.33\%| 3034| 3.07\%| 2995 | 1.80\% | 
|G22 | 2000 | 19990 |__13359__ | 12960 | __13359__ | 13028 | 13181  |13062 |37.90%|13129| 28.94\%|13159| 21.83\%| 13167 |  1.44\% | 
|G49 | 3000 | 6000 | __6000__ | __6000__ | __6000__ | __6000__ | 5918  |__6000__ |0|__6000__ |0| __6000__ |0 | 5790|  3.50\% | 
|G50 | 3000 | 6000 | __5880__ | __5880__ | __5880__ | __5880__ | 5820  |__5880__|0|__5880__|0| __5880__|0 | 5720|  2.72\% | 
|G55 | 5000 | 12468 | __10294__ | 9960 | 10236 | 10116 | 10138  | 10103 | 15.39\%|10103| 11.92\%|10103 | 10.69\%  |10017 |  2.69\% | 
|G70 | 10000 | 9999 |__9541__ | 9456 | 9458 | - | 9421  | 9489 | 2.41\% |9490|2.26\%| 9580| 0.96\% |9358 | 1.92\% | 



2) __Syn__ 

We use the whole synthetic data. For graphs with n nodes, there are 5 datasets, and we run once for each dataset, and finally calcualte the average and standard deviation for the objective values. 

In the following table, the first row illustrates the limited time for solvers. The average running duration is exactly the limited time if we do not write the average duration. We see that, when the number of nodes is not larger than 100, the optimal solutions are obtained, and the average running duraton is much less than 0.5 hour. The inference time of our method is less than 0.001 second.
 

|Datasets |Gurobi (0.5 h)| Gap |Gurobi (1 h) | Gap |Gurobi (1.5 h) |Gap |Ours|Gap_best |
|-------|------|----| ---- |------|----| ---- |---- |--|---- |---- |--|---- |---- |--|
|syn_10   | 17.40 $\pm$ 0.80 (0.004s) | 0| 17.40 $\pm$ 0.80 (0.004s)| 0 | 17.40 $\pm$ 0.80 (0.004s)| 0| $\pm$  | |  
|syn_50   | 134.20 $\pm$ 2.04 (0.30s)  | 0| 134.20 $\pm$ 2.04 (0.30s)| 0 | $\pm$ | | 134.20 $\pm$ 2.04 (0.30s)| 0|  $\pm$   |  |  
|syn_100  |  337.2 $\pm$ 2.71 (289.99s) |0 | 337.2 $\pm$ 2.71 (289.99s)| 0 | 337.2 $\pm$ 2.71 (289.99s)| 0|  $\pm$ |   |  
|syn_300   |  $\pm$ (1800s)  | \%| 1404.00 $\pm$ 7.54 (3600s) | 9.18\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_500   |  $\pm$ (1800s)  | \%| 2475.40 $\pm$ 15.00 (3600s)| 13.86\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_700   |  $\pm$ (1800s)  | \%| 2852.2 $\pm$ 14.30 (3600s) | 13.26\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_900   |  $\pm$ (1800s)  | \% | 3624.00 $\pm$ 9.86 (3600s) | 13.88\%   | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_1000  |  $\pm$ (1800s)  | \%| 4437.8 $\pm$ 16.85 (3600s) |  15.59\%   | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_3000  |  $\pm$ (1800s)  | \% | 17145.00 $\pm$ 33.60 (3600s) | 32.73\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_5000  |  $\pm$ (1800s)  | \% | 30500.80 $\pm$ 223.32 (3600s) | 52.17\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_7000  |  $\pm$ (1800s)  | \% | 47460.00 $\pm$ 473.76 (3600s) |  56.87\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_9000  |  $\pm$ (1800s)  | \% | 57730.20 $\pm$ 502.51 (3600s) | 60.00\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  
|syn_10000 |  $\pm$ (1800s)  | \% | 60768.40 $\pm$ 585.41 (3600s) |  58.67\%  | $\pm$ (5400s)| \%|   $\pm$   |  |  



 
