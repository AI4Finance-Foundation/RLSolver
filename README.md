# RLSolver: High-performance GPU-based Solvers for Nonconvex and NP-complete Problems

We aim to showcase that reinforcement learning (RL) or machine learning (ML) with GPUs delivers the best benchmark performance for large-scale nonconvex and NP-complete problems. When the size of these problems becomes large, it is very hard to obtain optimal or near optimal solutions. RL with the help of GPU computing can obtain high-quality solutions within short time. RLSolver collects many RL/ML tricks and also operations research (OR) tricks to improve the performance. We encourage users to try RL/ML tricks, and implement their own new ideas based on the datasets. If encountering any problems, please submit github issues, and we can talk there. 

# Key Technologies

The following two key technologies are under active development: 

- **Massively parallel sampling** of Markov chain Monte Carlo simulations on GPU, using thousands of CUDA cores and tensor cores.
- **Podracer scheduling** on a GPU cloud, e.g., DGX-2 SuperPod.
- **OR tricks** such as local search, and tabu search.
- **RL/ML tricks** such as learn to optimize, and curriculum learning.



# Key References

- Mazyavkina, Nina, et al. "Reinforcement learning for combinatorial optimization: A survey." Computers & Operations Research 134 (2021): 105400.

- Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine learning for combinatorial optimization: a methodological tour d’horizon." European Journal of Operational Research 290.2 (2021): 405-421.

- Peng, Yun, Byron Choi, and Jianliang Xu. "Graph learning for combinatorial optimization: a survey of state-of-the-art." Data Science and Engineering 6, no. 2 (2021): 119-141.

- Nair, Vinod, et al. "Solving mixed integer programs using neural networks." arXiv preprint arXiv:2012.13349 (2020).

- Makoviychuk, Viktor, et al. "Isaac Gym: High performance GPU based physics simulation for robot learning." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021.

# Workflow
<a target="\_blank">
	<div align="center">
		<img src=fig/work_flow.png width="90%"/>
	</div>
</a>  



## Datasets
* Maxcut:
  
  1) [Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is opened by Standford university, and is stored in the "data" folder of this repo. The number of nodes is from 800 to 10000. 
  
  2) __Syn__ is the synthetic data obtained by calling the function generate_write in utils.py. The number of nodes is from 10 to 10000. The (partial) synthetic data is stored in the "data" folder of this repo. If users need all the synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE hojh for China users). The results of experiments are obtained based on all the synthetic data.
  
* TSP: [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
  

## Benchmarks


* Learning to branch
  
[code](https://github.com/cwfparsonson/retro_branching/tree/master) 2023 AAAI Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories 

[code](https://github.com/ds4dm/branch-search-trees) 2021 AAAI Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies

* Learning to cut

[code](https://github.com/Wenbo11/learntocut) 2020 ICML Reinforcement learning for integer programming: Learning to cut


* RL/ML-driven heuristic
  
[code](https://github.com/Hanjun-Dai/graph_comb_opt)  (greedy) 2017 NeurIPS Learning Combinatorial Optimization Algorithms over Graphs

[code](https://github.com/optsuite/MCPG) (local search) 2023, A Monte Carlo Policy Gradient Method with Local Search for Binary Optimization

[code](https://github.com/JHL-HUST/VSR-LKH) (LKH for TSP) 2021 AAAI Combining reinforcement learning with Lin-Kernighan-Helsgaun algorithm for the traveling salesman problem 

* Classical methods
  1) Random walk
  2) Greedy
  3) $\epsilon$-greedy
  4) Simulated annealing
  5) Local search
  6) Beam search
  7) Branch-and-bound
  8) Cutting plane


## Solvers to Compare with

[Gurobi](https://www.gurobi.com/) is the state-of-the-art solver. The license is required, and professors/students at universities can obtain the __academic license for free__. We recommend to use Gurobi if users have licenses, since its performance is the best.

[SCIP](https://www.scipopt.org/index.php#welcome) is a well-known open-source solver, and its simplex is commonly used in "learn to branch/cut". If users do not have Gurobi licenses, SCIP is a good choice since it is __open-source and free__. Although its performance is not as good as Gurobi, we recommend to use SCIP if users do not have Gurobi licenses. 


## Other Solvers

[COPT](https://www.copt.de/): a mathematical optimization solver for large-scale problems.

[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer): a high-performance mathematical programming solver for linear programming, mixed integer programming, and quadratic programming.

[Xpress](https://www.fico.com/en/products/fico-xpress-optimization): an extraordinarily powerful, field-installable Solver Engine.

[BiqMac](https://biqmac.aau.at/): a solver only for binary quadratic or maxcut. Users should upload txt file, but the response time is not guaranteed. If users use it, we recommend to [download](https://biqmac.aau.at/) the sources and run it by local computers. 


## Store Results 

The partial results are stored in the folder "result" of this repo. All the results are stored in [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE: hojh for China users). 

## Performance
The performance of maxcut compared with other methods or solvers is presented [here](https://github.com/AI4Finance-Foundation/RLSolver/tree/main/helloworld/maxcut).




## File Structure

```
RLSolver
└──helloworld
   └──maxcut
        └──data
        └──mcmc_sim
        └──result
        └──learn_to_anneal_x.py (ours)
        └──gurobi.py
        └──scip.py
        └──random_walk.py
        └──greedy.py
        └──simulated_annealing.py
        └──utils.py
└──opt_methods
└──readme
   └──graph_partitioning.md
   └──maxcut.md
   └──tsp.md
└──rlsolver (main folder)
   └──data (datasets for problems)
   └──mcmc_sim
   └──result (store output files)
   └──rlsolver_learn2opt
      └──mimo
      └──tensor_train
   └──graph_partitioning.py
   └──graph_partitioning_gurobi.py
   └──maxcut.py
   └──maxcut_H2O.py
   └──maxcut_gurobi.py
   └──tsp.py
   └──tsp_gurobi.py
   └──utils.py


```

## Progress
- [x] MIMO
- [x] Maxcut
- [x] TNCO
- [x] quantum_circuits 
## TODO
- [ ] TSP
- [ ] Vehicle routing problem
- [ ] Graph partitioning
- [ ] Minimum vertex cover 
- [ ] MILP
- [ ] portfolio_management


## Related Websites

+ [Benchmarks for optimization softwre](http://plato.asu.edu/bench.html) 
+ [Awesome Machine Learning for Combinatorial Optimization Resources](https://github.com/Thinklab-SJTU/awesome-ml4co)
+ [Machine Learning for Combinatorial Optimization - NeurIPS 2021 Competition](https://github.com/ds4dm/ml4co-competition)
