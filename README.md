# RLSolver: High-performance GPU-based Solvers for Nonconvex and NP-Complete Problems


We aim to showcase that reinforcement learning (RL) or machine learning (ML) with GPUs delivers the best benchmark performance for large-scale nonconvex and NP-complete problems. RL with the help of GPU computing can obtain high-quality solutions within short time. 

[![](https://dcbadge.vercel.app/api/server/trsr8SXpW5)](https://discord.gg/trsr8SXpW5)


# Problem-oriented Repos

- [Maxcut](https://github.com/zhumingpassional/Maxcut)
- [TSP](https://github.com/zhumingpassional/TSP)

# Key Technologies
- **RL/ML tricks** such as learn to optimize and curriculum learning.
- **OR tricks** such as local search and tabu search.
- **Massively parallel sampling** of Markov chain Monte Carlo (MCMC) simulations on GPU using thousands of CUDA cores and tensor cores.
- **Podracer scheduling** on a GPU cloud such as DGX-2 SuperPod.
- 
# Key References

- Mazyavkina, Nina, et al. "Reinforcement learning for combinatorial optimization: A survey." Computers & Operations Research 134 (2021): 105400.

- Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine learning for combinatorial optimization: a methodological tour d’horizon." European Journal of Operational Research 290.2 (2021): 405-421.

- Peng, Yun, Byron Choi, and Jianliang Xu. "Graph learning for combinatorial optimization: a survey of state-of-the-art." Data Science and Engineering 6, no. 2 (2021): 119-141.

- Nair, Vinod, et al. "Solving mixed integer programs using neural networks." arXiv preprint arXiv:2012.13349 (2020).

- Makoviychuk, Viktor, et al. "Isaac Gym: High performance GPU based physics simulation for robot learning." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021.

# Workflow
<a target="\_blank">
	<div align="center">
		<img src=fig/work_flow.png width="60%"/>
	</div>
</a>  

# Structure
<a target="\_blank">
	<div align="center">
		<img src=fig/RLSolver_structure.png width="70%"/>
	</div>
</a>  

## Datasets
* [Maxcut](https://github.com/zhumingpassional/Maxcut):
  
  1) [Gset](https://web.stanford.edu/~yyye/yyye/Gset/) is stored in the "data" folder of this repo. The number of nodes is from 800 to 10000. 
  
  2) __Syn__ is the synthetic data obtained by calling the function generate_write in utils.py. The number of nodes is from 10 to 50000. The (partial) synthetic data is stored in the "data" folder of this repo. If users need all the synthetic data, please refer to [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE hojh for China users). 
  
* [TSP](https://github.com/zhumingpassional/TSP): [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
  

## Benchmarks


* Learning to branch
  
[code](https://github.com/cwfparsonson/retro_branching/tree/master) 2023 AAAI Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories 

[code](https://github.com/ds4dm/branch-search-trees) 2021 AAAI Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies

* Learning to cut

[code](https://github.com/Wenbo11/learntocut) 2020 ICML Reinforcement learning for integer programming: Learning to cut


* RL/ML-based heuristic
  
[code](https://github.com/Hanjun-Dai/graph_comb_opt)  (greedy) 2017 NeurIPS Learning Combinatorial Optimization Algorithms over Graphs

[code](https://github.com/optsuite/MCPG) (local search) 2023, A Monte Carlo Policy Gradient Method with Local Search for Binary Optimization

[code](https://github.com/JHL-HUST/VSR-LKH) (LKH for TSP) 2021 AAAI Combining reinforcement learning with Lin-Kernighan-Helsgaun algorithm for the traveling salesman problem 

* Annealing

[code](https://github.com/zhumingpassional/Maxcut/tree/master/baseline/variational_classical_annealing) (VCA) 2023 Machine_Learning Supplementing recurrent neural networks with annealing to solve combinatorial optimization problems

[code](https://github.com/zhumingpassional/Maxcut/tree/master/baseline/variational_neural_annealing) (VNA) 2021 Nature_Machine_Intelligence Variational neural annealing

* Classical methods
  - [Random walk](https://github.com/zhumingpassional/Maxcut/blob/master/baseline/random_walk.py)
  - [Greedy](https://github.com/zhumingpassional/Maxcut/blob/master/baseline/greedy.py)
  - [Simulated annealing](https://github.com/zhumingpassional/Maxcut/blob/master/baseline/simulated_annealing.py)
  - Local search
  - Beam search
  - Tabu search
  - Branch-and-bound
  - Cutting plane


## Solvers to Compare with

[Gurobi](https://www.gurobi.com/) is the state-of-the-art solver. The license is required, and professors/students at universities can obtain the __academic license for free__. 

[SCIP](https://www.scipopt.org/index.php#welcome) is a well-known open-source solver, and its simplex is commonly used in "learning to branch/cut". SCIP is __open-source and free__. 


## Other Solvers

[COPT](https://www.copt.de/): a mathematical optimization solver for large-scale problems.

[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer): a high-performance mathematical programming solver for linear programming, mixed integer programming, and quadratic programming.

[Xpress](https://www.fico.com/en/products/fico-xpress-optimization): an extraordinarily powerful, field-installable Solver Engine.

[BiqMac](https://biqmac.aau.at/): a solver only for binary quadratic or maxcut. Users should upload txt file, but the response time is not guaranteed. If users use it, we recommend to [download](https://biqmac.aau.at/) the sources and run it by local computers. 


## Store Results 

Partial results are stored in the folder "result" of this repo. All the results are stored in [Google Drive](https://drive.google.com/drive/folders/1gkpndZPj09ew-s9IvrWEZvvCFDWzd7vL?usp=sharing) or [Baidu Wangpan](https://pan.baidu.com/s/11ljW8aS2IKE9fDzjSm5xVQ) (CODE: hojh for China users). 

With respect to maxcut, please refer to [Maxcut](https://github.com/zhumingpassional/Maxcut). With respect to TSP, please refer to [TSP](https://github.com/zhumingpassional/TSP). 

## Performance
[Maxcut](https://github.com/zhumingpassional/Maxcut).
[TSP](https://github.com/zhumingpassional/TSP). 
[Quantum circuits](https://github.com/AI4Finance-Foundation/RLSolver/tree/main/rlsolver/rlsolver_quantum_circuits)
[MIMO](https://github.com/AI4Finance-Foundation/RLSolver/tree/main/rlsolver/rlsolver_mimo_beamforming)
[Compressive sensing](https://github.com/AI4Finance-Foundation/RLSolver/tree/main/rlsolver/rlsolver_compressive_sensing)

## File Structure

```
RLSolver
└──helloworld
   └──maxcut
        └──data
        └──result
        └──utils.py
        └──mcmc.py
        └──l2a.py (ours)
        └──baseline
            └──gurobi.py
            └──random_walk.py
            └──greedy.py
            └──simulated_annealing.py
└──benchmark
   └──maxcut.md
   └──graph_partitioning.md
   └──tsp.md
   └──tnco.md
└──rlsolver (main folder)
   └──utils.py
   └──data
      └──graph
      └──quantum_circuits
      └──milp_coefs
      └──binary_coefs
   └──problems
      └──maxcut
          └──baseline
          └──mcmc.py
          └──l2a.py(ours)
      └──tnco
          └──baseline
          └──mcmc.py
          └──l2a.py(ours)
      └──mimo
          └──baseline
          └──mcmc.py
          └──l2a.py(ours)




```

## Finished
- [x] MIMO
- [x] [Maxcut](https://github.com/zhumingpassional/Maxcut)
- [x] TNCO
- [x] quantum_circuits 
## TODO
- [ ] [TSP](https://github.com/zhumingpassional/TSP)
- [ ] Vehicle routing problem
- [ ] Graph partitioning
- [ ] Minimum vertex cover 
- [ ] MILP
- [ ] portfolio_management


## Related Websites
+ [Maxcut](https://github.com/zhumingpassional/Maxcut)
+ [TSP](https://github.com/zhumingpassional/TSP)
+ [Benchmarks for optimization softwre](http://plato.asu.edu/bench.html) 
+ [Awesome Machine Learning for Combinatorial Optimization Resources](https://github.com/Thinklab-SJTU/awesome-ml4co)
+ [Machine Learning for Combinatorial Optimization - NeurIPS 2021 Competition](https://github.com/ds4dm/ml4co-competition)
