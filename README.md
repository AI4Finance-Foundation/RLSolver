# ElegantRL_Solver: High-performance GPU-based Solvers for NP-hard and NP-complete Problems

We aim to showcase the machine learning deliver the best benchmark performance for NP-hard and NP-complete problems.

In the meantime, this repo will also include our codes and tricks when playing with nonconvex and nonlinear optimization problems.

[Benchmark](http://plato.asu.edu/bench.html) for combinatorial optimization problems.

For deep reinforcement learning algorithms, we use [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) and [OpenAI Gym](https://github.com/openai/gym).

The following two key technologies are under active development: 

- **Massively parallel simuations** of gym-environments on GPU, using thousands of CUDA cores and tensor cores.

- **Podracer scheduling** on a GPU cloud, e.g., DGX-2 SuperPod.

Key references:

- Mazyavkina, Nina, et al. "Reinforcement learning for combinatorial optimization: A survey." Computers & Operations Research 134 (2021): 105400.

- Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine learning for combinatorial optimization: a methodological tour d’horizon." European Journal of Operational Research 290.2 (2021): 405-421.

- Peng, Yun, Byron Choi, and Jianliang Xu. "Graph learning for combinatorial optimization: a survey of state-of-the-art." Data Science and Engineering 6, no. 2 (2021): 119-141.

- Nair, Vinod, et al. "Solving mixed integer programs using neural networks." arXiv preprint arXiv:2012.13349 (2020).

- Makoviychuk, Viktor, et al. "Isaac Gym: High performance GPU based physics simulation for robot learning." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021.

  

## Environments

ElegantRL_Solver will includes the following environments:
* Classical Simulation of Quantum Circuits.
* Compressive Sensing.
* Portfolio Management.
* MIMO Beamforming in 5G/6G.
* OR-Gym.
* Massively parallel environments.
* Gym-style environments.

## Datasets
* Maxcut: [Gset dataset at Stanford](https://web.stanford.edu/~yyye/yyye/Gset/)
* TSP: [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
  
  
## Massively parallel gym-style environments

Important functions 

* reset(): Initialize the variables
* obj(): Calculate the objective function, i.e., Halmiltonian.
* reward(mu1: Tensor, mu2: Tensor): Calculate the Halmiltonian of from the graph mu1 to another graph mu2. 

## Benchmarks

* Learn to branch
  
[code](https://github.com/cwfparsonson/retro_branching/tree/master) 2023 AAAI Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories 

[code](https://github.com/ds4dm/branch-search-trees) 2021 AAAI Parameterizing Branch-and-Bound Search Trees to Learn Branching Policies

* Learn to cut

[code](https://github.com/Wenbo11/learntocut) 2020 ICML Reinforcement learning for integer programming: Learning to cut


* RL + algorithm/heuristic

[code](https://github.com/JHL-HUST/VSR-LKH) 2021 AAAI Combining reinforcement learning with Lin-Kernighan-Helsgaun algorithm for the traveling salesman problem 

  
## Solvers to compare with

[Gurobi](https://www.gurobi.com/) (the state-of-the-art solver),  [download/install](https://www.gurobi.com/downloads/gurobi-software/)

[SCIP](https://www.scipopt.org/index.php#welcome) (an open-source solver, and its simplex is commonly used in "learning to branch/cut") [download/install](https://scipopt.org/doc/html/md_INSTALL.php)

## Solvers may be used

[CPLEX](https://www.ibm.com/products/ilog-cplex-optimization-studio/cplex-optimizer) [download/install](https://www.ibm.com/support/pages/downloading-ibm-ilog-cplex-optimization-studio-2010)

[Xpress](https://www.fico.com/en/products/fico-xpress-optimization) [download/install](https://www.fico.com/fico-xpress-optimization/docs/latest/installguide/dhtml/chapinst1.html)


## File Structure

```
RLSolver
└──helloworld
   └──maxcut.py
   └──maxcut_env.py
└──opt_methods
└──readme
   └──graph_partitioning.md
   └──maxcut.md
   └──tsp.md
└──rlsolver (main folder)
   └──data (datasets for problems)
   └──envs
   └──result (store output files)
   └──rlsolver_learn2opt
      └──mimo
      └──tensor_train
   └──graph_partitioning.py
   └──graph_partitioning_gurobi.py
   └──maxcut.py
   └──maxcut_gurobi.py
   └──tsp.py
   └──tsp_gurobi.py
   └──utils.py


```

## Progress

- [x] MIMO
- [x] Maxcut
- [x] TNCO
- [ ] TSP
- [ ] MILP
- [ ] portfolio_management
- [ ] quantum_circuits


## Related Repos:
+ [RL4CO](https://github.com/kaist-silab/rl4co)
+ [Awesome Machine Learning for Combinatorial Optimization Resources](https://github.com/Thinklab-SJTU/awesome-ml4co)
+ [Machine Learning for Combinatorial Optimization - NeurIPS 2021 Competition](https://github.com/ds4dm/ml4co-competition)
