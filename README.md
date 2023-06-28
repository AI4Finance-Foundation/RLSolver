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

## Datasets
* Maxcut: [Gset dataset at Stanford](https://web.stanford.edu/~yyye/yyye/Gset/)
* TSP: [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/)
* 
  

## Benchmarks
[2023 AAAI Reinforcement Learning for Branch-and-Bound Optimisation using Retrospective Trajectories] (https://github.com/cwfparsonson/retro_branching/tree/master)


## Outline

- [File Structure](#File-Structure)

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
   └──data
   └──envs
   └──result (store output files)
   └──rlsolver_learn2opt
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
