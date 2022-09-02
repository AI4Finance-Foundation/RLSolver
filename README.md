# RLSolver: High-performance RL solvers

For combinatorial optimizations and nonconvex optimizations.

For combinatorial optimization problems, we compare with [Benchmark](http://plato.asu.edu/bench.html).

For nonconvex optimization problems, we aim to find high-quality local minimum, even global minimum.

This project is built on [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) and [OpenAI Gym](https://github.com/openai/gym).

The following two key technologes are under active development: 

- **Massively parallel simuations** of gym-environments on GPU, using thousands of CUDA cores and tensor cores.

- **Podracer scheduling** on a GPU cloud, e.g., DGX-2 SuperPod.

Several key references:

- Mazyavkina, Nina, et al. "Reinforcement learning for combinatorial optimization: A survey." Computers & Operations Research 134 (2021): 105400.

- Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine learning for combinatorial optimization: a methodological tour d’horizon." European Journal of Operational Research 290.2 (2021): 405-421.

- Makoviychuk, Viktor, et al. "Isaac Gym: High Performance GPU Based Physics Simulation For Robot Learning." Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). 2021.

- Nair, Vinod, et al. "Solving mixed integer programs using neural networks." arXiv preprint arXiv:2012.13349 (2020).


## Outline

- [File Structure](#File-Structure)

## File Structure

```
RLSolver
├── optimal
|   ├──branch-and-bound.py
|   └──cutting_plane.py
├── helloworld
|   ├──milp
|   ├──tsp
|   └──graph_maxcut
├── rlsolver (main folder)
|   ├── envs
|   |   (nonconvex optimiations)
|   |   ├── learn2optimize
|   |   └── mimo_beamforming
|   |   (combinatorial optimizations)
|   |   ├── portfolio_management
|   |   ├── quantum_circuits
|   |   ├── vehicle_routing
|   |   ├── virtual_machine_placement
|   |   └── chip_design
|   |── rlsolver_learn2optimize
|   |── rlsolver_mimo_beamforming
|   |── rlsolver_portfolio_management
|   |── rlsolver_quantum_circuits
    └── utils


```
