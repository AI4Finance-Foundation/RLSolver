# RLSolver: High-performance RL solvers 

Combinatorial optimizations and nonconvex optimizations

It is based on [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) and [OpenAI Gym](https://github.com/openai/gym).

The following two key technologes are under active development: 

- **Massively parallel simuations** of gym-environments on GPU, using thousands of CUDA cores and tensor cores.

- **Podracer scheduling** on a GPU cloud, e.g., DGX-2 SuperPod.


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
└── └── utils


```
