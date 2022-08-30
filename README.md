# RLSolver: High-performance RL solvers 

Combinatorial optimizations and nonconvex optimizations

It is based on [ElegantRL](https://github.com/AI4Finance-Foundation/ElegantRL) and [OpenAI Gym](https://github.com/openai/gym).

The key technologes are: 

- **Massively parallel simuations** of gym-environments on GPU, using thousands of CUDA cores and tensor cores.

- **Podracer scheduling** on a GPU cloud, e.g., DGX-2 SuperPod.


## Outline

- [File Structure](#File Structure)

## File Structure

```
RLSolver
├── rlsolver (main folder)
    ├── envs
    |   (combinatorial optimiations)
    |   ├── base.py
    |   ├── maxcut.py
    |   ├── tsp.py
    |   ├── milp.py
    |   ├── portfolio.py
    |   (nonconvex optimizations)
    |   ├── mimo_beamforming.py
    |   ├── mimo_beamforming_relay.py
    |   └── learn2optimize
    ├── helloworld
    └── utils


```
