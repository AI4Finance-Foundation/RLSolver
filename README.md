# RLSolver: 

Massively Parallel Gym-Environments for Combinatorial and Nonconvex Optimizations
## Outline

- [File Structure](#File Structure)

## File Structure

```
RLSolver
├── rlsolver (main folder)
│   ├── cfg
|   |   ├── task
|   |   |   ├── maxcut.yaml
|   |   |   ├── tsp.yaml
|   |   |   └── mimo_beamforming.yaml
|   |   └── train
|   |       ├── maxcut_DQN.yaml
|   |       ├── maxcut_REINFORCE.yaml
|   |       ├── tsp_DQN.yaml
|   |       ├── tsp_REINFORCE.yaml
|   |       ├── mimo_beamforming_DQN.yaml
|   |       └── mimo_beamforming_REINFORCE.yaml
|   ├── envs
|   |   ├── base.py
|   |   ├── maxcut.py
|   |   ├── tsp.py
|   |   └── mimo_beamforming.py
|   ├── HPClib
|   |   ├── linux
|   |   └── windows
|   └── utils
|
|   
└── README.md

```