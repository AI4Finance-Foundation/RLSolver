# Graph Maxcut
## File structure
```python
main.py  #main functions: main, train_opt_net, roll_out
opt_gurobi.py # run Gurobi for reference performance, Gurobi should be installed and its license is required
utils.py # utils file, including opt_net, opt_variable, obj_fun, etc.
```
## Run our method with command 

Format:
```
python main.py #gpu_id (-1: cpu, >=0: gpu) #choice (0: Synthetic data, 1: Gset) ...
```

If using synthetic data:
```
python main.py #gpu_id (-1: cpu, >=0: gpu) 0 #N #Sparsity 
```

If using Gset:
```
python main.py #gpu_id (-1: cpu, >=0: gpu) 1 #GsetId
```
## Download data using the following link:

```
https://drive.google.com/drive/folders/13tTd73NahyX2_WU9cFL0BhoZc-MIbtxc?usp=sharing
```

## Download the label found by us using the following link:

```
https://drive.google.com/drive/folders/1W-NwXVnDhKbRnWS3qh54m0WL8J_xx-6s?usp=sharing
```


## Run Gurobi with command 

```
python opt_gurobi.py #N #Sparsity #gpu_id (-1: cpu, >=0: gpu) #choice (0: Synthetic data, 1: Gset)
```


## Experiment Results

Synthetic data at sparsity = 0.5
 
|Maxcut |Gurobi, (Running Time)| Ours-CPU (Training Time) | Ours-GPU (Training Time) |improvement using CPU |improvement using GPU |
|-------|------|----| ---- |---- |---- |
|N=20   | 67, (5s)  | 67, (20s)| 71, (36)s | same | +5.97% |
|N=30   | 132, (10s)  | 132, (30s)| 135, (93s) | same | +2.27% |
|N=100   | 1408, (2000s)  | 1409, (800s)| 1415, (33s) | + 0.071\%, (2.5x) | +0.49%, (151.5x) |
|N=1000   |  128508, (4400s)  |  129900, (1000s)| 129714, (119s)| + 1.072\%, (4.4x) | +0.94%, (36.97x) |
|N=5000 | |3124937, (4520s)  | 3175813, (202s) | | |

Inference time of our method is less than 10 seconds.


[Gset dataset at Stanford](https://web.stanford.edu/~yyye/yyye/Gset/)

| graph | #nodes| #edges | BLS | DSDP | KHLWG | RUN-CSP | PI-GNN | Gurobi | Ours-GPU | improvement | 
|---|----------|----|---|-----|-----|--------|----------|------|----|----|
|G14 | 800 | 4694 | 3064| | 2922 | 3061 | 2943  |3056 (24h) | 3003 | -1.99\%|
|G15 | 800 | 4661 | 3050 | 2938 | 3050 | 2928 | 2990  | | 2965 | -2.78\% | 
|G22 | 2000 | 19990 |13359 | 12960 | 13359 | 13028 | 13181  | |12991 |  -2.75\% | 
|G49 | 3000 | 6000 | 6000 | 6000 | 6000 | 6000 | 5918  | | 5326|  -11.23\% | 
|G50 | 3000 | 6000 | 5880 | 5880 | 5880 | 5880 | 5820  | | 5320|  -9.52\% | 
|G55 | 5000 | 12468 | 10294 | 9960 | 10236 | 10116 | 10138  | |9709 |  -5.68\% | 
|G70 | 10000 | 9999 |9541 | 9456 | 9458 | - | 9421  | |8917 | -6.54 \% | 


<!-- 
## Workflow
 ![pipeline](pipeline.jpg) -->
