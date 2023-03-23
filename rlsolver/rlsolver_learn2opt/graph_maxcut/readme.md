# Graph Maxcut
## File structure
```python
main.py  #main functions: main, train_opt_net, roll_out
opt_gurobi.py # run Gurobi for reference performance, Gurobi license is required
utils.py # utils file, including opt_net, opt_variable, obj_fun, etc.
```
## Run our method with command 

```
python main.py #N #Sparsity
```

## Run Gurobi with command 

```
python opt_gurobi.py #N #Sparsity
```


## Experiment Results

Sprsity=0.5
|Maxcut |Gurobi, (Running Time)|Ours, (Training Time)|
|-------|------|----|
|N=20   | 67 (5s)  | 67 (20s)|
|N=30   | 132 (10s)  | 132 (30s)|
|N=100   | 1408 (2000s)  | 1409 (800s)|
|N=1000   |  128508 (4400s)  |  129900 (1000s)|
|N=5000||3179905.75(1000s)|

Inference time of our method is less than 10 seconds.


Gset dataset

| graph | #nodes| #edges |  BLS | DSDP | KHLWG | RUN-CSP | PI-GNN | Ours | relative error $\epsilon$ |
|---|----------|-------|-----|-----|--------|----------|------|----|---------------------------|
|G14 | 800 | 4694 | 3064| | 2922 | 3061 | 2943 | 3026 || $1.24 %$|
|G15 | 800 | 4661 |  $3050$ | 2938 | $3050$ | 2928 | 2990 || $\%$ |
|G22 | 2000 | 19990 |  $13359$ | 12960 | $13359$ | 13028 | 13181 || $0.89 \%$ |
|G49 | 3000 | 6000 |  $6000$ | $6000$ | $6000$ | $6000$ | 5918 || $1.37 \%$ |
|G50 | 3000 | 6000 |  $5880$ | $5880$ | $5880$ | $5880$ | 5820 || $1.00 \%$ |
|G55 | 5000 | 12468 |  $10294$ | 9960 | 10236 | 10116 | 10138 || $1.25 \%$ |
|G70 | 10000 | 9999 |  $9541$ | 9456 | 9458 | - | 9421 |8917.02 | $1.20 \%$ |



## Workflow
 ![pipeline](pipeline.jpg)
