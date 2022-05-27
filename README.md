# graph_maxcut
## Command to run DHN
```python maxcut.py #GPU_ID #vertex_num```

In each train epsode, DHN save its model as ***policy_{vertex_num}.pth***.

The adjacency matrix is saved as ***adjacency_{vertex_num}.npy***.


For example, command to train a DHN on a graph with 10 nodes on GPU 0 is

```python maxcut.py 0 10```.

The adjacency matrix is saved as ***adjacency_10.npy***.

In each train epsode, DHN save its model as ***policy_10.pth***.


## Command to test DHN
```python test_dhn.py #vertex_num```

test_dhn.py auto load a trained model with filename ***policy_{vertex_num}.pth*** and load a adjacency matrix with filename ***adjacency_{vertex_num}.npy***.

## Command to run brute force search
```python brute_force_search.py #vertex_num```

brute_force_search.py auto load a adjacency matrix with filename ***adjacency_{vertex_num}.npy*** and do the brute force.
