# Classical simulation of quantum circuits using tensor networks

Roadmap
- [x] Tensor network representations
- [ ] Gym-environments
- [ ] Massively parallel simulation of environment
- [ ] Baseline methods: greedy, BRB
- [ ] Dataset
- [ ] RL achieves optimality and outperforms baselines

## Data Generation

![image](https://user-images.githubusercontent.com/75991833/218404111-e23e9e9b-c2ac-4648-aa04-9a7208fa7693.png)

In order to facilitate the calculation of Reward without affecting the results, we set the value of the tensor index of the unconnected relation in the state matrix to 1.

## Experimental Results
- **OE_greedy**: Daniel, G., Gray, J., et al. (2018). Opt\_einsum-a python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 3(26):753
https://github.com/dgasmith/opt_einsum

- **CTG_Greedy、CTG_Kahypar**: Gray, J. and Kourtis, S. (2021). Hyper-optimized tensor network contraction. Quantum, 5:410.
https://github.com/jcmgray/cotengra

|Form|N=10|N=30|N=50|N=100|
|-------| ----|------- | -----|------ |
|![TT](https://user-images.githubusercontent.com/75991833/217780619-40f42213-62b8-4db5-bfa9-0c9f8d97081d.png)|OE_greedy: 3.848<br>CTG_Greedy: 3.693<br>CTG_Kahypar: 3.69<br>RL_Simulator: **3.392**<br>Brute_Force: 3.392|OE_greedy: 9.855<br>CTG_Greedy: 9.633<br>CTG_Kahypar: 9.63<br>RL_Simulator: **9.332** <br>Brute_Force: 9.332|OE_greedy: 15.875<br>CTG_Greedy: 15.654<br>CTG_Kahypar: 15.65<br>RL_Simulator: **15.353**<br>Brute_Force: 15.353|OE_greedy: 30.927<br>CTG_Greedy: 30.705<br>CTG_Kahypar: 30.71<br>RL_Simulator: **30.404**<br>Brute_Force: 30.404|
|![TR](https://user-images.githubusercontent.com/75991833/217780649-80acaa33-030b-46b9-9fc6-bf5bc84167a3.png)|3.687<br>3.687<br>GAP：0.0%|9.633<br>9.633<br>|15.654<br>15.654|30.705<br>30.705|

|Form|N=200|N=300|N=500|N=800|
|-------| ----|------- | -----|------ |
|TT|OE_greedy: 61.030<br>CTG_Greedy: 60.808<br>CTG_Kahypar: 60.81<br>RL_Simulator: **xxx**<br>Brute_Force: xxx|OE_greedy:  91.133<br>CTG_Greedy: 90.911<br>CTG_Kahypar: 90.91<br>RL_Simulator: **xxx**<br>Brute_Force: xxx|OE_greedy: 151.339<br>CTG_Greedy: 151.337<br>CTG_Kahypar: 151.12<br>RL_Simulator: **xxx**<br>Brute_Force: xxx|OE_greedy: 241.648<br>CTG_Greedy: 241.426<br>CTG_Kahypar: 241.43<br>RL_Simulator: **xxx**<br>Brute_Force: xxx|




|Form|N=7|N=15|N=31|N=63|N=127|
|-------| ----|------- | -----|------ |------ |
|![TTN](https://user-images.githubusercontent.com/75991833/217782955-cd2cd6e8-d0b8-4187-b7e7-d202266bcbfb.png)|30<br>30<br>GAP：0%|78<br>78<br>GAP：0%|174<br>180<br>GAP：3.45%|366<br>400<br>GAP：9.29%|750<br>832<br>GAP：10.94%|

|Form|N=9|N=16|N=49|N=81|N=100|
|-------| ----|------- | -----|------ |------ |
|![G](https://user-images.githubusercontent.com/75991833/217780858-eff2a41e-3847-4ed2-bbcb-5db8aa86d9ce.png)||||||
