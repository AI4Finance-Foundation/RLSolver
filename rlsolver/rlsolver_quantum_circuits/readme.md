# Classical simulation of quantum circuits using tensor networks

Roadmap
- [x] Tensor network representations
- [ ] Gym-environments
- [ ] Massively parallel simulation of environment
- [ ] Baseline methods: greedy, BRB
- [ ] Dataset
- [ ] RL achieves optimality (or near-optimality) and outperforms other methods

## Data Generation

![image](https://user-images.githubusercontent.com/75991833/218404111-e23e9e9b-c2ac-4648-aa04-9a7208fa7693.png)

In order to facilitate the calculation of Reward without affecting the results, we set the value of the tensor index of the unconnected relation in the state matrix to 1.

## Experimental Results

Our results

|Sycamore_Circuit|m=12|m=14|m=16|m=18|m=20|
|-------|------- | -----|------ |------ |------ |
|Results|OE_greedy: 17.795<br>CTG_Greedy: 17.065<br>CTG_Kahypar: 13.408<br>RL:|OE_greedy: 19.679<br>CTG_Greedy: 19.282<br>CTG_Kahypar: 14.152<br>RL:|OE_greedy: 25.889<br>CTG_Greedy: 23.151<br>CTG_Kahypar: 17.012<br>RL:|OE_greedy: 26.793<br>CTG_Greedy: 23.570<br>CTG_Kahypar: 17.684<br>RL:|OE_greedy: 26.491<br>CTG_Greedy: 25.623<br>CTG_Kahypar: 18.826<br>RL:|

- **OE_greedy**: Daniel, G., Gray, J., et al. (2018). Opt\_einsum-a python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 3(26):753
https://github.com/dgasmith/opt_einsum

- **CTG_Greedy、CTG_Kahypar**: Gray, J. and Kourtis, S. (2021). Hyper-optimized tensor network contraction. Quantum, 5:410.
https://github.com/jcmgray/cotengra

Results of **ICML-Optimizing tensor network contraction using reinforcement learning**

|Sycamore_Circuit|m=10|m=12|m=14|m=16(Not-Giving)|m=18(Not-Giving)|m=20|
|-------| ----|------- | -----|------ |------ |------ |
|Results|OE_greedy: 14.756<br>CTG_Greedy: 10.577<br>CTG_Kahypar: 10.304<br>RL_TNCO: 10.736|OE_greedy: 20.471<br>CTG_Greedy: 14.009<br>CTG_Kahypar: 13.639<br>RL_TNCO: 12.869|OE_greedy: 18.182<br>CTG_Greedy: 15.283<br>CTG_Kahypar: 14.704<br>RL_TNCO: 14.420|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL_TNCO: |OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL_TNCO: |OE_greedy: 31.310<br>CTG_Greedy: 18.934<br>CTG_Kahypar: 18.765<br>RL_TNCO: 18.544|

- **TNCO**: Meirom E, Maron H, Mannor S, et al. Optimizing tensor network contraction using reinforcement learning[C]//International Conference on Machine Learning. PMLR, 2022: 15278-15292.



![TT_TR](https://user-images.githubusercontent.com/75991833/225349458-a374eee6-01ea-4bdc-8c37-341f4f5cf87d.png)

|TT|N=10|N=30|N=50|N=100|N=200|N=300|N=500|N=800|
|-------| ----|------- | -----|------| ----|------- | -----|------ |
|RES|OE_greedy: 3.848<br>CTG_Greedy: 3.693<br>CTG_Kahypar: 3.69<br>RL: **3.392**|OE_greedy: 9.855<br>CTG_Greedy: 9.633<br>CTG_Kahypar: 9.63<br>RL: **9.332** |OE_greedy: 15.875<br>CTG_Greedy: 15.654<br>CTG_Kahypar: 15.65<br>RL: **15.353**|OE_greedy: 30.927<br>CTG_Greedy: 30.705<br>CTG_Kahypar: 30.71<br>RL: **30.404**|OE_greedy: 61.030<br>CTG_Greedy: 60.808<br>CTG_Kahypar: 60.81<br>RL: **xxx**|OE_greedy:  91.133<br>CTG_Greedy: 90.911<br>CTG_Kahypar: 90.91<br>RL: **xxx**|OE_greedy: 151.339<br>CTG_Greedy: 151.337<br>CTG_Kahypar: 151.12<br>RL: **xxx**|OE_greedy: 241.648<br>CTG_Greedy: 241.426<br>CTG_Kahypar: 241.43<br>RL: **xxx**|

|TR|N=10|N=30|N=50|N=100|N=200|N=300|N=500|N=800|
|-------| ----|------- | -----|------| ----|------- | -----|------ |
|RES|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **3.687**|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **9.633** |OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **15.654**|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **30.705**|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **xxx**|OE_greedy:  <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **xxx**|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **xxx**|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: **xxx**|




|Form|N=7|N=15|N=31|N=63|N=127|N=255|
|-------| ----|------- | -----|------ |------ |------ |
|![image](https://user-images.githubusercontent.com/75991833/226081000-e507cc10-5c7f-4268-8f7a-418b8de0faa4.png)|OE_greedy: 3.304<br>CTG_Greedy: 3.158<br>CTG_Kahypar: 3.16<br>RL: **XXX**|OE_greedy: 5.476<br>CTG_Greedy: 5.427<br>CTG_Kahypar: 5.43<br>RL: **XXX**|OE_greedy: 10.238<br>CTG_Greedy: 10.235<br>CTG_Kahypar: 10.24<br>RL: **XXX**|OE_greedy: 19.868<br>CTG_Greedy: 19.868<br>CTG_Kahypar: 19.87<br>RL: **XXX**|OE_greedy: 39.134<br>CTG_Greedy: 39.134<br>CTG_Kahypar: 39.13<br>RL: **XXX**|OE_greedy: 77.666<br>CTG_Greedy: 77.666<br>CTG_Kahypar: 77.67<br>RL: **XXX**|

|Form|N=9|N=16|N=49|N=81|N=100|
|-------| ----|------- | -----|------ |------ |
|![G](https://user-images.githubusercontent.com/75991833/217780858-eff2a41e-3847-4ed2-bbcb-5db8aa86d9ce.png)||||||
