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

The quantum circuits are of quantum bit number n=53 and periods m=12, 14, 16, 18, 20, respectively, and the source of the dataset is: https://datadryad.org/stash/dataset/doi:10.5061/dryad.k6t1rj8                                                                                                                                          
The file and the transformed txt file are located in sycamore_circuits/sycamore.

**Our results**

|Sycamore_Circuit|m=12|m=14|m=16|m=18|m=20|
|-------|------- | -----|------ |------ |------ |
|Results|OE_greedy: 17.795<br>CTG_Greedy: 17.065<br>CTG_Kahypar: 13.408<br>**RL: 14.704**|OE_greedy: 19.679<br>CTG_Greedy: 19.282<br>CTG_Kahypar: 14.152<br>RL:|OE_greedy: 25.889<br>CTG_Greedy: 23.151<br>CTG_Kahypar: 17.012<br>RL: **16.589**|OE_greedy: 26.793<br>CTG_Greedy: 23.570<br>CTG_Kahypar: 17.684<br>RL:|OE_greedy: 26.491<br>CTG_Greedy: 25.623<br>CTG_Kahypar: 18.826<br>RL: **17.078**|

- **OE_greedy**: Daniel, G., Gray, J., et al. (2018). Opt\_einsum-a python package for optimizing contraction order for einsum-like expressions. Journal of Open Source Software, 3(26):753
https://github.com/dgasmith/opt_einsum

- **CTG_Greedy、CTG_Kahypar**: Gray, J. and Kourtis, S. (2021). Hyper-optimized tensor network contraction. Quantum, 5:410.
https://github.com/jcmgray/cotengra

Results of **ICML-Optimizing tensor network contraction using reinforcement learning**

|Sycamore_Circuit|m=10|m=12|m=14|m=16(Not-Giving)|m=18(Not-Giving)|m=20|
|-------| ----|------- | -----|------ |------ |------ |
|Results|OE_greedy: 14.756<br>CTG_Greedy: 10.577<br>CTG_Kahypar: 10.304<br>RL_TNCO: 10.736|OE_greedy: 20.471<br>CTG_Greedy: 14.009<br>CTG_Kahypar: 13.639<br>RL_TNCO: 12.869|OE_greedy: 18.182<br>CTG_Greedy: 15.283<br>CTG_Kahypar: 14.704<br>RL_TNCO: 14.420|OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL_TNCO: |OE_greedy: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL_TNCO: |OE_greedy: 31.310<br>CTG_Greedy: 18.934<br>CTG_Kahypar: 18.765<br>RL_TNCO: 18.544|

- **TNCO**: Meirom E, Maron H, Mannor S, et al. Optimizing tensor network contraction using reinforcement learning. International Conference on Machine Learning. PMLR, 2022: 15278-15292.

![image](https://user-images.githubusercontent.com/75991833/227595309-a341713d-0247-4f3b-a12b-d94ac74af351.png)


|TT|N=10|N=50|N=100|N=200|N=400|N=600|N=800|N=1000|N=1500|N=2000|
|-------| ----|------- | -----|------| ----|------- | -----|------ | -----|------ |
|Results|OE_greedy: 3.848<br>OE_dynamic: 3.693<br>CTG_Greedy: 3.693<br>CTG_Kahypar: 3.690<br>**RL: 3.392**|OE_greedy: 15.875<br>OE_dynamic: N<br>CTG_Greedy: 15.654 <br>CTG_Kahypar: 15.650<br>**RL: 15.352**|OE_greedy: 30.927<br>OE_dynamic: N<br>CTG_Greedy: 30.705<br>CTG_Kahypar: 30.710<br>**RL: 30.404**|OE_greedy: 61.030<br>OE_dynamic: N<br>CTG_Greedy: 60.808<br>CTG_Kahypar: 60.810<br>**RL: 60.507**|OE_greedy: 121.236<br>OE_dynamic: N<br>CTG_Greedy: 121.014<br>CTG_Kahypar: 121.010<br>**RL: 120.713**|OE_greedy:  181.442<br>OE_dynamic: N<br>CTG_Greedy: 181.220<br>CTG_Kahypar: 181.220<br>**RL: 180.919**|OE_greedy: 241.648<br>OE_dynamic: N<br>CTG_Greedy: 241.426<br>CTG_Kahypar: 241.430<br>**RL: 241.125**|OE_greedy: 301.854<br>OE_dynamic: N<br>CTG_Greedy: 301.632<br>CTG_Kahypar: 301.630<br>**RL: 301.331**|OE_greedy: N<br>OE_dynamic: N<br>CTG_Greedy: N<br>CTG_Kahypar: 452.150<br>**RL: 451.846**|OE_greedy: N<br>OE_dynamic: N<br>CTG_Greedy: N<br>CTG_Kahypar: 602.660<br>**RL: 602.361**|

|TR|N=10|N=50|N=100|N=200|N=400|N=600|N=800|N=1000|N=1500|N=2000|
|-------| ----|------- | -----|------| ----|------- | -----|------ | -----|------ |
|Results|OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |OE_greedy: <br>OE_dynamic: <br>CTG_Greedy: <br>CTG_Kahypar: <br>RL: |



|TTN|Height=3|Height=4|Height=5|Height=6|Height=7|Height=8|Height=9|Height=10|
|-------| ----|------- | -----|------ |------ |------ |------ |------ |
|Results|OE_greedy: 3.304<br>OE_dynamic: 3.158<br>CTG_Greedy: 3.158<br>CTG_Kahypar: 3.16<br>RL: **XXX**|OE_greedy: 5.476<br>OE_dynamic: 5.427<br>CTG_Greedy: 5.427<br>CTG_Kahypar: 5.43<br>RL: **XXX**|OE_greedy: 10.238<br>OE_dynamic: N<br>CTG_Greedy: 10.235<br>CTG_Kahypar: 10.24<br>RL: **XXX**|OE_greedy: 19.868<br>OE_dynamic: N<br>CTG_Greedy: 19.868<br>CTG_Kahypar: 19.87<br>RL: **XXX**|OE_greedy: 39.134<br>OE_dynamic: N<br>CTG_Greedy: 39.134<br>CTG_Kahypar: 39.13<br>RL: **XXX**|OE_greedy: 77.666<br>OE_dynamic: N<br>CTG_Greedy: 77.666<br>CTG_Kahypar: 77.67<br>RL: **XXX**|OE_greedy: 154.729<br>OE_dynamic: N<br>CTG_Greedy: 154.729<br>CTG_Kahypar: 154.730<br>RL: **XXX**|OE_greedy: N<br>OE_dynamic: N<br>CTG_Greedy: N<br>CTG_Kahypar: 308.860<br>RL: **XXX**|



|MERA|Height=3|Height=4|Height=5|Height=6|Height=7|Height=8|Height=9|Height=10|
|-------| ----|------- | -----|------ |------ |------ |------ |------ |
|Results|OE_greedy: 3.595<br>OE_dynamic: 3.325<br>CTG_Greedy: 3.609<br>CTG_Kahypar: 3.600<br>RL: **XXX**|OE_greedy: 5.793<br>OE_dynamic: N<br>CTG_Greedy: 5.393<br>CTG_Kahypar: 5.390<br>RL: **XXX**|OE_greedy: 11.446<br>OE_dynamic: N<br>CTG_Greedy: 10.111<br>CTG_Kahypar: 10.110<br>RL: **XXX**|OE_greedy: 21.079<br>OE_dynamic: N<br>CTG_Greedy: 19.743<br>CTG_Kahypar: 19.740<br>RL: **XXX**|OE_greedy: 39.009<br>OE_dynamic: N<br>CTG_Greedy: 39.009<br>CTG_Kahypar: 39.010<br>RL: **XXX**|OE_greedy: 77.541<br>OE_dynamic: N<br>CTG_Greedy: 77.541<br>CTG_Kahypar: 77.540<br>RL: **XXX**|OE_greedy: 154.604<br>OE_dynamic: N<br>CTG_Greedy: 154.604<br>CTG_Kahypar: 154.600<br>RL: **XXX**|OE_greedy: N<br>OE_dynamic: N<br>CTG_Greedy: N<br>CTG_Kahypar: 308.730<br>RL: **XXX**|

|PEPS|N=36|N=64|N=100|N=144|N=196|N=256|
|-------| ----|------- | -----|------ |------ |------ |
|Results|OE_greedy: 12.996<br>OE_dynamic: N<br>CTG_Greedy: 12.944<br>CTG_Kahypar: 12.590<br>RL: **XXX**|OE_greedy: 21.983<br>OE_dynamic: N<br>CTG_Greedy: 21.975<br>CTG_Kahypar: 21.050<br>RL: **XXX**|OE_greedy: 34.317<br>OE_dynamic: N<br>CTG_Greedy: 33.715<br>CTG_Kahypar: 31.890<br>RL: **XXX**|OE_greedy: 48.165<br>OE_dynamic: N<br>CTG_Greedy: 47.262<br>CTG_Kahypar: 45.140<br>RL: **XXX**|OE_greedy: 64.420<br>OE_dynamic: N<br>CTG_Greedy: 64.420<br>CTG_Kahypar: 60.790<br>RL: **XXX**|OE_greedy: 83.084<br>OE_dynamic: N<br>CTG_Greedy: 82.783<br>CTG_Kahypar: 78.850<br>RL: **XXX**|


