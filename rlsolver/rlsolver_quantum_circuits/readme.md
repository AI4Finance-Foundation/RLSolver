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

|Form|N=10|N=30|N=50|N=100|
|-------| ----|------- | -----|------ |
|![TT](https://user-images.githubusercontent.com/75991833/217780619-40f42213-62b8-4db5-bfa9-0c9f8d97081d.png)|2464<br>2464<br>GAP：0%<br>Cotengra:2506|**RL_log10[FLOPs]**:9.33<br>**Brute_Force_log10[FLOPs]**:9.33<br>**Cotengra_log10[FLOPs]**:9.63|**RL_log10[FLOPs]**:15.35<br>**Brute_Force_log10[FLOPs]**:15.35<br>**Cotengra_log10[FLOPs]**:15.65|**RL_log10[FLOPs]**:30.40<br>**Brute_Force_log10[FLOPs]**:30.40<br>**Cotengra_log10[FLOPs]**:30.71|
|![TR](https://user-images.githubusercontent.com/75991833/217780649-80acaa33-030b-46b9-9fc6-bf5bc84167a3.png)|4864<br>4864<br>GAP：0.0%|4.2954993e9<br>4.29549879296e9<br>GAP：1.180e-7|4.5036e15<br>4.503600314565263e15<br>GAP：6.9e-8|5.0706024e+30<br>5.070602400912918e+30<br>GAP：1.8e-10|

|Form|N=7|N=15|N=31|N=63|N=127|
|-------| ----|------- | -----|------ |------ |
|![TTN](https://user-images.githubusercontent.com/75991833/217782955-cd2cd6e8-d0b8-4187-b7e7-d202266bcbfb.png)|30<br>30<br>GAP：0%|78<br>78<br>GAP：0%|174<br>180<br>GAP：3.45%|366<br>400<br>GAP：9.29%|750<br>832<br>GAP：10.94%|

|Form|N=9|N=16|N=49|N=81|N=100|
|-------| ----|------- | -----|------ |------ |
|![G](https://user-images.githubusercontent.com/75991833/217780858-eff2a41e-3847-4ed2-bbcb-5db8aa86d9ce.png)||||||
