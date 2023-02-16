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

|Method|N=10|N=30|N=50|N=100|
|-------| ----|------- | -----|------ |
|![TT](https://user-images.githubusercontent.com/75991833/217780619-40f42213-62b8-4db5-bfa9-0c9f8d97081d.png)|2464<br>2464<br>GAP：0%|2147751200.00<br>2147751690.24<br>Gap: 0.0000002283%|2251800000000000<br>2251800157282631.5<br>Gap: 0.0000000698%|2.5353012e+30<br>2.535301200456459e+30<br>Gap: 0.0000000180%|
|![TR](https://user-images.githubusercontent.com/75991833/217780649-80acaa33-030b-46b9-9fc6-bf5bc84167a3.png)|4864<br>4864<br>GAP：0%|4295500000.0<br>?<br>GAP：0%|||
|![TTN](https://user-images.githubusercontent.com/75991833/217782955-cd2cd6e8-d0b8-4187-b7e7-d202266bcbfb.png)||||
|![G](https://user-images.githubusercontent.com/75991833/217780858-eff2a41e-3847-4ed2-bbcb-5db8aa86d9ce.png)||||

