# Classical simulation of quantum circuits using tensor networks

Roadmap
- [x] Tensor network representations
- [ ] Gym-environments
- [ ] Massively parallel simulation of environment
- [ ] Baseline methods: greedy, BRB
- [ ] Dataset
- [ ] RL achieves optimality and outperforms baselines

## Data Generation

![image](https://user-images.githubusercontent.com/75991833/218318283-33c5ee67-4523-42e2-bda9-71f63b9ab34a.png)

## Experimental Results

|Method|N=10|N=30|N=50|N=100|
|-------| ----|------- | -----|------ |
|![TT](https://user-images.githubusercontent.com/75991833/217780619-40f42213-62b8-4db5-bfa9-0c9f8d97081d.png)|2496<br>2496<br>GAP：0%|2147751200.00<br>2147751690.24<br>Gap: 0.0000002283%|2251800000000000<br>2251800157282631.5<br>Gap: 0.0000000698%|2.5353012e+30<br>2.535301200456459e+30<br>Gap: 0.0000000180%|
|![TR](https://user-images.githubusercontent.com/75991833/217780649-80acaa33-030b-46b9-9fc6-bf5bc84167a3.png)||||
|![TTN](https://user-images.githubusercontent.com/75991833/217782955-cd2cd6e8-d0b8-4187-b7e7-d202266bcbfb.png)||||
|![G](https://user-images.githubusercontent.com/75991833/217780858-eff2a41e-3847-4ed2-bbcb-5db8aa86d9ce.png)||||

