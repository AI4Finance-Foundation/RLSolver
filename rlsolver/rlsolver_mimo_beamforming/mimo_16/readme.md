Train by running the following command:

```
python train.py reward_mode N
```

Number of antennas and users N = 9, 10, 11, 12, 13, 14, 15

reward_mode:

- 3 curriculum learning

- 4 supervised learning + reinforcement learning

|Method| Optimal|MMSE | REINFORCE|Optimal| MMSE | REINFORCE |
| ----------- | -----|------ |-----|------ |-----|------ |
| SNR | SNR=10 dB| SNR=10 dB |SNR=10 dB  |SNR=20 dB|SNR=20 dB| SNR=20 dB|
| N=4 | 9.8|8.36 | 9.8| 19.42 |17.03 | 19.1|
| N=8 | 18.19|15.45 | 17.95|34.5| 31.13 |34.0|
| N=9 | |17.79 | || 34.49 ||
| N=10 |22.89 |19.63 |19.9||37.82|32.921|
| N=11 | |21.56 |||40.68||
| N=12 | 27.10 | 23.49|23.03||44.64|37.2|
| N=13 | |25.32 |||47.39||
| N=14 | 30.2|27.37 |||51.17||
| N=15 | |29.20 |||55.03||
| N=16 |35.5 |31.09 |30.295||57.73|48.742|
