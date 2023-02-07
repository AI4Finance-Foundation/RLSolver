# Compressive Sensing using Generative Models
 
 [1] Wu, Yan, Mihaela Rosca, and Timothy Lillicrap. "Deep compressed sensing." International Conference on Machine Learning, 2019.
 
 First case, linear measurment process: $\textbf{y} = \textbf{F} \textbf{x}$, where the true signal $\textbf{x} \in \mathbb{R}^n$, $\textbf{F} \in \mathbb{R}^{m \times n}$, and $\textbf{y} \in \mathbb{R}^m $, $m \ll n$.

## Recovery Error $\lVert x-\hat{x}\rVert_2$ for MNIST

A pretrained model $G_\theta$: $G$ is a neural network with parameter $\theta$.

- Ours: 4.78
- DCS: 3.4

Ours: Formula (7) is trained as a deep neural network.

## Recovery on the MNIST dataset

 $\textbf{F}_\phi$: $\textbf{F}$ is reparameterized as a deep neural network with parameter $\phi$.

|Method|LOSS|Origin image| 1 steps|3 steps | 5 steps|
|-------| ----|------- | -----|------ |-----|
|$\textbf{F}_\phi$ (L) + grad|4.78|![alt_text](./fig/origin.png)|![alt_text](./fig/reconstruction_0.png)|![alt_text](./fig/reconstruction_3.png)|![alt_text](./fig/reconstruction_5.png)|
|$\textbf{F}_\phi$ (L) + NN|10.20|![alt_text](./fig/origin.png)|![alt_text](./fig/reconstruction_0_nn.png)|![alt_text](./fig/reconstruction_3_nn.png)|![alt_text](./fig/reconstruction_5_nn.png)|
|Fix $\textbf{F}$ + grad steps          (m = 100) |6.97|![alt_text](./fig/origin.png)|![alt_text](./fig/reconstruction_0_4_last.png)|![alt_text](./fig/reconstruction_3_4_last.png)|![alt_text](./fig/reconstruction_5_4_last.png)|
|Fix $\textbf{F}$ + grad steps          (m = 300)|4.50|![alt_text](./fig/origin.png)|![alt_text](./fig/reconstruction_0_3_last.png)|![alt_text](./fig/reconstruction_3_3_last.png)|![alt_text](./fig/reconstruction_5_3_last.png)|

|${\overline{X}}$|$\overline{G_\theta (z_0)}$|
|---------|----------------------|
|![alt_text](./fig/origin_average.png)|![alt_text](./fig/recon_average.png)|

## Recovery on the synthetic sparse signal
<!-- ### DCS
|Method|Number of iterations|Origin|Recovery|
|---|----|----|----|
|LASSO|10|![alt_text](./fig/origin_signal_11.png)|![alt_text](./fig/recovery_signal_lasso.png)|
|$G_\theta(z)$|10|![alt_text](./fig/origin_signal_11.png)|![alt_text](./fig/recovery_signal_11.png)|
 -->

### Synthetic Signal
- Sparse signal $\textbf{z}$  $\in \mathbb{B}^{n}$, where $\mathbb{B}$ =  $\lbrace -1,0, 1\rbrace$, $\lVert \textbf{z} \rVert_1 = k$, and sparsity $s = \frac{k}{n}$.
- Representation domain $\boldsymbol{\phi} \in \mathbb{R}^{n\times n}$.
- Sample signal $\textbf{x} = \boldsymbol{\phi} \textbf{z}$.

    $n=100, k=10$
    | $\phi$|$\textbf{z}$|$\textbf{x}$|
    |---|----|----|
    |Identity|fig|fig|
    |DCT|fig|fig|


### Generator $G_\theta(z)$
- Training samples: $\lbrace (\textbf{z},\textbf{x}=\boldsymbol{\phi} \textbf{z})\rbrace$
- Loss function:  $MSE(G_\theta(\textbf{z}), \textbf{x})$
- Test error = $\lVert\boldsymbol{\phi}\boldsymbol{z}-G_\theta(z)\rVert_2$
- $G_\theta$

|$\textbf{z}_{test}$|$\boldsymbol{\phi} \textbf{z}_{test}$|$G_\theta(\textbf{z}_{test})$|Test Error|
|---|----|----|---|
||![alt_text](./fig/origin_signal_supervised.png)|![alt_text](./fig/gen_signal_supervised.png)|$<1e-3$


### Lasso (CS)
- Linear measurment process: $\textbf{y} = \textbf{A} \textbf{x}$ = $\mathbf{A} \boldsymbol{ \phi } \mathbf{z}$ Random measurement $\textbf{A}\in \mathbb{R}^{m\times n}$.
- $\hat{\boldsymbol{z}} = Lasso(y, \mathbf{A} \boldsymbol{ \phi })$
- $\hat{\boldsymbol{z}}_0 \xrightarrow[\sim\text{30 iterations}]{Lasso} \hat{\boldsymbol{z}}$
- Error = $\lVert \textbf{x} - \hat{\textbf{x}} \lVert_2 $, where $\hat{\textbf{x}} = \boldsymbol{ \phi } \hat{\textbf{z}} $.

|Number of iterations to converge| $\frac{m}{n} = 0.3 $ | $\frac{m}{n} = 0.5 $ |  $\frac{m}{n} = 0.7 $|
|-------|------|------|-----|
|n=100||||
|n=1000||||
|n=10000||||

### DCS
- Given train $G_\theta$, using Eqn. (7) in [1] to recover $\boldsymbol{\hat{z}}$ and $\boldsymbol{\hat{x}}$.
- $\hat{\boldsymbol{z}}_0 \xrightarrow[\text{hundreds of iterations}]{Eqn. (7)} \hat{\boldsymbol{z}}$
- Error = $\lVert \textbf{x} - \hat{\textbf{x}} \lVert_2 $, where $\hat{\textbf{x}} = G_\theta( \hat{\textbf{z}}) $.

|Number of iterations to converge| $\frac{m}{n} = 0.3 $ | $\frac{m}{n} = 0.5 $ |  $\frac{m}{n} = 0.7 $|
|-------|------|------|-----|
|n=100||||
|n=1000||||
|n=10000||||

### Ours
- Train a neural network (NN) to replace Eqn. (7) in [1].
- $\hat{\boldsymbol{z}}_0 \xrightarrow[\text{x iterations, x < 10}]{NN} \hat{\boldsymbol{z}}$
- Error = $\lVert \textbf{x} - \hat{\textbf{x}} \lVert_2 $, where $\hat{\textbf{x}} = G_\theta( \hat{\textbf{z}}) $.

|Number of iterations to converge| $\frac{m}{n} = 0.3 $ | $\frac{m}{n} = 0.5 $ |  $\frac{m}{n} = 0.7 $|
|-------|------|------|-----|
|n=100||||
|n=1000||||
|n=10000||||
