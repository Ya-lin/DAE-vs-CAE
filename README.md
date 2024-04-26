
# Autoencoder (AE)
An autoencoder learns two functions: a mapping $`g`$ from $`\mathbb{R}^D`$ to $`\mathbb{R}^d`$, called the **encoder**, and another mapping $`h`$ from $\mathbb{R}^d$ to $`\mathbb{R}^D`$, called the **decoder**. The model is trained to minimize
```math
\mathcal{L}=\|h(g(\mathbf{x}))-\mathbf{x}\|_2^2.
```
More generally, we can use
```math
\mathcal{L}=-\ln p(\mathbf{x}|h(g(\mathbf{x}))).
```
Applying AE in any application, we have to restrict the model in some way such that it does not learn the identity function.

### Bottleneck Autoencoder
The simplest approach to restrict the mode is using a narrow bottleneck layer, i.e., $`d\textless D`$.

$\ <$ $`d<.D`$

### Denoising Autoencoder (DAE)
A Denoising Autoencoder is a modification of the autoencoder to prevent the network from learning the identity function. Specifically, if the autoencoder is too big, then it can just learn the data, so the output equals the input, and does not perform any useful representation learning or dimensionality reduction. Denoising autoencoders solves this problem by corrupting the input data on purpose, adding noise, or masking some of the input values. The autoencoders are trained to reconstruct the original images (noise-free images) by taking corrupted or masked images.
Let $`g`$ and $`h`$ be the encoder and decoder network. Then the loss function in DAE is given by
```math
\min_{g, h}\frac{1}{n}\sum_{i=1}^n\left\|h(g(\mathbf{x}_i^{\epsilon}))-\mathbf{x}_i\right\|_2^2
```
where $`\mathbf{x}_i^{\epsilon}`$ is the corrupted version of $`\mathbf{x}_i`$. For image dataset, usually $`\mathbf{x}_i^{\epsilon}`$ is obtained by adding independent Gaussian noise and truncating values into $`[0,1]`$ (assume pixel values are normalized), i.e.,
```math
\mathbf{x}_{i}^{\epsilon}=\textbf{clip}_{to [0,1]}\left(\mathbf{x}_i+\epsilon_i\right).
```
where $`\epsilon_i\sim\mathcal{N}(0,\sigma^2)`$.


**Remark**
* DAE aims to remove noise from corrupted images. So it could be an overcomplete autoencoder, i.e., $`d>D`$.
* Don't forget to clip images after adding noise.
* Masking images is an alternative to corrupting images by Gaussian noise.


### Contractive Autoencoder (CAE)
A different way to regularize autoencoders is by adding the penalty term
```math
\Omega(\mathbf{z},\mathbf{x})=\lambda\left\|\frac{\partial 
g(\mathbf{x})}{\partial\mathbf{x}}\right\|_F^2=\lambda\sum_k\|\nabla
_{\mathbf{x}}g_k(\mathbf{x})\|_2^2
```
to the reconstruction loss, where $`g_{k}(\mathbf{x})`$ is the value of $`k'`$th component of $`g(\mathbf{x})`$. That is, we penalize the Frobenius norm of the encoder's Jacobian.



