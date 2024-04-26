# DAE-vs-CAE

### Denoising Autoencoder (DAE)
A Denoising Autoencoder is a modification of the autoencoder to prevent the network from learning the identity function. Specifically, if the autoencoder is too big, then it can just learn the data, so the output equals the input, and does not perform any useful representation learning or dimensionality reduction. Denoising autoencoders solves this problem by corrupting the input data on purpose, adding noise, or masking some of the input values. The autoencoders are trained to reconstruct the original images (noise-free images) by taking corrupted or masked images.
Let $g$ and $h$ be the encoder and decoder network. Then the loss function in DAE is given by
$$\min_{g, h}\frac{1}{n}\sum_{i=1}^n\|h(g(\mathbf{x}_i^{\epsilon}))-\mathbf{x}_i\|_2^2$$
where $\mathbf{x}_i^{\epsilon}$ is the corrupted version of $\mathbf{x}_i$. For image dataset, ususally $\mathbf{x}_i^{\epsilon}$ is obtained by adding independent Gaussian noise and 
truncating values into $[0,1]$ (assume pixel values are normalized), i.e.,
```math
\mathbf{x}_{i}^{\epsilon}=\textbf{clip}_{to [0,1]}\left(\mathbf{x}_i+\epsilon_i\right).
```
where $`\epsilon_i\sim\mathcal{N}(0,\sigma^2)`$.


**Remark**
* DAE aims to remove noise from corrupted images. So it could be an overcomplete 
autoencoder, i.e., the dimensionality of latent feature $`g(\mathbf{x}_i)`$ is larger 
than that of input data $`\mathbf{x}_i`$.
* Don't forget to clip images after adding noise.
* Masking images is an alternative to corrupting images by Gaussian noise.


### Contractive Autoencoder (CAE)



