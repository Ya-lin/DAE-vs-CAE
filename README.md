# DAE-vs-CAE

### Denoising Autoencoder (DAE)
A Denoising Autoencoder is a modification of the autoencoder to prevent the network from learning the identity function. 
Specifically, if the autoencoder is too big, then it can just learn the data, so the output equals the input, and does not 
perform any useful representation learning or dimensionality reduction. Denoising autoencoders solves this problem by corrupting 
the input data on purpose, adding noise, or masking some of the input values. The autoencoders are trained to reconstruct the
original images (noise-free images).


