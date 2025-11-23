# PROJECT IM01 - Telecom Paris

In this project, we tried to implement an old method to estimate motion blur kernel and then deblur images with Split-Bregman method for TV-Deconvolution.

Code explanations : 
- kernel_estimation : our implementation of the kernel estimation method described in the article article_lr.pdf. However the code is not functional at the moment… 
- kernel_estimation_bis : implementation delivered by professor based on article_lr. However, instead of applying whitening -> projections -> 1D autocorrelations, it applies directly the 2D autocorrelation on vx, vy and then projects. Not sure it is equivalent.
- tv_deconv : implements the split-bregman method for TV regularization minimization problem with symmetric boundary conditions.
- tvdeconv : implements the split-bregman method for TV regularization minimization problem with periodic boundary conditions.
- utils : contains different useful function to load and process images.
- test : contains different test functions to evaluate different aspects (robustness to noise, to kernel size, performance on real images, …).
- main : containing the principal code to apply automatically the whole method.

## TO DO : 
- finishing debug of kernel_estimation.
- implementing hyperlaplacian regularization fo deconvolution.
- improving power law decay model and pre-denoising before computing autocorrelation.
- implementing a better compensation algorithm.

