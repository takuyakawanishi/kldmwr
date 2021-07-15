# kldmwr
Parameter Estimation Based on KLDMWR Framework


## Introduction

KLDMWR (Kullback-Leibler Divergence 
Method of Weighted Residuals) 
is a framework of parameter estimation 
(Kawanishi, 2020).


In the KLDMWR framework, KLD is transformed to 
integral over [0, 1], and then minimizing KLD can be
seen as a boundary value problem.
Then, the Method of Weighted Residuals, a family of 
method of solving boundary value problems, is applied
to the transformed KLD.


The advantages of KLDMWR is stated in 
Kawanishi (2021).



* Maximum Likelihood Method
* Maximum Product of Spacings Method
* Zero-Boundary Galerkin Estimator (Jiang's version
  of MPS)
* KLDMWR Least Square Method


## Installation

     pip install -i https://test.pypi.org/simple/ kldmwr

## Basic Usage

### Finding a Estimates

We do the following:
* Define the pdf or cdf.
* Set the initial value of parameters p_i.
* kldmwr.univar.find_mle(x, p_i, pdf), etc. returns
[vector of estimates, -log likelihood, success or not,
  detail of the output of the minimizer ]

#### MLE

    import kldmwr.univar
    import scipy.stats
    
    
    def norm_pdf(x, p):
        return scipy.stats.norm.pdf(x, loc=p[0], scale=p[1])


    x = [0.12, -0.62, 1.16, -0.31, -0.02, -0.99, -0.45,  0.17]
    p_i = [0, 1]
    res = kldmwr.univar.find_mle(x, p_i, norm_pdf)
    print(res[0])
    print(res)
    
    # [-0.11747087  0.60645557]
    # (array([-0.11747087,  0.60645557]), 7.350032644753719, True,  final_simplex: (array([[-0.11747087,  0.60645557],
    #        [-0.11748272,  0.60636968],
    #        [-0.11740537,  0.60641523]]), array([7.35003264, 7.35003266, 7.3500327 ]))
    #            fun: 7.350032644753719
    #        message: 'Optimization terminated successfully.'
    #           nfev: 103
    #            nit: 54
    #         status: 0
    #        success: True
    #              x: array([-0.11747087,  0.60645557]))


#### ZGE (or Jiang's modified MPS Estimator, JMMPSE)

    import kldmwr.univar
    import scipy.stats
    
    
    def norm_cdf(x, p):
        return scipy.stats.norm.cdf(x, loc=p[0], scale=p[1])
    
    
    x = [0.12, -0.62, 1.16, -0.31, -0.02, -0.99, -0.45,  0.17]
    p_i = [0, 1]
    res = kldmwr.univar.find_zge(x, p_i, norm_cdf)
    print(res[0])

    # [-0.13663485  0.62631407]


#### NGE (or traditional MPS estimator)

    import kldmwr.univar
    import scipy.stats
    
    
    def norm_cdf(x, p):
        return scipy.stats.norm.cdf(x, loc=p[0], scale=p[1])
    
    
    x = [0.12, -0.62, 1.16, -0.31, -0.02, -0.99, -0.45,  0.17]
    p_i = [0, 1]
    res = kldmwr.univar.find_nge(x, p_i, norm_cdf)
    print(res[0])

    # [-0.11059383  0.80750186]


## Advanced Usage

### Estimation with various initial sets of parameters

#### GZE for the Generalized Extreme Value Distribution (GEVD)

    import kldmwr.univar
    import numpy as np
    import scipy.stats
    
    
    def gev_cdf(y, p):
    
        if p[2] * p[2] < 1e-25:
            return np.exp(- np.exp(- (y - p[0]) / p[1]))
        else:
            return np.exp(- (1. + p[2] * (y - p[0]) / p[1]) ** (- 1.0 / p[2]))
    
    
    x = [-0.29505663, -0.10061241,  0.93107122,  1.25161993,  1.31516917,
         1.39300232,  1.42739514,  1.49512478,  1.49936167, 1.49981574]
    p_0 = np.array([1, 1, -2])
    
    ################################################################################
    # initial values of parameters, p_is
    
    n_gen = 50
    n_int = 20
    mu_pls = np.random.uniform(0, .4, size=n_gen)
    sg_pls = np.random.uniform(-0.5, 0.5, size=n_gen)
    p_0t = p_0.reshape(3, 1)
    p_0s = np.repeat(p_0t, n_gen, axis=1)
    p_gs = p_0s + np.array([mu_pls, sg_pls, np.zeros(n_gen)]) 
    p_gs = p_gs.transpose()
    p_gsels = []
    for p_g in p_gs:
        if 1 + p_g[2] * (x[-1] - p_g[0]) / p_g[1] > 0 and\
            1 + p_g[2] * (x[0] - p_g[0]) / p_g[1] > 0:
            p_gsels.append(p_g)
    
    p_is = np.array(p_gsels)
    p_is = p_is[:n_int]
    
    ################################################################################
    
    res_a = kldmwr.univar.find_min_viv_expl(
        x, p_0, kldmwr.univar.find_zge, gev_cdf,
        p_ints=p_is
    )
    
    print('res_a[0]', res_a[0])
    
    # res_a[0] [ 1.2789668   0.56955288 -2.57856013]


## Notes

### How we handle the tied values





## Reference

Jiang, R. “A Modified MPS Method for Fitting the 3-Parameter Weibull Distribution,” 983–85, 2013. https://doi.org/10.1109/QR2MSE.2013.6625731.

Kawanishi, Takuya. “Maximum Likelihood and the Maximum Product of Spacings from the Viewpoint of the Method of Weighted Residuals.” Computational and Applied Mathematics 39, no. 3 (May 29, 2020). https://doi.org/10.1007/s40314-020-01179-7.
