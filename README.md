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


Code:
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


#### ZGE (Zero-boundary Galerkin Estimator), or JMMPSE (Jiang's modified MPS Estimator)

    import kldmwr.univar
    import scipy.stats
    
    
    def norm_cdf(x, p):
        return scipy.stats.norm.cdf(x, loc=p[0], scale=p[1])
    
    
    x = [0.12, -0.62, 1.16, -0.31, -0.02, -0.99, -0.45,  0.17]
    p_i = [0, 1]
    res = kldmwr.univar.find_glz(x, p_i, norm_cdf)
    print(res[0])

    # [-0.13663485  0.62631407]


#### MPSE or NGE (Nonzero-boundary Galerkin Estimator)

    import kldmwr.univar
    import scipy.stats
    
    
    def norm_cdf(x, p):
        return scipy.stats.norm.cdf(x, loc=p[0], scale=p[1])
    
    
    x = [0.12, -0.62, 1.16, -0.31, -0.02, -0.99, -0.45,  0.17]
    p_i = [0, 1]
    res = kldmwr.univar.find_gln(x, p_i, norm_cdf)
    print(res[0])

    # [-0.11059383  0.80750186]


### Advanced Usage

