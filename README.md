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

Code:
    Finding MLE

    import kldmwr


    def norm_pdf(x, p):
        return scipy.stats.norm.pdf(x, loc=p[0], scale=p[1])


    p_i = [0, 1]
    res = kldmwr.find_mle(x, p_i, norm_pdf)
    print(res[0])
    


* Define the pdf or cdf.
* Set the initial value of parameters p_i.
* Of course p_i has the same dimension as the
estimating parameters.
* kldmwr.find_mle(x, p_i, pdf) returns
[vector of estimates, -log likelihood, success or not,
  detail of the output of the minimizer ]


Code:
    Finding MPS
    
    def norm_cdf(x, p):
        return scipy.stats.norm.cdf(x, log=p[0], scale=p[1])

    
    p_i = [0, 1]
    res = kldmwr.find_gln(x, p_i, norm_cdf)
    print(res[0])


Code:
    Finding Zero-boundary Galerkin Estimator


    p_i = [0, 1]
    res = kldmwr.find_glz(x, p_i, norm_cdf)
    print(res[0])


## Advanced Usage

You can apply 
>>>>>>> develop
