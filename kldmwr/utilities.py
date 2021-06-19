from ad import gh
from ad.admath import *
import numpy as np
import sys
# sys.path.append('../../../../../packages/kldmwr/kldmwr')
from kldmwr import univar
from kldmwr import utilities
from distributions import *


def gev_cdf_nnp(y, p):
    if p[2] * p[2] < 1e-25:
        return exp(- exp(- (y - p[0]) / p[1]))
    else:
        return exp(- (1. + p[2] * (y - p[0]) / p[1]) ** (- 1.0 / p[2]))


def gev_pdf_nnp(x, p):
    if p[2] * p[2] < 1e-25:
        return 1.0 / p[1] * exp(- (x - p[0]) / p[1]) * \
               exp(- exp(- (x - p[0]) / p[1]))
    else:
        return 1.0 / p[1] * \
               exp(- (1.0 + p[2] * (x - p[0]) / p[1]) ** (- 1.0 / p[2])) * \
               (1.0 + p[2] * (x - p[0]) / p[1]) ** (-1.0 - 1.0 / p[2])


def gev_log_product_of_spacings_nnp(x_unq, wgt, p):
    ffs = []
    for u in x_unq:
        ffs.append(gev_cdf_nnp(u, p))
    effs = [0] + ffs + [1]
    spcs = []
    for i in range(len(x_unq) + 1):
        spcs.append(effs[i + 1] - effs[i])
    lspcs = []
    for spc in spcs:
        if spc <= 0:
            print(effs, x_unq)
        lspcs.append(log(spc))
    s = 0
    for i in range(len(lspcs)):
        s += lspcs[i] * wgt[i]
    return s


def gev_log_likelihood_nnp(x_unq, wgt, p):
    fs = [gev_pdf_nnp(x_i, p) for x_i in x_unq]
    lls = [log(f) for f in fs]
    s = 0
    for i in range(len(lls)):
        s += lls[i] * wgt[i]
    return s


class GradHessAd(object):
    """
    Require ad, np
    """

    def __init__(self, unq, wgt, func):
        self.unq = unq
        self.wgt = wgt
        self.func = func

    def func_of_only_p(self, p):
        return self.func(self.unq, self.wgt, p)

    def ghs(self, p):
        g, h = gh(self.func_of_only_p)  # gh from ad
        # gs = g(p)
        hs = h(p)
        return hs

    def observed_information(self, p, i):
        h = self.ghs(p)
        h = np.array(h)
        return - h[i, i]


def main():
    x = [
        0.17848831, 0.43254182, 0.50380753, 0.50738053, 0.57967216,
        0.58942647, 0.75051722, 0.96180827, 1.12087741, 1.31818877,
        1.41784379, 1.55049636, 1.57126291, 1.67930892, 1.90272559,
        2.23979655, 2.86849759, 2.99667004, 4.72691153, 4.86307725
    ]
    cdf = gev_cdf  # from kldmwr/distributions
    pdf = gev_pdf  # from kldmwr/distributions
    p_i = [1, 1, 1]
    res = univar.find_min_viv(
            x, p_i, univar.find_glz, cdf, ipf=gev_ipf
        )
    success = res[2]
    print('ZMPSE succeeded? ',  success)

    # sys.exit()
    p_hat_z = list(res[0])
    x_unq, cnt = np.unique(x, return_counts=True)
    wgt = univar.weights_zbc(cnt)

    print('x_unq, wgt, pdf, cdf = ',   x_unq, wgt, pdf, cdf)
    print('========== Results for ZMPS')
    print('zmpse = ',  p_hat_z)
    lps_nnp = gev_log_product_of_spacings_nnp(x_unq, wgt, p_hat_z)
    print('lps by gev_log_product_of_spacings_nnp        = ', lps_nnp)
    lps_univar = univar.log_product_of_spacings_wgt_zbc(x, p_hat_z, cdf)
    print('lps by kldmwr.log_product_of_spacings_wgt_zbc = ', lps_univar)

    dat = GradHessAd(x_unq, wgt, func=gev_log_product_of_spacings_nnp)
    h = dat.ghs(p_hat_z)
    oi_xi = dat.observed_information(p_hat_z, 2)

    # print('1st derivatives of lps =',  g)
    print('2nd derivatives of lps =',  np.array(h))
    print('observed information by ZMPS for xi =',  oi_xi)

    print('========== Results for MLE')
    wgt = np.copy(cnt)
    res = univar.find_min_viv(
            x, p_i, univar.find_mle, pdf, ipf=gev_ipf
        )
    success = res[2]
    print('MLE succeeded? ',  success)
    p_hat_l = res[0]

    ll_nnp = gev_log_likelihood_nnp(x_unq, wgt, p_hat_l)
    print('ll by gev_log_likelihood_nnp =', ll_nnp)
    ll_univar = univar.log_likelihood(x, p_hat_l, pdf)
    print('ll by kldmwr.log_likelihood  =', ll_univar)
    dat = GradHessAd(x, wgt, func=gev_log_likelihood_nnp)
    h_l = dat.ghs(p_hat_l)
    oi_xi_l = dat.observed_information(p_hat_l, 2)
    # print('1st derivatives of ll =',  g_l)
    print('2nd derivatives of ll =',  h_l)
    print('observed information by ML for xi =',  oi_xi_l)


if __name__ == '__main__':
    main()
