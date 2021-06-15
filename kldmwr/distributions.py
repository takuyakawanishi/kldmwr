import numpy as np
import scipy.stats
import scipy.special


def bta_cdf(x, p):
    return scipy.stats.beta.cdf(x, p[0], p[1])


def bta_pdf(x, p):
    return scipy.stats.beta.pdf(x, p[0], p[1])


def bta_sampling(n_rep, n_size, p):
    return scipy.stats.beta.rvs(p[0], p[1], size=(n_rep, n_size))


def chy_loc_scl_cdf(x, p):
    return scipy.stats.cauchy.cdf(x, p[0], p[1])


def chy_loc_scl_pdf(x, p):
    return scipy.stats.cauchy.pdf(x, p[0], p[1])


def chy_loc_scl_sampling(n_rep, n_size, p):
    return scipy.stats.cauchy.rvs(p[0], p[1], size=(n_rep, n_size))


def chy_scl_cdf(x, p):
    return scipy.stats.cauchy.cdf(x, 0, p[0])


def chy_scl_pdf(x, p):
    return scipy.stats.cauchy.pdf(x, 0, p[0])


def chy_scl_sampling(n_rep, n_size, p):
    return scipy.stats.cauchy.rvs(0, p[0], size=(n_rep, n_size))


def exp_cdf(x, p):
    return scipy.stats.expon.cdf(x, loc=0, scale=1 / p[0])


def exp_pdf(x, p):
    return scipy.stats.expon.pdf(x, loc=0, scale=1 / p[0])


def exp_sampling(n_rep, n_size, p):
    return scipy.stats.expon.rvs(scale=1 / p[0], size=(n_rep, n_size))


def exp_scl_cdf(x, p):
    return scipy.stats.expon.cdf(x, loc=0, scale=p[0])


def exp_scl_pdf(x, p):
    return scipy.stats.expon.pdf(x, loc=0, scale=p[0])


def exp_scl_sampling(n_rep, n_size, p):
    return scipy.stats.expon.rvs(scale=p[0], size=(n_rep, n_size))


def gev_cdf_old(x, p):
    # changed Nov. 5, 2020.
    y = np.array(x)
    return np.exp(- (1. + p[2] * (y - p[0]) / p[1])**(- 1.0 / p[2]))


def gev_cdf(y, p):
    # changed Nov. 5, 2020.
    if p[2] * p[2] < 1e-25:
        return np.exp(- np.exp(- (y - p[0]) / p[1]))
    else:
        return np.exp(- (1. + p[2] * (y - p[0]) / p[1]) ** (- 1.0 / p[2]))


def gev_pdf_old(x, p):
    # changed Nov. 5, 2020.
    x = np.array(x)
    return (1.0 / p[1] * np.exp(- (1.0 + p[2] * (x - p[0]) / p[1]) **
            (- 1.0 / p[2])) * (1.0 + p[2] * (x - p[0]) / p[1]) **
            (-1.0 - 1.0 / p[2]))


def gev_pdf(x, p):
    # changed Nov. 5, 2020.
    if p[2] * p[2] < 1e-25:
        return 1.0 / p[1] * np.exp(- (x - p[0]) / p[1]) * \
               np.exp(- np.exp(- (x - p[0]) / p[1]))
    else:
        return 1.0 / p[1] * np.exp(- (1.0 + p[2] *
            (x - p[0]) / p[1]) ** (- 1.0 / p[2])) \
            * (1.0 + p[2] * (x - p[0]) / p[1]) ** (-1.0 - 1.0 / p[2])


def gev_inv(q, p):
    return p[0] + p[1] / p[2] * ((- np.log(q))**(- p[2]) - 1.)


def gev_sampling(n_rep, n_size, p):
    y = np.random.uniform(size=(n_rep, n_size))
    return p[0] + p[1] / p[2] * ((- np.log(y))**(- p[2]) - 1.)


def gev_ipf(x, p):
    res = 1 / (p[1] + p[2] * (x[0] - p[0])) + \
          1 / (p[1] + p[2] * (x[-1] - p[0]))
    return res


def gmm_scl_shp_cdf(x, p):
    return scipy.stats.gamma.cdf(x, p[1], loc=0, scale=p[0])


def gmm_scl_shp_pdf(x, p):
    return scipy.stats.gamma.pdf(x, p[1], loc=0, scale=p[0])


def gmm_scl_shp_sampling(n_rep, n_size, p):
    return scipy.stats.gamma.rvs(
        p[1], loc=0, scale=p[0], size=(n_rep, n_size))


def gmm_shp_cdf(x, p):
    return scipy.stats.gamma.cdf(x, p[0], loc=0, scale=1)


def gmm_shp_pdf(x, p):
    return scipy.stats.gamma.pdf(x, p[0], loc=0, scale=1)


def gmm_shp_sampling(n_rep, n_size, p):
    return scipy.stats.gamma.rvs(p[0], size=(n_rep, n_size))


def lgs_loc_scl_cdf(x, p):
    return scipy.stats.logistic.cdf(x, p[0], p[1])


def lgs_loc_scl_pdf(x, p):
    return scipy.stats.logistic.pdf(x, p[0], p[1])


def lgs_loc_scl_sampling(n_rep, n_size, p):
    return scipy.stats.logistic.rvs(p[0], p[1], size=(n_rep, n_size))


def lgs_scl_cdf(x, p):
    return scipy.stats.logistic.cdf(x, 0, p[0])


def lgs_scl_pdf(x, p):
    return scipy.stats.logistic.pdf(x, 0, p[0])


def lgs_scl_sampling(n_rep, n_size, p):
    return scipy.stats.logistic.rvs(0, p[0], size=(n_rep, n_size))


def nrm_loc_var_cdf(x, p):
    return scipy.stats.norm.cdf(x, loc=p[0], scale=p[1]**.5)


def nrm_loc_var_pdf(x, p):
    return scipy.stats.norm.pdf(x, loc=p[0], scale=p[1]**.5)


def nrm_loc_var_sampling(n_rep, n_size, p):
    return scipy.stats.norm.rvs(loc=p[0], scale=p[1]**.5, size=(n_rep, n_size))


def nrm_var_cdf(x, p):
    return scipy.stats.norm.cdf(x, loc=0, scale=p[0]**.5)


def nrm_var_pdf(x, p):
    return scipy.stats.norm.pdf(x, loc=0, scale=p[0]**.5)


def nrm_var_sampling(n_rep, n_size, p):
    return scipy.stats.norm.rvs(
        loc=0, scale=p[0]**.5, size=(n_rep, n_size))


def prt_scl_shp_cdf(x, p):
    return scipy.stats.pareto.cdf(x, p[1], loc=0, scale=p[0])


def prt_scl_shp_pdf(x, p):
    return scipy.stats.pareto.pdf(x, p[1], loc=0, scale=p[0])


def prt_scl_shp_sampling(n_rep, n_size, p):
    return scipy.stats.pareto.rvs(p[1], scale=p[0], size=(n_rep, n_size))


def prt_shp_cdf(x, p):
    return scipy.stats.pareto.cdf(x, p[0], loc=0, scale=1)


def prt_shp_pdf(x, p):
    return scipy.stats.pareto.pdf(x, p[0], loc=0, scale=1)


def prt_shp_sampling(n_rep, n_size, p):
    return scipy.stats.pareto.rvs(p[0], size=(n_rep, n_size))


def tpg_cdf(x, p):
    return scipy.stats.gamma.cdf(x, p[2], loc=p[0], scale=1 / p[1])


def tpg_pdf(x, p):
    return scipy.stats.gamma.pdf(x, p[2], loc=p[0], scale=1 / p[1])


def tpg_sampling(n_rep, n_size, p):
    return scipy.stats.gamma.rvs(
        p[2], loc=p[0], scale=1 / p[1], size=(n_rep, n_size))


def tpg_scl_cdf(x, p):
    return scipy.stats.gamma.cdf(x, p[2], loc=p[0], scale=p[1])



def tpw_cdf(x, p):
    return 1.0 - np.exp(- ((x - p[0]) / p[1])**p[2])


def tpw_pdf(x, p):
    return p[2] / p[1] * ((x - p[0]) / p[1])**(p[2] - 1.) * \
           np.exp(- ((x - p[0]) / p[1])**p[2])


def tpw_inv(y, p):
    return p[1] * (- np.log(1. - y))**(1. / p[2]) + p[0]


def tpw_sampling(n_rep, n_size, p_0):
    p_0 = np.asarray(p_0)
    rand = np.random.uniform(size=(n_rep, n_size))
    return tpw_inv(rand, p_0)


def unf_cdf(x, p):
    return scipy.stats.uniform.cdf(x, loc=p[0], scale=p[1])


def unf_pdf(x, p):
    return scipy.stats.uniform.pdf(x, loc=p[0], scale=p[1])


def unf_sampling(n_rep, n_size, p):
    return scipy.stats.uniform.rvs(loc=p[0], scale=p[1], size=(n_rep, n_size))


def unf_upp_cdf(x, p):
    return scipy.stats.uniform.cdf(x, 0, scale=p[0])


def unf_upp_pdf(x, p):
    return scipy.stats.uniform.pdf(x, 0, scale=p[0])


def unf_upp_sampling(n_rep, n_size, p):
    return scipy.stats.uniform.rvs(0, scale=p[0], size=(n_rep, n_size))


def wbl_scl_shp_cdf(x, p):
    return scipy.stats.weibull_min.cdf(x, p[1], scale=p[0])


def wbl_scl_shp_pdf(x, p):
    return scipy.stats.weibull_min.pdf(x, p[1], scale=p[0])


def wbl_scl_shp_sampling(n_rep, n_size, p):
    return scipy.stats.weibull_min.rvs(
        p[1], scale=p[0], size=(n_rep, n_size))


def wbl_shp_cdf(x, p):
    return scipy.stats.weibull_min.cdf(x, p[0])


def wbl_shp_pdf(x, p):
    return scipy.stats.weibull_min.pdf(x, p[0])


def wbl_shp_sampling(n_rep, n_size, p):
    return scipy.stats.weibull_min.rvs(p[0], size=(n_rep, n_size))


################################################################################
# TPW
################################################################################

def calc_h_tpw(x, p):
    return ((x - p[0]) / p[1])**p[2]


def calc_h_tpw_mu(x, p):
    return - p[2] / p[1] * ((x - p[0]) / p[1])**(p[2] - 1)


def calc_h_tpw_sg(x, p):
    return - p[2] / p[1] * ((x - p[0]) / p[1])**(p[2] - 1) * ((x - p[0]) / p[1])


def calc_h_tpw_xi(x, p):
    return np.log((x - p[0]) / p[1]) * ((x - p[0]) / p[1])**p[2]


def calc_h_tpw_mumu(x, p):
    return p[2] * (p[2] - 1) / p[1]**2 * ((x - p[0]) / p[1])**(p[2] - 2)


def calc_h_tpw_musg(x, p):
    return p[2] * (p[2] - 1) / p[1]**2 * ((x - p[0]) / p[1])**(p[2] - 2) * \
        ((x - p[0]) / p[1]) + \
        p[2] / p[1]**2 * ((x - p[0]) / p[1])**(p[2] - 1)

def calc_h_tpw_muxi(x, p):
    return - p[2] / p[1] * np.log((x - p[0]) / p[1]) * ((x - p[0]) / p[1])**(p[2] - 1) - \
        1 / p[1] * ((x - p[0]) / p[1])**(p[2] - 1)



def calc_h_tpw_sgsg(x, p):
    return 2 * p[2] / p[1]**3 * (x - p[0]) * ((x - p[0]) / p[1])**(p[2] - 1) + \
        p[2] * (p[2] - 1) / p[1]** 2 * ((x - p[0]) / p[1])**2 * \
        ((x - p[0]) / p[1])**(p[2] - 2)


def calc_h_tpw_sgxi(x, p):
    return - (x - p[0]) / p[1]**2 * ((x - p[0]) / p[1])**(p[2] - 1) - \
        p[2] * (x - p[0]) / p[1]**2 * np.log((x - p[0]) / p[1]) * \
        ((x - p[0]) / p[1])**(p[2] - 1)


def calc_h_tpw_xixi(x, p):
    return (np.log((x - p[0]) / p[1]))**2 * ((x - p[0]) / p[1])**p[2]


def tpw_cdf_mu(x, p):
    return calc_h_tpw_mu(x, p) * np.exp(- calc_h_tpw(x, p))


def tpw_cdf_sg(x, p):
    return calc_h_tpw_sg(x, p) * np.exp(- calc_h_tpw(x, p))


def tpw_cdf_xi(x, p):
    return calc_h_tpw_xi(x, p) * np.exp(- calc_h_tpw(x, p))


def tpw_cdf_mumu(x, p):
    return (calc_h_tpw_mumu(x, p) - calc_h_tpw_mu(x, p)**2) * \
        np.exp(- calc_h_tpw(x, p))


def tpw_cdf_musg(x, p):
    return (calc_h_tpw_musg(x, p) - calc_h_tpw_mu(x, p) *
        calc_h_tpw_sg(x, p)) * np.exp(- calc_h_tpw(x, p))


def tpw_cdf_muxi(x, p):
    return (calc_h_tpw_muxi(x, p) - calc_h_tpw_mu(x, p) *
        calc_h_tpw_xi(x, p)) * np.exp(- calc_h_tpw(x, p))


def tpw_cdf_sgsg(x, p):
    return (calc_h_tpw_sgsg(x, p) - calc_h_tpw_sg(x, p) *
        calc_h_tpw_sg(x, p)) * np.exp(- calc_h_tpw(x, p))


def tpw_cdf_sgxi(x, p):
    return (calc_h_tpw_sgxi(x, p) - calc_h_tpw_sg(x, p) *
        calc_h_tpw_xi(x, p)) * np.exp(- calc_h_tpw(x, p))


def tpw_cdf_xixi(x, p):
    return (calc_h_tpw_xixi(x, p) - calc_h_tpw_xi(x, p) *
        calc_h_tpw_xi(x, p)) * np.exp(- calc_h_tpw(x, p))


def main():
    pass


if __name__ == '__main__':
    main()