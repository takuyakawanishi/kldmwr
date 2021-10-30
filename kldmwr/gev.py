import numpy as np
from kldmwr import distributions


###############################################################################
# Derivatives of h = (1 + xi (x - mu) / sigma)^{-1 / xi}
###############################################################################


def calc_h(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    return phi ** (- 1 / p[2])


def calc_h_mu(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    return 1 / p[1] * phi ** (- 1 / p[2] - 1)


def calc_h_sg(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    return (x - p[0]) / p[1] ** 2 * phi ** (-1 / p[2] - 1)


def calc_h_xi(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    return 1 / p[2] ** 2 * phi ** (-1 / p[2]) * np.log(phi) - \
        (x - p[0]) / p[1] / p[2] * phi ** (-1 / p[2] - 1)


def calc_h_mumu(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    return (1 + p[2]) / p[1] ** 2 * phi ** (-1 / p[2] - 2)


def calc_h_sgsg(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    a = - 2 * (x - p[0]) / p[1] ** 3 * phi ** (-1 / p[2] - 1)
    b = (1 + p[2]) * (x - p[0]) ** 2 / p[1] ** 4 * phi ** (-1 / p[2] - 2)
    return a + b


def calc_h_xixi(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    a = - 2 / p[2] ** 3 * phi ** (-1 / p[2]) * np.log(phi)
    b = 1 / p[2] ** 4 * phi ** (-1 / p[2]) * np.log(phi) * np.log(phi)
    c = - 2 * (x - p[0]) / p[1] / p[2] ** 3 * phi ** (-1 / p[2] - 1) * np.log(
        phi)
    d = 2 * (x - p[0]) / p[1] / p[2] ** 2 * phi ** (-1 / p[2] - 1)
    e = ((x - p[0]) / p[1]) ** 2 * (1 + p[2]) / p[2] ** 2 * phi ** (
                - 1 / p[2] - 2)
    return a + b + c + d + e


def calc_h_musg(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    a = - 1 / p[1] ** 2 * phi ** (-1 / p[2] - 1)
    b = (x - p[0]) / p[1] * (1 + p[2]) / p[1] ** 2 * phi ** (-1 / p[2] - 2)
    return a + b


def calc_h_muxi(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    a = 1 / p[1] / p[2] ** 2 * phi ** (-1 / p[2] - 1) * np.log(phi)
    b = - ((x - p[0]) / p[1]) * (1 + p[2]) / p[1] / p[2] * phi ** (
                -1 / p[2] - 2)
    return a + b


def calc_h_sgxi(x, p):
    phi = 1 + p[2] * (x - p[0]) / p[1]
    a = (x - p[0]) / p[1] ** 2 / p[2] ** 2 * phi ** (-1 / p[2] - 1) * np.log(
        phi)
    b = - ((x - p[0]) / p[1]) ** 2 * (1 + p[2]) / p[1] / p[2] * phi ** (
                -1 / p[2] - 2)
    return a + b


###############################################################################
# Derivatives of G
###############################################################################


def calc_gg(x, p):
    return distributions.gev_cdf(x, p)


def calc_gg_mu(x, p):
    return - calc_gg(x, p) * calc_h_mu(x, p)


def calc_gg_sg(x, p):
    return - calc_gg(x, p) * calc_h_sg(x, p)


def calc_gg_xi(x, p):
    return - calc_gg(x, p) * calc_h_xi(x, p)


def calc_gg_mumu(x, p):
    return calc_gg(x, p) * \
           (- calc_h_mumu(x, p) + calc_h_mu(x, p) * calc_h_mu(x, p))


def calc_gg_sgsg(x, p):
    return calc_gg(x, p) * \
           (- calc_h_sgsg(x, p) + calc_h_sg(x, p) * calc_h_sg(x, p))


def calc_gg_xixi(x, p):
    return calc_gg(x, p) * \
           (- calc_h_xixi(x, p) + calc_h_xi(x, p) * calc_h_xi(x, p))


def calc_gg_musg(x, p):
    return calc_gg(x, p) * \
           (- calc_h_musg(x, p) + calc_h_mu(x, p) * calc_h_sg(x, p))


def calc_gg_muxi(x, p):
    return calc_gg(x, p) * \
           (- calc_h_muxi(x, p) + calc_h_mu(x, p) * calc_h_xi(x, p))


def calc_gg_sgxi(x, p):
    return calc_gg(x, p) * \
           (- calc_h_sgxi(x, p) + calc_h_sg(x, p) * calc_h_xi(x, p))


###############################################################################
# D_{k,i} / D_i,
# D_{Kl, i} / D_i
# (D_{k, i} / D_i)(D_{l, i} / D_i)
# l_{kl, i}
###############################################################################

def calc_eff(xv, p):
    return np.append([0], np.append(calc_gg(xv, p), [1]))


def calc_eff_p(xv, p):
    eff_mu = np.append([0], np.append(calc_gg_mu(xv, p), [0]))
    eff_sg = np.append([0], np.append(calc_gg_sg(xv, p), [0]))
    eff_xi = np.append([0], np.append(calc_gg_xi(xv, p), [0]))
    eff_p = np.concatenate([(eff_mu, eff_sg, eff_xi)], axis=0)
    return eff_p


def calc_gg_pp(x, p):
    a = np.array(
        [[calc_gg_mumu(x, p), calc_gg_musg(x, p), calc_gg_muxi(x, p)],
         [calc_gg_musg(x, p), calc_gg_sgsg(x, p), calc_gg_sgxi(x, p)],
         [calc_gg_muxi(x, p), calc_gg_sgxi(x, p), calc_gg_xixi(x, p)]])
    return a


def calc_eff_pp(xv, p):
    z331 = np.zeros((3, 3, 1))
    return np.append(z331, np.append(calc_gg_pp(xv, p), z331, axis=-1),
                     axis=-1)


def calc_dd_p_o_dd(xv, p):
    """Returns D_p,i / D_i
    res[k, i] = D_{k, i} / D_i
    """
    eff_p = calc_eff_p(xv, p)
    dgg_p = np.diff(eff_p, axis=-1)
    dff = np.diff(calc_eff(xv, p))
    dff_arr = np.concatenate([(dff, dff, dff)], axis=0)
    return np.divide(dgg_p, dff_arr)


def calc_dd_pp_o_dd(xv, p):
    """Returns D_pp, i / D_i
    res[k, l, i] = D_{kl, i} / D_i
    """
    eff_pp = calc_eff_pp(xv, p)
    dif_a = np.diff(eff_pp, axis=-1)
    eff = calc_eff(xv, p)
    dff = np.diff(eff)
    dff_arr = np.array([[dff, dff, dff], [dff, dff, dff], [dff, dff, dff]])
    return np.divide(dif_a, dff_arr)


def calc_dd_p_o_dd_sq(xv, p):
    """Returns $(D_{k, i} / D_i)(D_{l, i} / D_i)$ as res[k, l, i]
    """
    dd_p_o_dd = calc_dd_p_o_dd(xv, p)
    return np.einsum('ik,jk->ijk', dd_p_o_dd, dd_p_o_dd)


def calc_l_pp_is(xv, p):
    """Returns $l_{kl, i}$
    """
    return calc_dd_pp_o_dd(xv, p) - calc_dd_p_o_dd_sq(xv, p)


def calc_l_pp(xv, p):
    return calc_l_pp_is(xv, p).sum(axis=-1)


def calc_l_pp_glz(xv, p):
    weight = np.ones(len(xv) + 1)
    weight[0] = weight[0] * 0.5
    weight[-1] = weight[-1] * 0.5
    lpp_is = calc_l_pp_is(xv, p)
    return np.dot(lpp_is, weight)


#
# Auxiliary
#
def calc_dd_p_o_dd_sq_lp(xv, p):
    """Subprogram for verifying that calc_dd_p_o_dd_sq returns
    intended values.
    """
    ys = []
    for i in range(len(xv) + 1):
        dd_p_o_dd_i = calc_dd_p_o_dd(xv, p)[:, i]
        print(dd_p_o_dd_i)
        ys.append(np.einsum('i,j->ij', dd_p_o_dd_i, dd_p_o_dd_i))
    ys = np.array(ys)
    ys = np.transpose(ys, axes=[1, 2, 0])
    return ys



def gev_cdf(x, p):
    return distributions.gev_cdf(x, p)


def gev_pdf(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    return 1 / p[1] * u ** (-1 / p[2] - 1) * distributions.gev_cdf(x, p)


def gev_pdf_mu(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    return (
        (p[2] + 1) / p[1] ** 2 * u ** (-1 / p[2] - 2)
        - 1 / p[1] ** 2 * u ** (-2 / p[2] - 2)
           ) * gev_cdf(x, p)


def gev_pdf_sg(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    return (
        - 1 / p[1] ** 2 * u ** (-1 / p[2] - 1)
        + (1 + p[2]) / p[1] ** 2 * (x - p[0]) / p[1] * u**(-1 / p[2] - 2)
        - 1 / p[1] ** 2 * (x - p[0]) / p[1] * u ** (-2 / p[2] - 2)
           ) * gev_cdf(x, p)


def gev_pdf_xi(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    return (
                   1 / p[1] / p[2] ** 2 * np.log(u) * \
                   (u ** (-1 / p[2] - 1) - u ** (-2 / p[2] - 1))
                   + 1 / p[1] / p[2] * (x - p[0]) / p[1] * u ** (-2 / p[2] - 2)
                   + 1 / p[1] * (- 1 / p[2] - 1) * (x - p[0]) / p[1] * u ** (
                               -1 / p[2] - 2)
           ) * gev_cdf(x, p)


################################################################################
# Derivatives of Log Likilihood
################################################################################

def gev_l(x, p):
    return np.log(gev_pdf(x, p))


def gev_l_mu(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    return (p[2] + 1) / p[1] * u ** (-1) - 1 / p[1] * u ** (-1 / p[2] - 1)


def gev_l_sg(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = - 1 / p[1]
    b = (1 + p[2]) / p[1] * (x - p[0]) / p[1] * \
        u ** (-1)
    c = - 1 / p[1] * (x - p[0]) / p[1] * u ** (-1 / p[2] - 1)
    return a + b + c


def gev_l_xi(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = 1 / p[2] ** 2 * np.log(u) * (1 - u ** (- 1 / p[2]))
    b = 1 / p[2] * (x - p[0]) / p[1] * u ** (- 1 / p[2] - 1)
    c = (-1 / p[2] - 1) * (x - p[0]) / p[1] * u ** (- 1)
    return a + b + c


def gev_l_mumu(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = p[2] * (1 + p[2]) / p[1] ** 2 * u ** (-2)
    b = - (1 + p[2]) / p[1] ** 2 * u ** (-1 / p[2] - 2)
    return a + b


def gev_l_musg(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = - (1 + p[2]) / p[1] ** 2 * u ** (-1)
    b = p[2] * (1 + p[2]) / p[1] ** 2 * y * u ** (-2)
    c = 1 / p[1] ** 2 * u ** (-1 / p[2] - 1)
    d = - (1 + p[2]) / p[1] ** 2 * y * u ** (-1 / p[2] - 2)
    return a + b + c + d


def gev_l_muxi(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = 1 / p[1] * u ** (-1)
    b = - (1 + p[2]) / p[1] * y * u ** (-2)
    c = - 1 / p[1] / p[2] ** 2 * np.log(u) * u ** (-1 / p[2] - 1)
    d = (1 + p[2]) / p[1] / p[2] * y * u ** (-1 / p[2] - 2)
    return a + b + c + d


def gev_l_sgsg(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = 1 / p[1] ** 2
    b = - 2 * (1 + p[2]) / p[1] ** 2 * y * u ** (-1)
    c = p[2] * (1 + p[2]) / p[1] ** 2 * y ** 2 * u ** (-2)
    d = 2 / p[1] ** 2 * y * u ** (-1 / p[2] - 1)
    e = - (1 + p[2]) / p[1] ** 2 * y ** 2 * u ** (-1 / p[2] - 2)
    return a + b + c + d + e


def gev_l_sgxi(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = 1 / p[1] * y * u ** (-1)
    b = - (1 + p[2]) / p[1] * y ** 2 * u ** (-2)
    c = - 1 / p[1] / p[2] ** 2 * y * np.log(1 + p[2] * y) * u ** (-1 / p[2] - 1)
    d = (1 + p[2]) / p[2] / p[1] * y ** 2 * u ** (-1 / p[2] - 2)
    return a + b + c + d


def gev_l_xixi(x, p):
    y = (x - p[0]) / p[1]
    u = 1 + p[2] * y
    a = - 2 / p[2] ** 3 * np.log(u) * (1 - u ** (-1 / p[2]))
    b = 1 / p[2] ** 2 * y / u * (1 - u ** (-1 / p[2]))
    c = - 1 / p[2] ** 4 * np.log(u) * np.log(u) * u ** (-1 / p[2])
    d = 1 / p[2] ** 3 * y * np.log(u) * u ** (-1 / p[2] - 1)
    e = - 1 / p[2] ** 2 * y * u ** (-1 / p[2] - 1)
    f = d
    g = - (1 + p[2]) / p[2] ** 2 * y ** 2 * u ** (-1 / p[2] - 2)
    h = 1 / p[2] ** 2 * y / u
    i = (1 + p[2]) / p[2] * y ** 2 * u ** (-2)
    return a + b + c + d + e + f + g + h + i
