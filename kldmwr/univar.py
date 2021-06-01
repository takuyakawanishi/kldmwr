"""Various methods of parameter estimation for univariate distributions.

This module provides some parameter estimation methods based on
KLDMWR framework, for uni-variate probability distributions.

"""
import numpy as np
import pandas as pd
import scipy.optimize


def weights_zbc(counts):
    wz = np.zeros(len(counts) + 1)
    wz[: -1] = counts
    wz[1:] += counts
    return 0.5 * wz


def weights_nbc(counts):
    wn = np.zeros(len(counts) + 1)
    wn[: -1] = counts
    wn[-1] = 1
    return wn


def calc_ls(p, x_unq, cdf, wgt, vtxvals):
    vtxvals[1: -1] = cdf(x_unq, p)
    sps = np.diff(vtxvals)
    lss = np.log(sps) * np.log(sps)
    return np.dot(lss, wgt)


def calc_gl(p, x_unq, cdf, wgt, vtxvals):
    vtxvals[1: -1] = cdf(x_unq, p)
    sps = np.diff(vtxvals)
    ls = np.log(sps)
    return - np.dot(ls, wgt)


def calc_gl_ipf(p, x_unq, cdf, wgt, vtxvals, ipf, k_ipf):
    vtxvals[1: -1] = cdf(x_unq, p)
    sps = np.diff(vtxvals)
    ls = np.log(sps)
    return - np.dot(ls, wgt) + ipf(x_unq, p) * k_ipf


def calc_col(p, x_unq, pdf, wgt, vtxvals):
    _ = vtxvals
    vxs = pdf(x_unq, p)
    ll = np.log(vxs)
    return - np.dot(ll, wgt)


def calc_col_ipf(p, x_unq, pdf, wgt, vtxvals, ipf, k_ipf):
    _ = vtxvals
    vxs = pdf(x_unq, p)
    ll = np.log(vxs)
    return - np.dot(ll, wgt) + ipf(x_unq, p) * k_ipf


def find_minimizer(
        x, p, to_be_minimized, pdf_or_cdf, variant='ml',
        ipf=None, k_ipf=1.0, r_ipf=0.05):
    """Returns the minimizer of the target function, e.g. log-likelihood.

    Parameters
    ----------
    x : ndarray
        1D array of data, ties are allowed.
    p : ndarray
        1D array of parameters.
    to_be_minimized: function
        A function accepting the arguments (x, p, pdf_or_cdf).
        The function is usually the log likelihood or log product of spacings.
    pdf_or_cdf: function
        The probability density function or
        the cumulative distribution function.
    variant: str
        Strings indicating which type of KLDMWRE should be used,
        either of 'ml', 'nbc' or 'zbc'.
    ipf : function, optional
        The internal penalty function, if it is to be used.
    k_ipf : float
        A parameter used for ipf iteration.
    r_ipf :
        A parameter used for ipf iteration.

    Returns
    -------
    list
       a list of objects of the results of the minimizing.
       (the array of estimated parameters, negative kldmwr value of
       the function to_be_minimized, boolean: success or not,
       the whole results of the scipy.optimize.minimize function)
    """
    x_unq, cnt = np.unique(x, return_counts=True)
    vtxvals = np.zeros(len(x_unq) + 2)
    vtxvals[-1] = 1.
    wgt = np.zeros(len(x_unq) + 1)
    if variant is 'zbc':
        wgt[:] = weights_zbc(cnt)
    elif variant is 'nbc':
        wgt[:] = weights_nbc(cnt)
    elif variant is 'ml':
        wgt = np.copy(cnt)
        vtxvals = np.zeros(len(x_unq))

    res = None
    res_x = np.empty(len(p))
    res_x[:] = np.NaN
    success = False
    minimum = np.NaN
    try:
        if ipf is None:
            res = scipy.optimize.minimize(
                to_be_minimized, p, args=(x_unq, pdf_or_cdf, wgt, vtxvals),
                method='Nelder-Mead',
                options={'maxiter': 10000, 'maxfev': 10000}
            )
        else:
            epsilon = 1.0
            count = 0
            while epsilon > 1e-6:
                count += 1
                res = scipy.optimize.minimize(
                    to_be_minimized, p,
                    args=(x_unq, pdf_or_cdf, wgt, vtxvals, ipf, k_ipf),
                    method='Nelder-Mead',
                )
                k_ipf *= r_ipf
                p_est = np.array(res.x)
                epsilon = ipf(x_unq, p_est) * k_ipf
                # The following is necessary, epsilon can be < 0.
                if epsilon < 0:
                    return res_x, minimum, success, res

    except(RuntimeError, TypeError, NameError):
        pass

    else:
        if res.success:
            est_input_same = False
            for i_p in range(len(p)):
                if res.x[i_p] == p[i_p]:
                    est_input_same = True
            if not est_input_same:
                success = True
                if ipf is None:
                    minimum = to_be_minimized(
                        res.x, x_unq, pdf_or_cdf, wgt, vtxvals)
                else:
                    minimum = to_be_minimized(
                        res.x, x_unq, pdf_or_cdf, wgt, vtxvals, ipf, k_ipf)
                res_x = res.x

    return res_x, minimum, success, res


########################################
# Wrapper Functions
########################################


def find_mle(x, p, pdf, ipf=None):
    """Returns the MLE.

    Parameters
    ----------
    x : array like
        1D array or a list of data, ties allowed.
    p : ndarray
        1D array of parameter values
    pdf : function
        The probability density function accepting x and p.
    ipf : function, optional
        The internal penalty function.

    Returns
    -------
    list :
       a list of objects in finding MLE,
       (the array of estimated parameters, negative maximum likelihood,
       boolean indicating success or not,
       the whole results of the scipy.optimize.minimize function)
    """
    if ipf is None:
        return find_minimizer(
            x, p, calc_col, pdf, variant='ml')
    else:
        return find_minimizer(
            x, p, calc_col_ipf, pdf, variant='ml', ipf=ipf)


def find_lsz(x, p, cdf):
    return find_minimizer(x, p, calc_ls, cdf, variant='zbc')


def find_lsn(x, p, cdf):
    return find_minimizer(x, p, calc_ls, cdf, variant='nbc')


def find_glz(x, p, cdf, ipf=None):
    """Returns the ZBGE.

    Parameters
    ----------
    x : ndarray
        1D array or a list of data, ties are allowed.
    p : ndarray
        1D array of parameter values
    cdf : function
        The cumulative distribution function accepting x and p.
    ipf : function, optional
        The internal penalty function.

    Returns
    -------
    list :
       a list of objects in finding ZBGE,
       (the array of estimated parameters, negative maximum product
       of spacings, boolean indicating success or not,
       the whole results of the scipy.optimize.minimize function)
    """
    if ipf is None:
        return find_minimizer(
            x, p, calc_gl, cdf, variant='zbc')
    else:
        return find_minimizer(
            x, p, calc_gl_ipf, cdf, variant='zbc', ipf=ipf)


def find_gln(x, p, cdf, ipf=None):
    """Returns the MPSE or NBGE.

    Parameters
    ----------
    x : ndarray
        1D array or a list of data, ties are allowed.
    p : ndarray
        1D array of parameter values
    cdf : function
        The cumulative distribution function accepting x and p.
    ipf : function, optional
        The internal penalty function.

    Returns
    -------
    list :
       a list of objects in finding NBGE,
       (the array of estimated parameters, negative maximum product
       of spacings, boolean indicating success or not,
       the whole results of the scipy.optimize.minimize function)
    """
    if ipf is None:
        return find_minimizer(
            x, p, calc_gl, cdf, variant='nbc')
    else:
        return find_minimizer(
            x, p, calc_gl_ipf, cdf, variant='nbc', ipf=ipf)


def calc_mse(x, x_0):
    return ((x - x_0) ** 2).mean()


########################################
# Utilities
########################################


def find_min_viv_expl(x, p, find_estimate, pdf_or_cdf, p_ints=None, ipf=None):
    """For backward compatibility, see find_min_viv
    """
    return find_min_viv(x, p, find_estimate, pdf_or_cdf, p_ints=p_ints, ipf=ipf)


def find_min_viv(x, p, find_estimate, pdf_or_cdf, p_ints=None, ipf=None):
    """Return the minimizer of the

    Parameters
    ----------
    x
    p
    find_estimate
    pdf_or_cdf
    p_ints
    ipf

    Returns
    -------

    """
    if p_ints is None:
        p_ints = np.array([p])
    else:
        p_ints = p_ints
    res_x = np.empty(len(p))
    res_x[:] = np.NaN
    success = False
    minimum = np.NaN
    df = None
    ress = []
    count = 0
    count_same = 0
    for p_int in p_ints:
        res = find_estimate(x, p_int, pdf_or_cdf, ipf)
        if res[2]:
            count += 1
            ress.append(
                np.append(res[0], np.append(res[1], np.append(res[2], p_int))))
            if 2 <= count:
                if np.allclose(
                        ress[count - 1][0:3], ress[count - 2][0:3], atol=1e-4):
                    count_same += 1
                else:
                    count_same = 0
        if count_same >= 2:
            break

    if count != 0:
        df = pd.DataFrame(ress)
        df.columns = ['loc', 'scl', 'shp', 'mmps', 'success',
                      'p_i_loc', 'p_i_scl', 'p_i_shp']
        minv = df.iloc[df['mmps'].idxmin(), :]
        res_x = np.array([minv['loc'], minv['scl'], minv['shp']])
        minimum = minv['mmps']
        success = True

    return res_x, minimum, success, df


########################################
# Trials
########################################


def log_likelihood(x, p, pdf):
    return np.sum(np.log(pdf(x, p)))


def log_product_of_spacings_wgt_zbc(x, p, cdf):
    x_unq, cnt = np.unique(x, return_counts=True)
    wgt = weights_zbc(cnt)
    vxtvals = np.zeros(len(x_unq) + 2)
    vxtvals[-1] = 1
    return - calc_gl(p, x_unq, cdf, wgt, vxtvals)


def integral_subdomain(x, p, cdf):
    ffs = cdf(x, p)
    effs = np.append([0], np.append(ffs, [1]))
    deffs = np.diff(effs)
    pffpp = deffs * (len(x) + 1)
    return np.sum(np.log(pffpp)) / (len(x) + 1)


def integral_trapezoidal(x, p, cdf):
    ffs = cdf(x, p)
    effs = np.append([0], np.append(ffs, [1]))
    deffs = np.diff(effs)
    pffpp = deffs * (len(x) + 1)
    return 0.5 * np.sum(np.log(pffpp[1:]) + np.log(pffpp[:-1])) / (len(x) + 1)


def extended_dxs(x, p_ref, pdf, cdf):
    ff1 = cdf(x[0], p_ref)
    ffn = cdf(x[-1], p_ref)
    f1 = pdf(x[0], p_ref)
    fn = pdf(x[-1], p_ref)
    dx1 = ff1 / (f1 * 0.5)
    dxn = (1 - ffn) / (fn * 0.5)
    dxs = np.diff(x)
    edxs = np.append([dx1], np.append(dxs, [dxn]))
    return edxs


def calc_h(x, p_ref, pdf, cdf):
    edxs = extended_dxs(x, p_ref, pdf, cdf)
    n = len(x)
    a = - n / (n + 1) * np.log(n + 1)
    wgt = np.ones(len(edxs))
    wgt[0] = 0.5
    wgt[-1] = 0.5
    b = np.dot(wgt, np.log(edxs))
    return a - b / (n + 1)
