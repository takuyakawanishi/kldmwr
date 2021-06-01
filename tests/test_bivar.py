import unittest
import numpy as np
import scipy.stats
import sys
from kldmwr import bivar
from kldmwr import distributions2d

def bvnrm_pdf(x, p):
    mu = [0, 0]
    sgm = [[p[0], p[1]], [p[1], p[2]]]
    return scipy.stats.multivariate_normal.pdf(x, mean=mu, cov=sgm)


def bvnrm_cdf(x, p):
    mu = [0, 0]
    sgm = [[p[0], p[1]], [p[1], p[2]]]
    return scipy.stats.multivariate_normal.cdf(x, mean=mu, cov=sgm)


def bvnrm_marx(t, p):
    return scipy.stats.norm.cdf(t, loc=0, scale=p[0]**.5)


def bvnrm_mary(t, p):
    return scipy.stats.norm.cdf(t, loc=0, scale=p[2]**.5)


def bvnrm_ptry(p):
    p1 = scipy.stats.uniform.rvs() * 2 - 1
    return([p[0], p1, p[2]])


class TestBivarBBVT(unittest.TestCase):

    def setUp(self):

        self.name = 'test_bivar'
        self.xy = np.array([
           [-1.797, -0.648],
           [0.436,  0.812],
           [0.436,  0.812],
           [-0.044, -0.884],
           [-0.064, -0.884],
           [1.886,  0.957]
        ])
        self.p_0 = [1, .5, 1]

    def test_order_stats(self):

        nux, nuy, ijunq, ijcnt, qqs_x, qqs_y = bivar.order_stats(self.xy)
        expected_ijunq = [
            [1, 2],
            [2, 1],
            [3, 1],
            [4, 3],
            [5, 4]
        ]
        expected_ijcnt = [1, 1, 1, 2, 1]
        expected_qqs_x = [-1.797, -0.064, -0.044, 0.436, 1.886]
        expected_qqs_y = [-0.884, -0.648, 0.812, 0.957]
        self.assertEqual(nux, 5)
        self.assertEqual(nuy, 4)
        np.testing.assert_equal(expected_ijunq, ijunq)
        np.testing.assert_equal(expected_ijcnt, ijcnt)
        np.testing.assert_almost_equal(expected_qqs_x, qqs_x, decimal=6)
        np.testing.assert_almost_equal(expected_qqs_y, qqs_y, decimal=6)

    def test_relbbs(self):

        ijunq_t = np.array([
            [1, 2],
            [2, 1],
            [3, 1],
            [4, 3],
            [5, 4]
        ])
        ijcnt_t = [1, 1, 1, 2, 1]
        relbbs, w_rbbs = bivar.find_relbbs(ijunq_t, ijcnt_t)
        expected_relbbs = np.array([
            [1, 2], [1, 3], [2, 3], [2, 2], [2, 1], [3, 2], [3, 1], [4, 2],
            [4, 1], [4, 3], [4, 4], [5, 4], [5, 3], [5, 5], [6, 5], [6, 4]
        ])
        expected_w_rbbs = np.array(
            [1, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 3, 2, 1, 1, 1]
        )
        np.testing.assert_equal(expected_relbbs, relbbs)
        np.testing.assert_equal(expected_w_rbbs, w_rbbs)

    def test_find_boundary_bbs(self):

        relbbs_t = [
            [1, 2], [1, 3], [2, 3], [2, 2], [2, 1], [3, 2], [3, 1], [4, 2],
            [4, 1], [4, 3], [4, 4], [5, 4], [5, 3], [5, 5], [6, 5], [6, 4]
        ]
        nux_t = 5
        nuy_t = 4
        w_bnds = bivar.find_boundary_bbs(relbbs_t, nux_t, nuy_t)
        expected_w_bnds = [
            2., 2., 1., 1., 2., 1., 2., 1., 2., 1., 1., 1., 1., 2., 4., 2.
        ]
        np.testing.assert_equal(expected_w_bnds, w_bnds)

    def test_find_relvts(self):

        relbbs_t = np.array([
            [1, 2], [1, 3], [2, 3], [2, 2], [2, 1], [3, 2], [3, 1], [4, 2],
            [4, 1], [4, 3], [4, 4], [5, 4], [5, 3], [5, 5], [6, 5], [6, 4]
        ])
        relvts = bivar.find_relvts(relbbs_t)
        expected_relvts = np.array([
            [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0],
            [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [5, 2], [5, 3],
            [5, 4], [5, 5], [6, 3], [6, 4], [6, 5]
        ])
        np.testing.assert_equal(expected_relvts, relvts)

    def test_find_relvts_in(self):

        relvts_in_t = np.array([
            [0, 1], [0, 2], [0, 3], [1, 0], [1, 1], [1, 2], [1, 3], [2, 0],
            [2, 1], [2, 2], [2, 3], [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [5, 2], [5, 3],
            [5, 4], [5, 5], [6, 3], [6, 4], [6, 5]
        ])
        nux_t = 5
        nuy_t = 4
        relvts_in = bivar.find_relvts_in(relvts_in_t, nux_t, nuy_t)
        expected_relvts_in = np.array([
            [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [3, 1], [3, 2],
            [3, 3], [3, 4], [4, 1], [4, 2], [4, 3], [4, 4], [5, 2], [5, 3],
            [5, 4]
        ])
        np.testing.assert_equal(expected_relvts_in, relvts_in)

    def test_zbce(self):

        res_zbce = bivar.zbce(
            self.xy, self.p_0, bvnrm_cdf, bvnrm_marx, bvnrm_mary, bvnrm_ptry
        )
        expected_res_0 = [1.40075002, 0.67631764, 0.89521719]
        expected_res_1 = True
        np.testing.assert_almost_equal(expected_res_0, res_zbce[0], decimal=6)
        self.assertEqual(expected_res_1, res_zbce[1])

    def test_mle(self):

        res_mle = bivar.mle(self.xy, self.p_0, bvnrm_pdf, bvnrm_ptry)
        expected_res_0 = [1.19534318, 0.62878002, 0.70289784]
        expected_res_1 = True
        np.testing.assert_almost_equal(expected_res_0, res_mle[0], decimal=6)
        self.assertEqual(expected_res_1, res_mle[1])


class TestBivarNormal(unittest.TestCase):

    def setUp(self):

        self.name = 'test_bivar_normal'
        self.xyf = [
            -0.7205309867971943, 0.6441177698286161, 0.2705656886204725,
            0.47168357423144375, -0.9074452994750679, 0.09476762389772658,
            -0.5850334108847408, -0.34369186490160786, -0.664133497596725,
            0.15114917436089836, 1.602745021788436, -0.5084754998713794,
            -2.4903447379498234, -0.8001755224577575, 0.5613884958479634,
            -0.21340871665608885, -1.2699194113545624, -0.4390206699334978,
            -1.481148579488512, 0.49020300893012597
        ]
        self.xyf = np.array(self.xyf)
        self.xyf = np.reshape(self.xyf, (10, 2))
        self.p_i = [1, .5, 1]

    def test_zbce(self):
        res_zbc = bivar.zbce(
            self.xyf, self.p_i, bvnrm_cdf, bvnrm_marx, bvnrm_mary, bvnrm_ptry
        )
        expected_zbc_0 = [1.68423674, 0.0144281 , 0.24791642]
        expected_zbc_1 = True
        expected_zbc_2 = -218.26642802609823
        expected_zbc_3_is_not = 1
        np.testing.assert_almost_equal(expected_zbc_0, res_zbc[0], decimal=4)
        self.assertEqual(expected_zbc_1, res_zbc[1])
        np.testing.assert_almost_equal(expected_zbc_2, res_zbc[2], decimal=4)
        self.assertNotEqual(expected_zbc_3_is_not, res_zbc[3])

    def test_zbce_returns_nan(self):
        res_zbc = bivar.zbce(
            self.xyf, self.p_i, bvnrm_cdf, bvnrm_marx, bvnrm_mary, bvnrm_ptry,
            max_count=1
        )
        expected_zbc_0 = [np.NaN, np.NaN, np.NaN]
        expected_zbc_1 = False
        expected_zbc_2 = np.NaN
        np.testing.assert_equal(expected_zbc_0, res_zbc[0])
        self.assertEqual(expected_zbc_1, res_zbc[1])
        np.testing.assert_almost_equal(expected_zbc_2, res_zbc[2], decimal=4)


if __name__ == '__main__':
    unittest.main()
