import unittest
import numpy as np
from kldmwr import univar
from kldmwr.distributions import *


class TestUniverFinds(unittest.TestCase):

    def test_find_gln(self):
        x = [2, 1, 0]
        p_0 = [0, 1]
        res_test = univar.find_gln(x, p_0, nrm_loc_var_cdf)
        np.testing.assert_almost_equal(res_test[0][0], 1, decimal=4)

    def test_find_glz(self):
        x = [3, 2, 1]
        p_0 = [0, 1]
        res_test = univar.find_glz(x, p_0, nrm_loc_var_cdf)
        np.testing.assert_almost_equal(res_test[0][0], 2, decimal=4)


class TestUniverWeights(unittest.TestCase):

    def setUp(self):
        self.x = [1., 2., 3., 3., 4., 5., 5., 5., 6.]
        x_unq, self.cnt = np.unique(self.x, return_counts=True)

    def test_weights_zbc(self):
        w_zbc = univar.weights_zbc(self.cnt)
        w_zbc_expctd = [0.5, 1., 1.5, 1.5, 2., 2., 0.5]
        np.testing.assert_equal(w_zbc, w_zbc_expctd)

    # Depreciated.
    # def test_weights_nbc(self):
    #    w_nbc = univar.weights_nbc(self.cnt)
    #    w_nbc_expctd = [1, 1, 2, 1, 3, 1, 1]
    #    np.testing.assert_almost_equal(w_nbc, w_nbc_expctd, decimal=4)

    def test_weights_nbc_2(self):
        w_zbc = univar.weights_nbc_2(self.cnt)
        w_zbc_expctd = [1, 1., 1.5, 1.5, 2., 2., 1]
        np.testing.assert_equal(w_zbc, w_zbc_expctd)


class TestUniverCalcs(unittest.TestCase):

    def setUp(self):
        self.x = [1., 2., 3., 3., 4., 5., 5., 5., 6.]

    def test_calc_ls(self):
        x_unq, cnt = np.unique(self.x, return_counts=True)
        wgt = univar.weights_zbc(cnt)
        vtxvals = np.zeros(len(x_unq) + 2)
        vtxvals[-1] = 1.
        p_0 = [4, 1]
        res = univar.calc_ls(p_0, x_unq, nrm_loc_var_cdf, wgt, vtxvals)
        res_expctd = 61.7512
        np.testing.assert_almost_equal(res, res_expctd, decimal=4)

    def test_calc_gl(self):
        x_unq, cnt = np.unique(self.x, return_counts=True)
        wgt = univar.weights_zbc(cnt)
        vtxvals = np.zeros(len(x_unq) + 2)
        vtxvals[-1] = 1.
        p_0 = [4, 1]
        res = univar.calc_gl(p_0, x_unq, nrm_loc_var_cdf, wgt, vtxvals)
        res_expctd = 19.78712
        np.testing.assert_almost_equal(res, res_expctd, decimal=4)

    def test_calc_col(self):
        x_unq, cnt = np.unique(self.x, return_counts=True)
        wgt = np.copy(cnt)
        vtxvals = np.zeros(len(x_unq))
        p_0 = [4, 1]
        res = univar.calc_col(p_0, x_unq, nrm_loc_var_pdf, wgt, vtxvals)
        res_expctd = 19.2704
        np.testing.assert_almost_equal(res, res_expctd, decimal=4)


class TestUniverFindMinVivGEVGLZ(unittest.TestCase):

    def setUp(self):
        self.x = [1.40394047, 1.11188719, 1.4997714, 0.86211755, 0.54971989,
                  -3.42834528, 1.34470115, 0.85459417, 1.49991069, 1.17818155,
                  1.34353202, 1.40941076, 1.11944558, -3.53627481, 1.06328572,
                  1.05364191, 1.17654182, 0.13336439, -1.30429455, 1.29939666]

    def test_find_min_viv(self):
        p_0 = [1, 1, -2]
        cdf = gev_cdf
        res = univar.find_min_viv(self.x, p_0, univar.find_glz, cdf)
        hat_p_expected = [0.82456064,  1.16233274, -1.72081437]
        minimum_expected = 80.86083174620651
        np.testing.assert_almost_equal(res[0], hat_p_expected, decimal=4)
        np.testing.assert_almost_equal(res[1], minimum_expected, decimal=4)


class TestUniverFindMinVivGEVGLZ2(unittest.TestCase):

    def setUp(self):
        self.x = [1.56168611, 1.11896332, 1.97861754, 0.87050237, 0.62139193,
                  -1.13953668, 1.44268707, 0.86386107, 1.98663502, 1.19773015,
                  1.44059321, 1.57434934, 1.12758447, -1.17372803, 1.065426,
                  1.05516341, 1.19568889, 0.34673922, -0.368246, 1.36659121]

    def test_find_min_viv(self):
        p_0 = [1, 1, -1]
        cdf = gev_cdf
        res = univar.find_min_viv(self.x, p_0, univar.find_glz, cdf)
        hat_p_expected = [0.78482996,  0.94230863, -0.72683338]
        minimum_expected = 80.19902138890097
        np.testing.assert_almost_equal(res[0], hat_p_expected, decimal=4)
        np.testing.assert_almost_equal(res[1], minimum_expected, decimal=4)


class TestUniverFindMinVivGEVMLE(unittest.TestCase):

    def setUp(self):
        self.x = [0.36292384, 0.54536925, 0.72295333, 0.8862828, 0.89242325,
                  2.10243197, 3.43829774, 6.54005558, 6.63949518, 8.81658036]

    def test_find_min_viv(self):
        p_0 = [1, 1, 1]
        pdf = gev_pdf
        res = univar.find_min_viv(self.x, p_0, univar.find_mle, pdf)
        hat_p_expected = [0.92544841, 0.86717912, 1.20619691]
        minimum_expected = 20.667777025526618
        np.testing.assert_almost_equal(res[0], hat_p_expected, decimal=4)
        np.testing.assert_almost_equal(res[1], minimum_expected, decimal=4)


if __name__ == '__main__':
    unittest.main()
