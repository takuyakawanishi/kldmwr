import unittest
from kldmwr import univar
from kldmwr.distributions import *
import numpy as np


class TestUniverDerivs(unittest.TestCase):

    def setUp(self):
        self.x = np.array([5.13, 5.34, 5.43, 5.72])
        self.p_hat = [4.91, 0.56, 2.15]
        self.ps = [tpw_cdf_mu, tpw_cdf_sg, tpw_cdf_xi]
        self.pps = [[tpw_cdf_mumu, tpw_cdf_musg, tpw_cdf_muxi],
                    [tpw_cdf_musg, tpw_cdf_sgsg, tpw_cdf_sgxi],
                    [tpw_cdf_muxi, tpw_cdf_sgxi, tpw_cdf_xixi]]

    def test_calc_dds_tpw(self):
        res_test = univar.calc_dds(self.x, self.p_hat, tpw_cdf)
        res_expt = [0.12554485, 0.30706018, 0.14113808, 0.31669366, 0.10956323]
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_diffs_for_derivs(self):
        res_test = univar.calc_diffs_for_derivs(self.x, self.p_hat, tpw_cdf_mu)
        res_expt = [-1.14645714, -0.46125543,  0.10488395,  0.85976079,
                    0.64306783]
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_dd_1st_derivs(self):
        res_test = univar.calc_dd_1st_derivs(
            self.x, self.p_hat, tpw_cdf, self.ps)
        res_expt = np.array(
            [[-9.13185327, -3.58751379, -0.87303929],
             [-1.50216622, -2.55357013, 0.08034133],
             [0.74313007, -1.14065694, 0.41094209],
             [2.71480264, 1.46934433, 0.36741622],
             [5.86937656, 8.48963396, -0.81616806]]
        )
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_dd_2nd_derivs(self):
        res_test = univar.calc_dd_2nd_derivs(
            self.x, self.p_hat, tpw_cdf, self.pps)
        res_expt = np.array(
            [[[35.76234293, 30.35637271, 3.14000038],
              [30.35637271, 18.33199247, 1.23357158],
              [3.14000038, 1.23357158, 0.70626055]],

             [[-15.45472965, -3.70140326, -3.11981268],
              [-3.70140326, 6.37212101, -1.91413688],
              [-3.11981268, -1.91413688, -0.25710224]],

             [[-12.18038575, -12.92860402, -0.84192097],
              [-12.92860402, -6.92274789, -1.42373795],
              [-0.84192097, -1.42373795, -0.06679628]],

             [[-2.79935914, -12.1262409, 2.11870953],
              [-12.1262409, -18.77395502, 1.9483973],
              [2.11870953, 1.9483973, -0.12716378]],

             [[26.11651575, 27.29464499, 0.10591692],
              [27.29464499, 24.31969372, 0.15320126],
              [0.10591692, 0.15320126, 0.36488474]]]
        )
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_lps_2nd_deriv_terms(self):
        res_test = univar.calc_lps_2nd_deriv_terms(
            self.x, self.p_hat, tpw_cdf, self.ps, self.pps)
        res_expt = np.array(
            [[[-47.62840124, -2.40427679, -4.83246631],
              [-2.40427679, 5.46173731, -1.89846891],
              [-4.83246631, -1.89846891, -0.05593705]],

             [[-17.71123299, -7.53729004, -2.99912664],
              [-7.53729004, -0.14859942, -1.70897965],
              [-2.99912664, -1.70897965, -0.26355697]],

             [[-12.73262805, -12.08094754, -1.1473044],
              [-12.08094754, -8.22384615, -0.954994],
              [-1.1473044, -0.954994, -0.23566969]],

             [[-10.16951252, -16.11522078, 1.12124701],
              [-16.11522078, -20.93292779, 1.40853637],
              [1.12124701, 1.40853637, -0.26215846]],

             [[-8.33306549, -22.53421359, 4.89631459],
              [-22.53421359, -47.75419101, 7.08216932],
              [4.89631459, 7.08216932, -0.30124556]]]
        )
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_lps_2nd_deriv(self):
        wgt = np.array([0.5, 1., 1., 1., 0.5])
        res_test = univar.calc_lps_2nd_deriv(
            self.x, self.p_hat, tpw_cdf, self.ps, self.pps, wgt)
        res_expt = np.array(
            [[-68.59410693, -48.20270355, -2.99325988],
             [-48.20270355, -50.45160022, 1.33641292],
             [-2.99325988, 1.33641292, -0.93997642]]
        )
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)


if __name__ == '__main__':
    unittest.main()


