import unittest
import numpy as np
from kldmwr import univar
from kldmwr.distributions import *


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
            [[-1.14645714, -0.45039388, -0.10960559],
             [-0.46125543, -0.7840997,  0.02466962],
             [0.10488395, -0.16099013,  0.05799958],
             [0.85976079,  0.46533204,  0.11635839],
             [0.64306783,  0.93015168, -0.089422]]
        )
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_dd_2nd_derivs(self):
        res_test = univar.calc_dd_2nd_derivs(
            self.x, self.p_hat, tpw_cdf, self.pps)
        res_expt = np.array(
            [[[4.48977796,  3.81108624,  0.39421087],
              [3.81108624,  2.30148723,  0.15486856],
              [0.39421087,  0.15486856,  0.08866737]],
             [[-4.74553206, -1.13655355, -0.95797024],
              [-1.13655355,  1.95662462, -0.58775521],
              [-0.95797024, -0.58775521, -0.07894586]],
             [[-1.7191163, -1.82471839, -0.11882711],
              [-1.82471839, -0.97706337, -0.20094365],
              [-0.11882711, -0.20094365, -0.0094275]],
             [[-0.8865393, -3.84030364,  0.67098188],
              [-3.84030364, -5.94559257,  0.61704508],
              [0.67098188,  0.61704508, -0.04027196]],
             [[2.8614097,  2.99048934,  0.0116046],
              [2.99048934,  2.66454408,  0.01678522],
              [0.0116046,  0.01678522,  0.03997795]]]
        )
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_lps_2nd_deriv_terms(self):
        res_test = univar.calc_lps_2nd_deriv_terms(
            self.x, self.p_hat, tpw_cdf, self.ps, self.pps)
        res_expt = np.array(
            [[[3.17541398, 3.29472896,  0.26855277],
              [3.29472896,  2.09863259,  0.10550287],
              [0.26855277,  0.10550287,  0.07665399]],
             [[-4.95828863, -1.49822379, -0.94659124],
              [-1.49822379,  1.34181228, -0.56841177],
              [-0.94659124, -0.56841177, -0.07955445]],
             [[-1.73011694, -1.80783311, -0.12491034],
              [-1.80783311, -1.00298119, -0.19160629],
              [-0.12491034, -0.19160629, -0.01279145]],
             [[-1.62572792, -4.24037788,  0.5709415],
              [-4.24037788, -6.16212648,  0.56289979],
              [0.5709415,  0.56289979, -0.05381124]],
             [[2.44787347,  2.39233872,  0.06910901],
              [2.39233872,  1.79936194,  0.09996125],
              [0.06910901,  0.09996125,  0.03198165]]]
        )
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)

    def test_calc_lps_2nd_deriv(self):
        res_test = univar.calc_lps_2nd_deriv(
            self.x, self.p_hat, tpw_cdf, self.ps, self.pps)
        res_expt = np.array(
            [[-2.69084604, -1.8593671, -0.1628983],
             [-1.8593671 , -1.92530087,  0.00834586],
             [-0.1628983 ,  0.00834586, -0.0375215]])
        np.testing.assert_almost_equal(res_test, res_expt, decimal=4)


if __name__ == '__main__':
    unittest.main()
