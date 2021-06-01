import unittest
import numpy as np
from kldmwr import distributions
from kldmwr import gev


class TestCalcDerivativesOfh(unittest.TestCase):

    def setUp(self):
        xi = np.random.uniform(-2, 2)
        # print(xi)
        self.p_0 = np.array([1, 1, xi])
        self.x = distributions.gev_sampling(1, 4, self.p_0)
        self.dlt = 1e-8
        self.rtol_set = 1e-3
        self.ppdmu = self.p_0 + np.array([self.dlt, 0, 0])
        self.ppdsg = self.p_0 + np.array([0, self.dlt, 0])
        self.ppdxi = self.p_0 + np.array([0, 0, self.dlt])

    def test_xi_in_range(self):
        xi = self.p_0[2]
        # print(self.p_0)
        self.assertGreaterEqual(2, xi)
        self.assertLessEqual(-2, xi)

    def test_calc_h_mu(self):
        h_mu_num = (gev.calc_h(self.x, self.ppdmu) -
                    gev.calc_h(self.x, self.p_0)) / self.dlt
        h_mu_fml = gev.calc_h_mu(self.x, self.p_0)
        # print(h_mu_num)
        # print(h_mu_fml)
        np.testing.assert_allclose(h_mu_fml, h_mu_num, rtol=self.rtol_set)

    def test_calc_h_sg(self):
        h_sg_num = (gev.calc_h(self.x, self.ppdsg) -
                    gev.calc_h(self.x, self.p_0)) / self.dlt
        h_sg_fml = gev.calc_h_sg(self.x, self.p_0)
        # print(h_sg_num)
        # print(h_sg_fml)
        np.testing.assert_allclose(h_sg_fml, h_sg_num, rtol=self.rtol_set)

    def test_calc_h_xi(self):
        h_xi_num = (gev.calc_h(self.x, self.ppdxi) -
                    gev.calc_h(self.x, self.p_0)) / self.dlt
        h_xi_fml = gev.calc_h_xi(self.x, self.p_0)
        # print(h_xi_num)
        # print(h_xi_fml)
        np.testing.assert_allclose(h_xi_fml, h_xi_num, rtol=self.rtol_set)

    def test_calc_h_mumu(self):
        h_mumu_num = (gev.calc_h_mu(self.x, self.ppdmu) -
                      gev.calc_h_mu(self.x, self.p_0)) / self.dlt
        h_mumu_fml = gev.calc_h_mumu(self.x, self.p_0)
        # print(h_mumu_num)
        # print(h_mumu_fml)
        np.testing.assert_allclose(h_mumu_fml, h_mumu_num, rtol=self.rtol_set)

    def test_calc_h_sgsg(self):
        h_sgsg_num = (gev.calc_h_sg(self.x, self.ppdsg) -
                      gev.calc_h_sg(self.x, self.p_0)) / self.dlt
        h_sgsg_fml = gev.calc_h_sgsg(self.x, self.p_0)
        # print(h_sgsg_num)
        # print(h_sgsg_fml)
        np.testing.assert_allclose(h_sgsg_fml, h_sgsg_num, rtol=self.rtol_set)

    def test_calc_h_xixi(self):
        h_xixi_num = (gev.calc_h_xi(self.x, self.ppdxi) -
                      gev.calc_h_xi(self.x, self.p_0)) / self.dlt
        h_xixi_fml = gev.calc_h_xixi(self.x, self.p_0)
        # print(h_xixi_num)
        # print(h_xixi_fml)
        np.testing.assert_allclose(h_xixi_fml, h_xixi_num, rtol=self.rtol_set)

    def test_calc_h_musg(self):
        h_musg_num = (gev.calc_h_mu(self.x, self.ppdsg) -
                      gev.calc_h_mu(self.x, self.p_0)) / self.dlt
        h_musg_fml = gev.calc_h_musg(self.x, self.p_0)
        # print(h_musg_num)
        # print(h_musg_fml)
        np.testing.assert_allclose(h_musg_fml, h_musg_num, rtol=self.rtol_set)

    def test_calc_h_muxi(self):
        h_muxi_num = (gev.calc_h_mu(self.x, self.ppdxi) -
                      gev.calc_h_mu(self.x, self.p_0)) / self.dlt
        h_muxi_fml = gev.calc_h_muxi(self.x, self.p_0)
        # print(h_muxi_num)
        # print(h_muxi_fml)
        np.testing.assert_allclose(h_muxi_fml, h_muxi_num, rtol=self.rtol_set)

    def test_calc_h_sgxi(self):
        h_sgxi_num = (gev.calc_h_sg(self.x, self.ppdxi) -
                      gev.calc_h_sg(self.x, self.p_0)) / self.dlt
        h_sgxi_fml = gev.calc_h_sgxi(self.x, self.p_0)
        # print(h_sgxi_num)
        # print(h_sgxi_fml)
        np.testing.assert_allclose(h_sgxi_fml, h_sgxi_num, rtol=self.rtol_set)


class TestCalcDerivativesOfG(unittest.TestCase):

    def setUp(self):
        xi = np.random.uniform(-2, 2)
        # print(xi)
        self.p_0 = np.array([1, 1, xi])
        self.x = distributions.gev_sampling(1, 4, self.p_0)
        self.dlt = 1e-8
        self.rtol_set = 1e-3
        self.ppdmu = self.p_0 + np.array([self.dlt, 0, 0])
        self.ppdsg = self.p_0 + np.array([0, self.dlt, 0])
        self.ppdxi = self.p_0 + np.array([0, 0, self.dlt])

    def test_xi_in_range_G(self):
        xi = self.p_0[2]
        # print(self.p_0)
        self.assertGreaterEqual(2, xi)
        self.assertLessEqual(-2, xi)

    def test_calc_h_mu(self):
        gg_mu_num = (gev.calc_gg(self.x, self.ppdmu) -
                     gev.calc_gg(self.x, self.p_0)) / self.dlt
        gg_mu_fml = gev.calc_gg_mu(self.x, self.p_0)
        # print(h_mu_num)
        # print(h_mu_fml)
        np.testing.assert_allclose(gg_mu_fml, gg_mu_num, rtol=self.rtol_set)

    def test_calc_h_sg(self):
        gg_sg_num = (gev.calc_gg(self.x, self.ppdsg) -
                     gev.calc_gg(self.x, self.p_0)) / self.dlt
        gg_sg_fml = gev.calc_gg_sg(self.x, self.p_0)
        # print(h_sg_num)
        # print(h_sg_fml)
        np.testing.assert_allclose(gg_sg_fml, gg_sg_num, rtol=self.rtol_set)

    def test_calc_h_xi(self):
        gg_xi_num = (gev.calc_gg(self.x, self.ppdxi) -
                     gev.calc_gg(self.x, self.p_0)) / self.dlt
        gg_xi_fml = gev.calc_gg_xi(self.x, self.p_0)
        # print(h_xi_num)
        # print(h_xi_fml)
        np.testing.assert_allclose(gg_xi_fml, gg_xi_num, rtol=self.rtol_set)

    def test_calc_gg_mumu(self):
        gg_mumu_num = (gev.calc_gg_mu(self.x, self.ppdmu) -
                       gev.calc_gg_mu(self.x, self.p_0)) / self.dlt
        gg_mumu_fml = gev.calc_gg_mumu(self.x, self.p_0)
        # print(gg_mumu_num)
        # print(gg_mumu_fml)
        np.testing.assert_allclose(gg_mumu_fml, gg_mumu_num, rtol=self.rtol_set)

    def test_calc_gg_sgsg(self):
        gg_sgsg_num = (gev.calc_gg_sg(self.x, self.ppdsg) -
                       gev.calc_gg_sg(self.x, self.p_0)) / self.dlt
        gg_sgsg_fml = gev.calc_gg_sgsg(self.x, self.p_0)
        # print(gg_sgsg_num)
        # print(gg_sgsg_fml)
        np.testing.assert_allclose(gg_sgsg_fml, gg_sgsg_num, rtol=self.rtol_set)

    def test_calc_gg_xixi(self):
        gg_xixi_num = (gev.calc_gg_xi(self.x, self.ppdxi) -
                       gev.calc_gg_xi(self.x, self.p_0)) / self.dlt
        gg_xixi_fml = gev.calc_gg_xixi(self.x, self.p_0)
        # print(gg_xixi_num)
        # print(gg_xixi_fml)
        np.testing.assert_allclose(gg_xixi_fml, gg_xixi_num, rtol=self.rtol_set)

    def test_calc_gg_musg(self):
        gg_musg_num = (gev.calc_gg_mu(self.x, self.ppdsg) -
                       gev.calc_gg_mu(self.x, self.p_0)) / self.dlt
        gg_musg_fml = gev.calc_gg_musg(self.x, self.p_0)
        # print(gg_musg_num)
        # print(gg_musg_fml)
        np.testing.assert_allclose(gg_musg_fml, gg_musg_num, rtol=self.rtol_set)

    def test_calc_gg_muxi(self):
        gg_muxi_num = (gev.calc_gg_mu(self.x, self.ppdxi) -
                       gev.calc_gg_mu(self.x, self.p_0)) / self.dlt
        gg_muxi_fml = gev.calc_gg_muxi(self.x, self.p_0)
        # print(gg_muxi_num)
        # print(gg_muxi_fml)
        np.testing.assert_allclose(gg_muxi_fml, gg_muxi_num, rtol=self.rtol_set)

    def test_calc_gg_sgxi(self):
        gg_sgxi_num = (gev.calc_gg_sg(self.x, self.ppdxi) -
                       gev.calc_gg_sg(self.x, self.p_0)) / self.dlt
        gg_sgxi_fml = gev.calc_gg_sgxi(self.x, self.p_0)
        # print(gg_sgxi_num)
        # print(gg_sgxi_fml)
        np.testing.assert_allclose(gg_sgxi_fml, gg_sgxi_num, rtol=self.rtol_set)


class TestCalcOfDs(unittest.TestCase):

    def setUp(self):
        self.x = np.array([-1.05, 1, 1.12, 1.37])
        self.p_hat = np.array([0.59, 1.7, -2.07])

    def test_eff(self):
        eff = gev.calc_eff(self.x, self.p_hat)
        eff_exp = np.array(
            [0., 0.18280481, 0.48871727, 0.54549993, 0.78997112, 1.]
        )
        np.testing.assert_almost_equal(eff, eff_exp, decimal=6)

    def test_eff_p(self):
        eff_p = gev.calc_eff_p(self.x, self.p_hat)
        eff_p_exp = np.array(
            [[0, -0.06097332, -0.4110272, -0.54835238, -2.18082809, 0],
             [0, 0.05882132, -0.09913009, -0.17095692, -1.00061524, 0],
             [0, -0.03126592, -0.02493318, -0.060418, -0.69175619, 0]]
        )
        np.testing.assert_almost_equal(eff_p, eff_p_exp, decimal=6)

    def test_eff_pp(self):
        eff_pp = gev.calc_eff_pp(self.x, self.p_hat)
        eff_pp_exp = np.array(
            [[[0.00000000e+00, 3.31427373e-02, 8.62307908e-01,
               1.52441108e+00, 3.33446801e+01, 0.00000000e+00],
              [0.00000000e+00, 3.89366308e-03, 4.49749086e-01,
               7.97817797e-01, 1.65821639e+01, 0.00000000e+00],
              [0.00000000e+00, -1.53354101e-02, 1.89638685e-01,
               4.42569662e-01, 1.37280726e+01, 0.00000000e+00]],

             [[0.00000000e+00, 3.89366308e-03, 4.49749086e-01,
               7.97817797e-01, 1.65821639e+01, 0.00000000e+00],
              [0.00000000e+00, -3.83570144e-02, 1.66780715e-01,
               3.49294325e-01, 8.19688416e+00, 0.00000000e+00],
              [0.00000000e+00, 1.47941603e-02, 4.57363886e-02,
               1.37977601e-01, 6.29876271e+00, 0.00000000e+00]],

             [[0.00000000e+00, -1.53354101e-02, 1.89638685e-01,
               4.42569662e-01, 1.37280726e+01, 0.00000000e+00],
              [0.00000000e+00, 1.47941603e-02, 4.57363886e-02,
               1.37977601e-01, 6.29876271e+00, 0.00000000e+00],
              [0.00000000e+00, -1.24579270e-02, 1.46143135e-02,
               6.06982830e-02, 4.87356594e+00, 0.00000000e+00]]]
        )
        np.testing.assert_almost_equal(eff_pp, eff_pp_exp, decimal=6)

    def test_calc_dd_p_o_dd(self):
        dd_p_o_dd = gev.calc_dd_p_o_dd(self.x, self.p_hat)
        dd_p_o_dd_exp = np.array(
            [[-0.33354328, -1.14429432, -2.41843517, -6.67757923, 10.38346746],
             [0.32177116, -0.51632878, -1.26494304, -3.39368553, 4.76417919],
             [-0.17103444, 0.02070116, -0.62492361, -2.58246463, 3.29362406]]
        )
        np.testing.assert_almost_equal(
            dd_p_o_dd, dd_p_o_dd_exp, decimal=6)

    def test_calc_dd_pp_o_dd(self):
        dd_pp_o_dd = gev.calc_dd_pp_o_dd(self.x, self.p_hat)
        dd_pp_o_dd_exp = np.array(
            [[[1.81301230e-01, 2.71046552e+00, 1.16603059e+01,
               1.30159589e+02, -1.58762354e+02],
              [2.12995655e-02, 1.45746082e+00, 6.12984172e+00,
               6.45652619e+01, -7.89518256e+01],
              [-8.38895314e-02, 6.70041670e-01, 4.45437011e+00,
               5.43438398e+01, -6.53627838e+01]],

             [[2.12995655e-02, 1.45746082e+00, 6.12984172e+00,
               6.45652619e+01, -7.89518256e+01],
              [-2.09824970e-01, 6.70576578e-01, 3.21424909e+00,
               3.21002650e+01, -3.90274136e+01],
              [8.09287244e-02, 1.01147330e-01, 1.62446095e+00,
               2.52004551e+01, -2.99899831e+01]],

             [[-8.38895314e-02, 6.70041670e-01, 4.45437011e+00,
               5.43438398e+01, -6.53627838e+01],
              [8.09287244e-02, 1.01147330e-01, 1.62446095e+00,
               2.52004551e+01, -2.99899831e+01],
              [-6.81487909e-02, 8.84966914e-02, 8.11585271e-01,
               1.96868504e+01, -2.32042652e+01]]]
        )
        np.testing.assert_almost_equal(
            dd_pp_o_dd, dd_pp_o_dd_exp, decimal=6)

    def test_calc_dd_p_o_dd_sq(self):
        dd_p_o_dd_sq = gev.calc_dd_p_o_dd_sq(self.x, self.p_hat)
        dd_p_o_dd_sq_exp = np.array(
            [[[1.11251118e-01, 1.30940948e+00, 5.84882869e+00,
               4.45900644e+01, 1.07816396e+02],
              [-1.07324608e-01, 5.90832092e-01, 3.05918275e+00,
               2.26616040e+01, 4.94686996e+01],
              [5.70473879e-02, -2.36882221e-02, 1.51133725e+00,
               1.72446122e+01, 3.41992383e+01]],

             [[-1.07324608e-01, 5.90832092e-01, 3.05918275e+00,
               2.26616040e+01, 4.94686996e+01],
              [1.03536681e-01, 2.66595412e-01, 1.60008090e+00,
               1.15171014e+01, 2.26974033e+01],
              [-5.50339506e-02, -1.06886058e-02, 7.90492778e-01,
               8.76407284e+00, 1.56914152e+01]],

             [[5.70473879e-02, -2.36882221e-02, 1.51133725e+00,
               1.72446122e+01, 3.41992383e+01],
              [-5.50339506e-02, -1.06886058e-02, 7.90492778e-01,
               8.76407284e+00, 1.56914152e+01],
              [2.92527798e-02, 4.28538112e-04, 3.90529524e-01,
               6.66912357e+00, 1.08479595e+01]]]
        )
        np.testing.assert_almost_equal(
            dd_p_o_dd_sq, dd_p_o_dd_sq_exp, decimal=6)

    def test_calc_l_pp_is(self):
        l_pp_is = gev.calc_l_pp_is(self.x, self.p_hat)
        l_pp_is_exp = np.array(
            [[[7.00501114e-02, 1.40105603e+00, 5.81147720e+00,
               8.55695250e+01, -2.66578750e+02],
              [1.28624174e-01, 8.66628729e-01, 3.07065898e+00,
               4.19036579e+01, -1.28420525e+02],
              [-1.40936919e-01, 6.93729892e-01, 2.94303286e+00,
               3.70992276e+01, -9.95620220e+01]],

             [[1.28624174e-01, 8.66628729e-01, 3.07065898e+00,
               4.19036579e+01, -1.28420525e+02],
              [-3.13361651e-01, 4.03981165e-01, 1.61416819e+00,
               2.05831636e+01, -6.17248169e+01],
              [1.35962675e-01, 1.11835936e-01, 8.33968175e-01,
               1.64363822e+01, -4.56813983e+01]],

             [[-1.40936919e-01, 6.93729892e-01, 2.94303286e+00,
               3.70992276e+01, -9.95620220e+01],
              [1.35962675e-01, 1.11835936e-01, 8.33968175e-01,
               1.64363822e+01, -4.56813983e+01],
              [-9.74015707e-02, 8.80681533e-02, 4.21055747e-01,
               1.30177269e+01, -3.40522246e+01]]]
        )
        np.testing.assert_almost_equal(
            l_pp_is, l_pp_is_exp, decimal=6)

    def test_calc_l_pp(self):
        l_pp = gev.calc_l_pp(self.x, self.p_hat)
        l_pp_exp = np.array(
            [[-173.72664203, -82.45095537, -58.96696857],
             [-82.45095537, -39.43686565, -28.16324933],
             [-58.96696857, -28.16324933, -20.62277544]]
        )
        np.testing.assert_almost_equal(l_pp, l_pp_exp, decimal=6)


if __name__ == '__main__':
    unittest.main()
