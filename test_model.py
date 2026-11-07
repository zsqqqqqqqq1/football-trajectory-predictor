"""Numerical and event regression checks independent of external datasets."""
import unittest
import numpy as np
from model import (Physics, basis_from_observations, predict_positions,
                   propagate, fit, rhs, validate_input, log_posterior)
from ekf import _rhs_and_jac, _cross, ekf_predict


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.r0 = np.array([0., 0., .11])
        self.obs = np.array([[1.50, .45, .44], [2.96, .86, .74]])
        self.b = basis_from_observations(self.r0, self.obs)

    def test_reference_basis(self):
        np.testing.assert_allclose(self.b.T @ self.b, np.eye(2), atol=1e-14)
        np.testing.assert_allclose((self.obs[-1]-self.r0) @ self.b, 0., atol=1e-14)

    def test_likelihood_rejects_pre_observation_ground(self):
        theta=np.array([30.,0.,-1.,.3,0.,0.])
        self.assertEqual(log_posterior(theta,self.r0,[.05,.1],self.obs,self.b,'gaussian','wide'),-np.inf)
        with self.assertRaises(ValueError):
            predict_positions([5,0,1,0,0,0],self.r0,[.3,.4],self.b,reject_ground=True)

    def test_gravity_analytic_nonuniform_timestamps(self):
        theta = np.array([30., 4., 7., 0., 0., 0.])
        times = np.array([.037, .093])
        positions = predict_positions(theta, self.r0, times, self.b)
        exact = self.r0 + times[:, None] * theta[:3]
        exact[:, 2] -= .5 * 9.81 * times**2
        np.testing.assert_allclose(positions, exact, atol=1e-10)
        crossing = propagate(theta, self.r0, self.b)
        t = 11 / 30
        self.assertEqual(crossing['status'], 'crossed')
        np.testing.assert_allclose([crossing['tg'], crossing['yg'], crossing['zg']],
                                   [t, 4*t, .11+7*t-.5*9.81*t*t], atol=1e-9)

    def test_ground_at_start_does_not_stop_upward_launch(self):
        crossed = propagate([30, 0, 7, 0, 0, 0], self.r0, self.b)
        self.assertEqual(crossed['status'], 'crossed')
        grounded = propagate([30, 0, -.01, 0, 0, 0], self.r0, self.b)
        self.assertEqual(grounded['status'], 'ground')
        self.assertTrue(np.isnan(grounded['tg']))

    def test_ground_before_plane(self):
        result = propagate([10, 0, 1, 0, 0, 0], self.r0, self.b)
        self.assertEqual(result['status'], 'ground')

    def test_timeout_separate_from_ground(self):
        result = propagate([10, 0, 20, 0, 0, 0], self.r0, self.b, max_time=.01)
        self.assertEqual(result['status'], 'timeout')

    def test_drag_horizontal_ratio(self):
        theta = [30., 7., 7., .3, 0., 0.]
        result = propagate(theta, self.r0, self.b)
        self.assertAlmostEqual(result['yg'] / 11, 7/30, places=10)

    def test_transverse_force_does_no_work(self):
        state = np.r_[self.r0, [30., 7., 8.]]
        accel = rhs(0, state, 0., np.array([.1, .2, -.3]))[3:]
        accel[2] += 9.81
        self.assertAlmostEqual(float(accel @ state[3:]), 0., places=10)

    def test_ekf_jacobian_and_equation(self):
        state = np.array([1., 2., 3., 30., 7., 8.])
        b = np.array([.1, .2, -.3])
        f, jac = _rhs_and_jac(state, .22, b, Physics())
        np.testing.assert_allclose(f, rhs(0, state, .22, b), atol=1e-12)
        numeric = np.column_stack([(_rhs_and_jac(state+np.eye(6)[i]*1e-5,.22,b,Physics())[0]
                                    -_rhs_and_jac(state-np.eye(6)[i]*1e-5,.22,b,Physics())[0])/2e-5
                                   for i in range(6)])
        np.testing.assert_allclose(jac, numeric, atol=1e-8)

    def test_ekf_posterior_state_below_surface(self):
        state = np.array([2., 0., .10, 30., 0., 5.])
        status, prediction = _cross(state, .08, .2, np.zeros(3), Physics(), 11.)
        self.assertEqual(status, 'ground')
        self.assertTrue(np.isnan(prediction).all())

    def test_invalid_times_rejected(self):
        for times in ([0., .1], [.1, .05], [.1, .1], [np.nan, .1]):
            with self.assertRaises(ValueError):
                validate_input(self.r0, times, self.obs)

    def test_noiseless_inverse_at_arbitrary_times(self):
        times = np.array([.037, .093])
        true_theta = np.array([31., 4., 7., .28, .05, -.1])
        b = self.b.copy()
        for _ in range(20):
            obs = predict_positions(true_theta, self.r0, times, b)
            b = basis_from_observations(self.r0, obs)
        obs = predict_positions(true_theta, self.r0, times, b)
        fitted = fit(self.r0, times, obs, regularize=False)
        np.testing.assert_allclose(fitted['theta'], true_theta, atol=2e-5)
        predicted = propagate(fitted['theta'], self.r0, fitted['B'])
        reference = propagate(true_theta, self.r0, b)
        np.testing.assert_allclose([predicted[k] for k in ('tg','yg','zg')],
                                   [reference[k] for k in ('tg','yg','zg')], atol=1e-6)

    def test_rounding_is_interval_likelihood(self):
        theta = np.array([31., 4., 7., .28, .05, -.1])
        times = [.04, .08]
        exact = predict_positions(theta, self.r0, times, self.b)
        self.assertTrue(np.isfinite(log_posterior(theta,self.r0,times,exact+.004,self.b,'rounding','wide')))
        self.assertEqual(log_posterior(theta,self.r0,times,exact+.006,self.b,'rounding','wide'), -np.inf)


if __name__ == '__main__':
    unittest.main(verbosity=2)
