"""Regression checks for three-axis spin, events and the nine-state filter."""
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from ekf import _cross, _rhs_and_jac, ekf_predict
from model import (DRAG_NODES, Physics, drag_coefficient, fit, flight_direction,
                   log_posterior, predict_positions, prior_spec, propagate, rhs,
                   sample_posterior, validate_input)


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.r0 = np.array([0., 0., .11])
        self.obs = np.array([[1.50, .45, .44], [2.96, .86, .74]])
        self.times = np.array([.05, .10])

    def test_drag_nodes_and_constant_endpoints(self):
        for profile, (speeds, coefficients) in DRAG_NODES.items():
            np.testing.assert_allclose([drag_coefficient(v, profile) for v in speeds], coefficients)
            self.assertEqual(drag_coefficient(100., profile), coefficients[-1])
            self.assertEqual(drag_coefficient(100., profile, derivative=True), 0.)

    def test_parallel_prior_shrinks_without_removing_dimension(self):
        d = flight_direction(self.r0, self.obs, self.times)
        p = prior_spec('wide', d)
        spin_transform = p['whitener'][3:, 3:]
        np.testing.assert_allclose(spin_transform @ d, d / p['spin_sd_parallel'])
        transverse = np.cross(d, [0, 0, 1])
        np.testing.assert_allclose(spin_transform @ transverse, transverse / p['spin_sd_transverse'], atol=1e-14)
        self.assertEqual(np.linalg.matrix_rank(spin_transform), 3)

    def test_likelihood_rejects_pre_observation_ground(self):
        p = prior_spec('wide', flight_direction(self.r0, self.obs, self.times))
        theta = [30., 0., -1., 0., 0., 0.]
        self.assertEqual(log_posterior(theta, self.r0, self.times, self.obs, p, 'gaussian'), -np.inf)
        with self.assertRaises(ValueError):
            predict_positions([5, 0, 1, 0, 0, 0], self.r0, [.3, .4], reject_ground=True)

    def test_gravity_analytic_nonuniform_timestamps(self):
        theta = np.array([30., 4., 7., 0., 0., 0.])
        times = np.array([.037, .093])
        positions = predict_positions(theta, self.r0, times, fixed_cd=0.)
        exact = self.r0 + times[:, None] * theta[:3]
        exact[:, 2] -= .5 * 9.81 * times**2
        np.testing.assert_allclose(positions, exact, atol=1e-10)
        crossing = propagate(theta, self.r0, fixed_cd=0.)
        t = 11 / 30
        self.assertEqual(crossing['status'], 'crossed')
        np.testing.assert_allclose([crossing['tg'], crossing['yg'], crossing['zg']],
                                   [t, 4*t, .11+7*t-.5*9.81*t*t], atol=1e-9)

    def test_ground_and_timeout_events(self):
        for theta, options, expected in (
            ([30, 0, 7, 0, 0, 0], {}, 'crossed'),
            ([30, 0, -.01, 0, 0, 0], {}, 'ground'),
            ([10, 0, 1, 0, 0, 0], {}, 'ground'),
            ([10, 0, 20, 0, 0, 0], {'max_time': .01}, 'timeout'),
        ):
            result = propagate(theta, self.r0, **options)
            self.assertEqual(result['status'], expected)
            self.assertEqual(np.isnan(result['tg']), expected != 'crossed')

    def test_drag_horizontal_ratio(self):
        result = propagate([30., 7., 7., 0., 0., 0.], self.r0)
        self.assertAlmostEqual(result['yg'] / 11, 7/30, places=10)

    def test_magnus_direction_and_spin_decay(self):
        physics = Physics(spin_decay=.37)
        wind = np.array([2., -1., 0.])
        state = np.r_[self.r0, [30., 7., 8., 10., -20., 30.]]
        u = state[3:6] - wind
        acceleration = rhs(0., state, physics, wind, fixed_cd=0.)[3:6]
        acceleration[2] += physics.g
        np.testing.assert_allclose(acceleration, physics.magnus * np.cross(state[6:], u))
        self.assertAlmostEqual(float(acceleration @ u), 0., places=10)
        solution = solve_ivp(rhs, (0., .4), state, args=(physics, wind), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(solution.y[6:, -1], state[6:] * np.exp(-.37 * .4), atol=1e-10)

    def test_ekf_jacobian_and_equation(self):
        physics = Physics()
        state = np.array([1., 2., 3., 16., 3., 1., 7., 8., -23.])
        wind = np.array([2., -1., 0.])
        f, jac = _rhs_and_jac(state, physics, wind)
        np.testing.assert_allclose(f, rhs(0, state, physics, wind))
        np.testing.assert_allclose(f, np.r_[state[3:6], f[3:6], -.10 * state[6:]])
        step = np.eye(9) * 1e-5
        numeric = np.column_stack([(rhs(0., state + delta, physics, wind)
                                    - rhs(0., state - delta, physics, wind)) / 2e-5 for delta in step])
        np.testing.assert_allclose(jac, numeric, atol=1e-8)

    def test_ekf_covariance_and_posterior_samples(self):
        result = ekf_predict(self.r0, self.times, self.obs, draws=24)
        self.assertEqual(result['states'].shape, (24, 9))
        self.assertEqual(result['predictions'].shape, (24, 3))
        cov = result['covariance']
        np.testing.assert_allclose(cov, cov.T)
        self.assertGreaterEqual(np.linalg.eigvalsh(cov).min(), -1e-12)
        self.assertGreater(np.linalg.norm(cov[3:6, 6:]), 0.)
        status, point = _cross(np.array([2., 0., .10, 30., 0., 5., 0., 0., 0.]),
                               .08, Physics(), 11.)
        self.assertEqual(status, 'ground')
        self.assertTrue(np.isnan(point).all())

    def test_invalid_times_rejected(self):
        for times in ([0., .1], [.1, .05], [.1, .1], [np.nan, .1]):
            with self.assertRaises(ValueError):
                validate_input(self.r0, times, self.obs)

    def test_exact_rounding_support(self):
        theta = np.array([31., 4., 7., 12., -15., 21.])
        times = np.array([.037, .093])
        exact = predict_positions(theta, self.r0, times)
        p = prior_spec('wide', flight_direction(self.r0, exact, times))
        self.assertTrue(np.isfinite(log_posterior(theta, self.r0, times, exact+.004, p, 'rounding')))
        self.assertEqual(log_posterior(theta, self.r0, times, exact+.006, p, 'rounding'), -np.inf)
        fitted = fit(self.r0, times, exact, noise='rounding')
        self.assertLessEqual(abs(fitted['residual_m']).max(), .0050001)

    def test_rounding_sampler_respects_all_six_intervals(self):
        result = sample_posterior(self.r0, self.times, self.obs, noise='rounding', draws=20, burn=0)
        for sample in result['samples']:
            residual = predict_positions(sample, self.r0, self.times) - self.obs
            self.assertLessEqual(np.abs(residual).max(), .005)

    def test_sampler_retains_exact_requested_count(self):
        result = sample_posterior(self.r0, self.times, self.obs, draws=20, burn=0)
        self.assertEqual(result['chains'].shape, (4, 5, 6))
        self.assertEqual(result['samples'].shape, (20, 6))
        self.assertEqual(result['diagnostics']['draws'], 20)


if __name__ == '__main__':
    unittest.main(verbosity=2)

