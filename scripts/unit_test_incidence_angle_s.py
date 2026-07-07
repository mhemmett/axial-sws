"""
Unit tests for calculate_incidence_angle_eigenvalue_jurkevics_s (S-wave eigenvalue/polarization
incidence angle, windowed on [S - 0.02s, S + T_dom]).

Named unit_test_* rather than test_* deliberately: scripts/test_*.py is Christian Baillard's
dormant legacy test suite (see README's "Files kept locally but not tracked"), and this is a
new, active test for splitting_functions.py.

The geometry tests mock swspy.splitting.get_dominant_period_baillard (the same welch-based
function SWSPy itself uses for its own Tmid) to a fixed T_dom, since spectral estimation on
short synthetic pulses is noisy and would make those tests flaky; a separate, unmocked smoke
test exercises the real T_dom-estimation-and-windowing path end to end.

Run with:
    python3 unit_test_incidence_angle_s.py

Note: this must insert the vendored swspy/ onto sys.path *before* importing splitting_functions,
exactly as the production notebooks do (see e.g. axial_splitting_mldd_AXEC2_batched.ipynb's
import cell) - otherwise a plain `import swspy` resolves to the upstream PyPI package (which is
also installed in the `seismo` env for unrelated reasons) instead of the vendored fork, and that
upstream copy doesn't even have get_dominant_period_baillard, silently breaking the T_dom
estimation this module relies on.
"""

import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
from obspy import Trace, UTCDateTime

_SWSPY_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swspy'))
if _SWSPY_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _SWSPY_LOCAL_PATH)
import swspy  # noqa: F401 - forces the vendored copy into sys.modules before splitting_functions imports it

from splitting_functions import calculate_incidence_angle_eigenvalue_jurkevics_s

FIXED_T_DOM = 0.15  # seconds, used by the mocked geometry tests


def _make_polarized_traces(direction, sampling_rate=100.0, duration_s=2.0,
                            s_arrival_offset=1.0, pulse_offset=0.05, pulse_width=0.02,
                            noise_amp=1e-6, seed=0):
    """
    Build synthetic Z/N/E obspy Traces whose particle motion is linearly polarized along
    <direction> (a 3-vector ordered [Z, N, E]), via a short Gaussian-enveloped pulse centered
    at s_arrival_offset + pulse_offset (i.e. inside [S - 0.02s, S + FIXED_T_DOM] for the
    mocked-T_dom tests), plus a small amount of isotropic noise so the covariance matrix
    isn't exactly singular.
    """
    rng = np.random.default_rng(seed)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    npts = int(duration_s * sampling_rate)
    t = np.arange(npts) / sampling_rate
    pulse_center = s_arrival_offset + pulse_offset
    signal = np.sin(2 * np.pi * 8.0 * t) * np.exp(-((t - pulse_center) ** 2) / (2 * pulse_width ** 2))

    data = np.outer(signal, direction) + rng.normal(scale=noise_amp, size=(npts, 3))

    starttime = UTCDateTime(0)
    traces = {}
    for i, comp in enumerate(['Z', 'N', 'E']):
        tr = Trace(data=data[:, i])
        tr.stats.sampling_rate = sampling_rate
        tr.stats.starttime = starttime
        tr.stats.channel = f'HH{comp}'
        traces[comp] = tr

    # calculate_incidence_angle_eigenvalue_jurkevics_s expects an absolute UTCDateTime for
    # s_arrival_offset (matching how the batch wrapper computes it), not a raw float offset.
    s_arrival_time = starttime + s_arrival_offset

    return traces['Z'], traces['N'], traces['E'], s_arrival_time


class TestSWaveEigenvalueIncidenceAngleGeometry(unittest.TestCase):
    """Geometry/formula correctness, with T_dom mocked to a fixed value so the window is
    deterministic and these tests aren't at the mercy of spectral-estimation noise.

    get_dominant_period_baillard returns (dominant_period_in_samples, dominant_freq), and the
    function under test converts back to seconds via `/ fs` (fs=100 for these synthetic
    traces), so returning (FIXED_T_DOM * 100, None) here makes the effective T_dom exactly
    FIXED_T_DOM regardless of which of the two calls (N or E component) is being answered.
    """

    def setUp(self):
        patcher = patch('swspy.splitting.get_dominant_period_baillard',
                         return_value=(FIXED_T_DOM * 100.0, None))
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_purely_horizontal_polarization_gives_zero_incidence(self):
        # Motion entirely in the N/E plane (Z=0) -> SV polarization perpendicular to a
        # vertical ray -> incidence should be ~0 degrees.
        trace_z, trace_n, trace_e, s_time = _make_polarized_traces([0.0, 1.0, 0.0])
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(trace_z, trace_n, trace_e, s_arrival_offset=s_time)
        self.assertFalse(np.isnan(inc))
        self.assertAlmostEqual(inc, 0.0, delta=2.0)

    def test_purely_vertical_polarization_gives_ninety_incidence(self):
        # Motion entirely along Z -> SV polarization perpendicular to a horizontal
        # (grazing) ray -> incidence should be ~90 degrees.
        trace_z, trace_n, trace_e, s_time = _make_polarized_traces([1.0, 0.0, 0.0])
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(trace_z, trace_n, trace_e, s_arrival_offset=s_time)
        self.assertFalse(np.isnan(inc))
        self.assertAlmostEqual(inc, 90.0, delta=2.0)

    def test_45_degree_polarization_gives_45_degree_incidence(self):
        # Equal Z and horizontal components -> sin(theta) = cos(theta) -> theta = 45 deg.
        trace_z, trace_n, trace_e, s_time = _make_polarized_traces([1.0, 1.0, 0.0])
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(trace_z, trace_n, trace_e, s_arrival_offset=s_time)
        self.assertFalse(np.isnan(inc))
        self.assertAlmostEqual(inc, 45.0, delta=2.0)

    def test_incidence_always_within_0_to_90_degrees(self):
        # Sweep many random polarization directions and confirm the returned incidence is
        # always in [0, 90] and matches the expected arcsin(|Z-component|) relationship.
        rng = np.random.default_rng(42)
        for trial in range(30):
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)

            trace_z, trace_n, trace_e, s_time = _make_polarized_traces(direction, seed=trial)
            inc = calculate_incidence_angle_eigenvalue_jurkevics_s(trace_z, trace_n, trace_e, s_arrival_offset=s_time)

            self.assertFalse(np.isnan(inc), f"trial {trial}: got NaN for direction {direction}")
            self.assertGreaterEqual(inc, 0.0, f"trial {trial}: incidence below 0 for direction {direction}")
            self.assertLessEqual(inc, 90.0, f"trial {trial}: incidence above 90 for direction {direction}")

            expected = math.degrees(math.asin(min(abs(direction[0]), 1.0)))
            self.assertAlmostEqual(inc, expected, delta=3.0,
                                    msg=f"trial {trial}: direction {direction}")

    def test_missing_data_returns_nan_not_exception(self):
        # An empty/degenerate window should fail gracefully (NaN), matching the P-wave
        # function's error-handling convention, not raise.
        trace_z, trace_n, trace_e, _ = _make_polarized_traces([0.0, 1.0, 0.0], duration_s=0.05)
        far_future = trace_z.stats.starttime + 5.0
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(trace_z, trace_n, trace_e, s_arrival_offset=far_future)
        self.assertTrue(np.isnan(inc))

    def test_window_uses_t_dom_not_fixed_duration(self):
        # A pulse placed just outside a short T_dom's window should be excluded (falls back
        # to the noise floor -> incidence dominated by noise, not the pulse direction), while
        # the same pulse placed within a longer T_dom's window should be picked up. This
        # confirms the window genuinely scales with the mocked T_dom rather than using some
        # other fixed duration.
        pulse_offset = 0.3  # outside FIXED_T_DOM=0.15, would have been outside the old 0.1s window too

        trace_z, trace_n, trace_e, s_time = _make_polarized_traces(
            [1.0, 0.0, 0.0], pulse_offset=pulse_offset, pulse_width=0.02, noise_amp=1e-6
        )

        with patch('swspy.splitting.get_dominant_period_baillard', return_value=(0.1 * 100.0, None)):
            inc_short_tdom = calculate_incidence_angle_eigenvalue_jurkevics_s(trace_z, trace_n, trace_e, s_arrival_offset=s_time)

        with patch('swspy.splitting.get_dominant_period_baillard', return_value=(0.5 * 100.0, None)):
            inc_long_tdom = calculate_incidence_angle_eigenvalue_jurkevics_s(trace_z, trace_n, trace_e, s_arrival_offset=s_time)

        # With the short T_dom, the window [S-0.02, S+0.1] misses the pulse at S+0.3 entirely,
        # so the eigenvector is driven by near-isotropic noise (angle far from the pulse's
        # true 90 degrees, and not reliably close to any fixed value). With the long T_dom,
        # the window [S-0.02, S+0.5] includes the pulse, so incidence should be ~90 degrees.
        self.assertFalse(np.isnan(inc_long_tdom))
        self.assertAlmostEqual(inc_long_tdom, 90.0, delta=2.0)
        self.assertNotAlmostEqual(inc_short_tdom, inc_long_tdom, delta=5.0)


class TestSWaveEigenvalueIncidenceAngleRealTDom(unittest.TestCase):
    """One unmocked end-to-end smoke test: real get_dominant_period_baillard + real windowing."""

    def test_real_t_dom_estimation_end_to_end(self):
        sampling_rate = 100.0
        duration_s = 4.0
        npts = int(duration_s * sampling_rate)
        t = np.arange(npts) / sampling_rate
        s_arrival_offset = 2.0

        # A clean low-frequency (5 Hz -> T_dom ~ 0.2s) carrier on N/E throughout the trace,
        # so get_dominant_period_baillard has real, unambiguous spectral content to lock onto,
        # plus a polarized pulse right at the S arrival for the incidence-angle geometry.
        carrier = np.sin(2 * np.pi * 5.0 * t)
        pulse = np.sin(2 * np.pi * 8.0 * t) * np.exp(-((t - (s_arrival_offset + 0.05)) ** 2) / (2 * 0.02 ** 2))

        direction = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)  # 45 degrees
        rng = np.random.default_rng(1)
        data = (
            np.outer(pulse, direction)
            + np.outer(carrier, [0.0, 1.0, 1.0]) * 0.3
            + rng.normal(scale=1e-6, size=(npts, 3))
        )

        starttime = UTCDateTime(0)
        traces = []
        for i, comp in enumerate(['Z', 'N', 'E']):
            tr = Trace(data=data[:, i])
            tr.stats.sampling_rate = sampling_rate
            tr.stats.starttime = starttime
            tr.stats.channel = f'HH{comp}'
            traces.append(tr)
        trace_z, trace_n, trace_e = traces

        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(
            trace_z, trace_n, trace_e, s_arrival_offset=starttime + s_arrival_offset
        )

        self.assertFalse(np.isnan(inc))
        self.assertGreaterEqual(inc, 0.0)
        self.assertLessEqual(inc, 90.0)


if __name__ == '__main__':
    unittest.main()
