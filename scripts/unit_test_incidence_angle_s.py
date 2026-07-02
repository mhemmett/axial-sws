"""
Unit tests for calculate_incidence_angle_eigenvalue_jurkevics_s (S-wave eigenvalue/polarization
incidence angle).

Named unit_test_* rather than test_* deliberately: scripts/test_*.py is Christian Baillard's
dormant legacy test suite (see README's "Files kept locally but not tracked"), and this is a
new, active test for splitting_functions.py.

Run with:
    python3 unit_test_incidence_angle_s.py
"""

import math
import unittest

import numpy as np
from obspy import Trace, UTCDateTime

from splitting_functions import calculate_incidence_angle_eigenvalue_jurkevics_s


def _make_polarized_traces(direction, sampling_rate=100.0, duration_s=2.0,
                            s_arrival_offset=1.0, noise_amp=1e-6, seed=0):
    """
    Build synthetic Z/N/E obspy Traces whose particle motion is linearly polarized along
    <direction> (a 3-vector ordered [Z, N, E]) around the S arrival, plus a small amount of
    isotropic noise so the covariance matrix isn't exactly singular.
    """
    rng = np.random.default_rng(seed)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)

    npts = int(duration_s * sampling_rate)
    t = np.arange(npts) / sampling_rate
    signal = np.sin(2 * np.pi * 8.0 * t) * np.exp(-((t - s_arrival_offset) ** 2) / (2 * 0.05 ** 2))

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


class TestSWaveEigenvalueIncidenceAngle(unittest.TestCase):

    def test_purely_horizontal_polarization_gives_zero_incidence(self):
        # Motion entirely in the N/E plane (Z=0) -> SV polarization perpendicular to a
        # vertical ray -> incidence should be ~0 degrees.
        trace_z, trace_n, trace_e, s_time = _make_polarized_traces([0.0, 1.0, 0.0])
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(
            trace_z, trace_n, trace_e, s_arrival_offset=s_time, analysis_window=0.2, s_window_before=0.02
        )
        self.assertFalse(np.isnan(inc))
        self.assertAlmostEqual(inc, 0.0, delta=2.0)

    def test_purely_vertical_polarization_gives_ninety_incidence(self):
        # Motion entirely along Z -> SV polarization perpendicular to a horizontal
        # (grazing) ray -> incidence should be ~90 degrees.
        trace_z, trace_n, trace_e, s_time = _make_polarized_traces([1.0, 0.0, 0.0])
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(
            trace_z, trace_n, trace_e, s_arrival_offset=s_time, analysis_window=0.2, s_window_before=0.02
        )
        self.assertFalse(np.isnan(inc))
        self.assertAlmostEqual(inc, 90.0, delta=2.0)

    def test_45_degree_polarization_gives_45_degree_incidence(self):
        # Equal Z and horizontal components -> sin(theta) = cos(theta) -> theta = 45 deg.
        trace_z, trace_n, trace_e, s_time = _make_polarized_traces([1.0, 1.0, 0.0])
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(
            trace_z, trace_n, trace_e, s_arrival_offset=s_time, analysis_window=0.2, s_window_before=0.02
        )
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
            inc = calculate_incidence_angle_eigenvalue_jurkevics_s(
                trace_z, trace_n, trace_e, s_arrival_offset=s_time, analysis_window=0.2, s_window_before=0.02
            )

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
        inc = calculate_incidence_angle_eigenvalue_jurkevics_s(
            trace_z, trace_n, trace_e, s_arrival_offset=far_future, analysis_window=0.2, s_window_before=0.02
        )
        self.assertTrue(np.isnan(inc))


if __name__ == '__main__':
    unittest.main()
