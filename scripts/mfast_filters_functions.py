"""
MFAST-style per-event bandpass selection.

Standard production pipeline (build_raw_axec2_all_batches.py) applies one fixed
5-40 Hz bandpass to every event. This module instead tries several candidate
bandpass filters per event, computes the S-wave SNR for each with the same
Baillard absolute-value SNR windowing logic already used in production
(splitting_functions.compute_snr_for_event_baillard), and keeps whichever
filtered version of the trace maximizes S-wave SNR.

compute_snr_for_event_baillard always re-applies its own hardcoded 5-40 Hz
bandpass internally (its apply_filter parameter is unused/dead), so it can't
be called directly on a stream already filtered to a candidate band without
double-filtering. _s_wave_snr_prefiltered below is the same windowing/SNR
logic (dominant-period-adapted signal window, P-contamination clamp,
SNR_pick on E/N) with that internal filter step removed, since the caller is
expected to hand in a stream already filtered to the band being evaluated.
"""

import numpy as np
from obspy import UTCDateTime

from splitting_functions import stream2data, get_dominant_period_baillard, SNR_pick

FILTER_BANDS = [
    (1.0, 5.0),
    (1.0, 8.0),
    (1.0, 15.0),
    (1.0, 20.0),
    (3.0, 5.0),
    (2.0, 8.0),
    (3.0, 15.0),
    (3.0, 30.0),
    (5.0, 10.0),
    (5.0, 15.0),
    (5.0, 30.0),
    (5.0, 45.0),
    (10.0, 20.0),
    (10.0, 45.0),
]

FILTER_CORNERS = 2  # two-pole Butterworth


def _s_wave_snr_prefiltered(event_stream, event_row):
    """
    Same windowing/SNR logic as splitting_functions.compute_snr_for_event_baillard,
    but assumes event_stream has already been detrended/tapered/filtered to the
    band being evaluated (no internal filter call).

    Returns (snr, dom_period) - dom_period is the same Baillard dominant-period
    estimate (mean of E/N components over the fs_window) used internally to adapt
    the SNR signal window, returned here so it can be reported/passed through
    rather than thrown away.
    """
    if len(event_stream[0].data) != len(event_stream[1].data) or \
       len(event_stream[0].data) != len(event_stream[2].data) or \
       len(event_stream[1].data) != len(event_stream[2].data):
        return np.nan, np.nan

    s_window = [0.02, 0.3]
    fs_window = [0.1, 0.3]
    s_snr_window = [0.4, 0.2]

    st_x = event_stream.select(channel='??E')
    st_y = event_stream.select(channel='??N')
    st_xy = st_x + st_y

    if len(st_x) == len(st_y) and len(st_x) > 0:
        xy_array = stream2data(st_xy)
    else:
        return np.nan, np.nan

    p_time = UTCDateTime(event_row['datetime']) + float(event_row['p_arrival_time'])
    s_time = UTCDateTime(event_row['datetime']) + float(event_row['s_arrival_time'])
    fs_window_time = [s_time - fs_window[0], s_time + fs_window[1]]
    s_window_time = [s_time - s_window[0], s_time + s_window[1]]
    trace_start_time = st_xy[0].stats.starttime
    sampling_rate = event_stream[0].stats.sampling_rate

    s_samples = int(round((s_time - trace_start_time) * sampling_rate))
    p_samples = int(round((p_time - trace_start_time) * sampling_rate))
    fs_w1 = int(round((fs_window_time[0] - trace_start_time) * sampling_rate))
    fs_w2 = int(round((fs_window_time[1] - trace_start_time) * sampling_rate))
    mid_samples = int(round(p_samples + (s_samples - p_samples) / 2))
    sw1 = int(round((s_window_time[0] - trace_start_time) * sampling_rate))

    if fs_w1 < mid_samples:
        fs_w1 = mid_samples

    xy_array_dom = xy_array[fs_w1:fs_w2, :]
    if xy_array_dom.shape[0] < 2:
        return np.nan, np.nan

    (dom_period_x, dom_freq_x) = get_dominant_period_baillard(
        xy_array_dom[:, 0], sampling_rate, flag_plot=False)
    (dom_period_y, dom_freq_y) = get_dominant_period_baillard(
        xy_array_dom[:, 1], sampling_rate, flag_plot=False)

    if dom_freq_x <= 0 or dom_freq_y <= 0:
        return np.nan, np.nan

    dom_period = np.mean([dom_period_x, dom_period_y])
    if np.isnan(dom_period) or dom_period <= 0:
        return np.nan, np.nan

    sw2 = int(round(sw1 + 2 * dom_period))

    s_snr_window_time = [s_time - s_snr_window[0], s_time + s_snr_window[1]]
    s_snr_w1 = int(round((s_snr_window_time[0] - trace_start_time) * sampling_rate))
    s_snr_w2 = int(round((s_snr_window_time[1] - trace_start_time) * sampling_rate))

    if s_snr_w1 < mid_samples:
        s_snr_w1 = mid_samples
    if s_snr_w2 > sw2:
        s_snr_w2 = sw2

    if s_snr_w1 >= s_samples or s_snr_w2 <= s_samples:
        return np.nan, float(dom_period)

    snr_x = SNR_pick(xy_array[:, 0], s_samples, s_samples - s_snr_w1, s_snr_w2 - s_samples, mode='mean')
    snr_y = SNR_pick(xy_array[:, 1], s_samples, s_samples - s_snr_w1, s_snr_w2 - s_samples, mode='mean')

    return float(np.mean((snr_x, snr_y))), float(dom_period)


def try_filters(raw_stream, event_row, bands=FILTER_BANDS):
    """
    Try each candidate bandpass filter on a copy of the raw (unfiltered) event
    stream, compute S-wave SNR for each, and return the band that maximizes it.

    Parameters
    ----------
    raw_stream : obspy.Stream
        Unfiltered 3-component event stream (Z/N/E).
    event_row : dict-like
        Must have 'datetime', 'p_arrival_time', 's_arrival_time'.
    bands : list of (freqmin, freqmax) tuples

    Returns
    -------
    best_band : (freqmin, freqmax) or None if every band failed
    best_stream : obspy.Stream filtered with best_band (or None)
    best_snr : float (np.nan if every band failed)
    best_dom_period : float, the Baillard dominant-period estimate for best_band (np.nan if every band failed)
    snr_by_band : dict {(freqmin, freqmax): snr}
    """
    snr_by_band = {}
    dom_period_by_band = {}
    streams_by_band = {}
    for (freqmin, freqmax) in bands:
        st = raw_stream.copy()
        st.detrend("linear")
        st.taper(max_percentage=0.05, type='hann')
        st.filter('bandpass', freqmin=freqmin, freqmax=freqmax, corners=FILTER_CORNERS)
        snr, dom_period = _s_wave_snr_prefiltered(st, event_row)
        snr_by_band[(freqmin, freqmax)] = snr
        dom_period_by_band[(freqmin, freqmax)] = dom_period
        streams_by_band[(freqmin, freqmax)] = st

    valid = {band: snr for band, snr in snr_by_band.items() if not np.isnan(snr)}
    if not valid:
        return None, None, np.nan, np.nan, snr_by_band

    best_band = max(valid, key=valid.get)
    return best_band, streams_by_band[best_band], valid[best_band], dom_period_by_band[best_band], snr_by_band
