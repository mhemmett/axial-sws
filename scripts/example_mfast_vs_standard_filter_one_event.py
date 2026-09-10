"""
One-event comparison: standard fixed 5-40 Hz bandpass vs. the new mfast-like
try_filters band scan (5-10, 5-15, 5-20, 10-20, 10-25, 15-30 Hz), both using
the same Baillard S-wave SNR windowing logic.

Uses AXEC2 raw batch 1 (scripts/axial_mldd_2015_2021_axec2_batch_1.mseed),
first event in the catalog for that batch, purely for a side-by-side example
ahead of running the full mfast rebuild.
"""

import os
import sys
import warnings

import obspy
import pandas as pd
from obspy import UTCDateTime

warnings.filterwarnings('ignore')

_SWSPY_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swspy'))
if _SWSPY_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _SWSPY_LOCAL_PATH)
import swspy  # noqa: F401

from splitting_functions import create_extended_catalog, organize_stream_by_events, organize_waveform_data
from mfast_filters_functions import FILTER_BANDS, try_filters, _s_wave_snr_prefiltered

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(HERE, '..', 'data', 'mldd_catalog_2015_2021.csv')
STATION = 'AXEC2'
BATCH_SIZE = 250
BATCH_NUM = 1
MSEED_PATH = os.path.join(HERE, f'axial_mldd_2015_2021_axec2_batch_{BATCH_NUM}.mseed')


def main():
    catalog = pd.read_csv(CATALOG_FILE)
    station_catalog = catalog[catalog['station'] == STATION].copy().reset_index(drop=True)
    station_catalog['datetime'] = pd.to_datetime(station_catalog['event_datetime'], format='ISO8601')
    station_catalog['datetime'] = station_catalog['datetime'].apply(lambda x: UTCDateTime(x))
    station_catalog['p_time'] = station_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_p'], axis=1)
    station_catalog['s_time'] = station_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_s'], axis=1)

    start_idx = (BATCH_NUM - 1) * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    batch_catalog = station_catalog.iloc[start_idx:end_idx].copy()

    extended_catalog = create_extended_catalog(batch_catalog, pre_p_time=1.0, post_s_time=2.0)
    extended_catalog['id'] = extended_catalog.index
    extended_catalog = extended_catalog.rename(columns={
        'event_depth': 'dep', 'event_lat': 'lat', 'event_lon': 'lon', 'event_magnitude': 'mag'
    })

    stream = obspy.read(MSEED_PATH)
    event_streams = organize_stream_by_events(stream, extended_catalog)
    organized_waveforms = organize_waveform_data(event_streams, extended_catalog)

    # First event with a full, valid 3-component stream
    eid = None
    for candidate_eid, ed in organized_waveforms.items():
        st = ed['traces']
        if st is not None and len(st) == 3 and all(tr.stats.npts > 0 for tr in st):
            eid = candidate_eid
            break
    if eid is None:
        raise RuntimeError("No valid event found in batch 1")

    ed = organized_waveforms[eid]
    raw_stream = ed['traces']
    row = extended_catalog.loc[extended_catalog['id'] == eid].iloc[0]
    event_row = {
        'datetime': row['datetime'],
        'p_arrival_time': UTCDateTime(row['p_time']) - UTCDateTime(row['datetime']),
        's_arrival_time': UTCDateTime(row['s_time']) - UTCDateTime(row['datetime']),
    }

    print(f"Event id: {eid}, datetime: {row['datetime']}, station: {STATION}")
    print(f"P arrival: {event_row['p_arrival_time']:.3f} s, S arrival: {event_row['s_arrival_time']:.3f} s")
    print()

    # --- Standard method: fixed 5-40 Hz bandpass ---
    st_standard = raw_stream.copy()
    st_standard.detrend("linear")
    st_standard.taper(max_percentage=0.05, type='hann')
    st_standard.filter('bandpass', freqmin=5.0, freqmax=40.0)
    snr_standard, dom_period_standard = _s_wave_snr_prefiltered(st_standard, event_row)

    print("=== Standard method (fixed 5-40 Hz) ===")
    print(f"  Filter band: 5-40 Hz")
    print(f"  S-wave SNR: {snr_standard:.3f}")
    print(f"  Dominant period: {dom_period_standard}")
    print()

    # --- mfast-like method: try_filters band scan ---
    best_band, best_stream, best_snr, best_dom_period, snr_by_band = try_filters(raw_stream, event_row, FILTER_BANDS)

    print("=== mfast-like method (try_filters band scan) ===")
    for band, snr in snr_by_band.items():
        marker = "  <-- selected" if band == best_band else ""
        snr_str = f"{snr:.3f}" if snr == snr else "NaN"
        print(f"  {band[0]:>4.1f}-{band[1]:<4.1f} Hz : SNR = {snr_str}{marker}")
    print()
    print(f"  Selected band: {best_band[0]}-{best_band[1]} Hz, S-wave SNR: {best_snr:.3f}, Dominant period: {best_dom_period}")
    print()

    print("=== Comparison ===")
    print(f"  Standard (5-40 Hz)   SNR: {snr_standard:.3f}")
    print(f"  mfast-like ({best_band[0]}-{best_band[1]} Hz) SNR: {best_snr:.3f}")
    if best_snr == best_snr and snr_standard == snr_standard:
        print(f"  Improvement: {best_snr - snr_standard:+.3f} ({(best_snr / snr_standard - 1) * 100:+.1f}%)")


if __name__ == '__main__':
    main()
