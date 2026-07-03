"""
Build organized_waveforms for AXAS2 raw batch 1 (the first 250 catalog events, pre-QC
entirely) - AXAS2 counterpart to build_raw_axec2_batch1.py. Source:
  - data/mldd_catalog_2015_2021.csv (full catalog)
  - scripts/axial_mldd_2015_2021_axas2_batch_1.mseed (raw, unfiltered, fetched via
    fetch_raw_axas2_batch1.py since no pre-existing cache existed for AXAS2)

SNR>=2.0 and rectilinearity>=0.7 QC applied; no incidence pre-filter. QC metrics computed
fresh: SNR, rectilinearity, P-Jurkevics incidence (kept for reference), Eigenvalue-S
incidence, and PyKonal-FMM incidence.

Run with:
    python3 fetch_raw_axas2_batch1.py   # once, to pull raw waveforms from IRIS
    python3 build_raw_axas2_batch1.py
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime

warnings.filterwarnings('ignore')

_SWSPY_LOCAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'swspy'))
if _SWSPY_LOCAL_PATH not in sys.path:
    sys.path.insert(0, _SWSPY_LOCAL_PATH)
import swspy  # noqa: F401

from splitting_functions import (
    create_extended_catalog, organize_stream_by_events, organize_waveform_data,
    calculate_snr_for_organized_waveforms, calculate_back_azimuth_for_organized_waveforms,
    calculate_rectilinearity_jurkevics_for_organized_waveforms,
    calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms,
    calculate_incidence_angle_eigenvalue_jurkevics_s_for_organized_waveforms,
)
from pykonal_raytracer import BaillardRayTracer
from baillard_raytraced_incidence import ll2xy

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(HERE, '..', 'data', 'mldd_catalog_2015_2021.csv')
STATIONS_FILE = os.path.join(HERE, '..', 'data', 'stations_axial.llz')
BATCH1_MSEED = os.path.join(HERE, 'axial_mldd_2015_2021_axas2_batch_1.mseed')
BATHY_FILE = os.path.join(HERE, '..', 'data', 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')
OUT_DIR = os.path.join(HERE, 'raw_axas2_batch1_data')
OUT_WAVEFORMS_DIR = os.path.join(OUT_DIR, 'waveforms')
OUT_METADATA_CSV = os.path.join(OUT_DIR, 'raw_axas2_batch1_metadata.csv')

STATION = 'AXAS2'
STA_LAT, STA_LON = 45.93377, -130.0141  # data/stations_axial.llz
BATCH_SIZE = 250


def main():
    print("Loading catalog...")
    catalog = pd.read_csv(CATALOG_FILE)
    station_catalog = catalog[catalog['station'] == STATION].copy().reset_index(drop=True)
    station_catalog['datetime'] = pd.to_datetime(station_catalog['event_datetime'], format='ISO8601')
    station_catalog['datetime'] = station_catalog['datetime'].apply(lambda x: UTCDateTime(x))
    station_catalog['p_time'] = station_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_p'], axis=1)
    station_catalog['s_time'] = station_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_s'], axis=1)
    station_catalog['event_id'] = station_catalog.index
    print(f"Total {STATION} events in raw catalog: {len(station_catalog)}")

    batch1_catalog = station_catalog.iloc[:BATCH_SIZE].copy()
    print(f"Raw batch 1: {len(batch1_catalog)} events "
          f"({batch1_catalog['datetime'].iloc[0]} to {batch1_catalog['datetime'].iloc[-1]})")

    extended_catalog = create_extended_catalog(batch1_catalog, pre_p_time=1.0, post_s_time=2.0)
    extended_catalog['id'] = extended_catalog.index
    extended_catalog = extended_catalog.rename(columns={
        'event_depth': 'dep', 'event_lat': 'lat', 'event_lon': 'lon', 'event_magnitude': 'mag'
    })

    print("Loading raw batch 1 waveforms...")
    stream = obspy.read(BATCH1_MSEED)
    print(f"  {len(stream)} traces loaded")

    print("Organizing waveforms by event...")
    event_streams = organize_stream_by_events(stream, extended_catalog)
    organized_waveforms = organize_waveform_data(event_streams, extended_catalog)
    print(f"  {len(organized_waveforms)} events organized")

    print("Tapering, detrending, and bandpass filtering (5-40 Hz) all traces...")
    events_to_remove = []
    for eid in organized_waveforms.keys():
        st = organized_waveforms[eid]['traces']
        if st is None:
            events_to_remove.append(eid)
            continue
        for tr in st:
            tr.detrend("linear")
            tr.taper(max_percentage=0.05, type='hann')
            tr.filter('bandpass', freqmin=5.0, freqmax=40.0)
        organized_waveforms[eid]['s_arrival_time'] = (
            UTCDateTime(extended_catalog.loc[extended_catalog['id'] == eid, 's_time'].values[0])
            - UTCDateTime(extended_catalog.loc[extended_catalog['id'] == eid, 'datetime'].values[0])
        )
        organized_waveforms[eid]['p_arrival_time'] = (
            UTCDateTime(extended_catalog.loc[extended_catalog['id'] == eid, 'p_time'].values[0])
            - UTCDateTime(extended_catalog.loc[extended_catalog['id'] == eid, 'datetime'].values[0])
        )
    for eid in events_to_remove:
        del organized_waveforms[eid]
    print(f"  Removed {len(events_to_remove)} events with no traces")

    # Remove duplicate traces / events without exactly 3 traces
    events_to_remove = []
    for eid, ed in organized_waveforms.items():
        st = ed['traces']
        if len(st) > 3:
            unique_traces = {}
            for tr in st:
                if tr.stats.channel not in unique_traces:
                    unique_traces[tr.stats.channel] = tr
            organized_waveforms[eid]['traces'] = obspy.Stream(traces=list(unique_traces.values()))
        st = organized_waveforms[eid]['traces']
        if len(st) != 3 or any(tr.stats.npts == 0 for tr in st):
            events_to_remove.append(eid)
    for eid in events_to_remove:
        del organized_waveforms[eid]
    print(f"  Removed {len(events_to_remove)} events without exactly 3 valid-length traces")
    print(f"  {len(organized_waveforms)} events remain")

    stations_df = pd.read_csv(STATIONS_FILE, delim_whitespace=True, header=None,
                               names=['Longitude (°W)', 'Latitude (°N)', 'Elevation (m)', 'Station ID'])
    stations_df['Elevation (m)'] = stations_df['Elevation (m)'] * 1000

    print("Computing SNR...")
    organized_waveforms = calculate_snr_for_organized_waveforms(organized_waveforms)
    print("Computing back-azimuth...")
    organized_waveforms = calculate_back_azimuth_for_organized_waveforms(organized_waveforms, stations_df)
    print("Computing P-wave rectilinearity (Jurkevics)...")
    organized_waveforms = calculate_rectilinearity_jurkevics_for_organized_waveforms(organized_waveforms)
    print("Computing P-wave incidence (Jurkevics, kept for reference only)...")
    organized_waveforms = calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms(organized_waveforms)
    print("Computing Eigenvalue-S incidence (this session's method)...")
    organized_waveforms = calculate_incidence_angle_eigenvalue_jurkevics_s_for_organized_waveforms(organized_waveforms)

    print("Computing PyKonal-FMM incidence (Baillard 3D S-velocity model)...")
    tracer = BaillardRayTracer(bathy_file=BATHY_FILE)
    sx, sy = ll2xy(STA_LAT, STA_LON)
    tracer.precompute_station(STATION, float(sx), float(sy))
    for eid, ed in organized_waveforms.items():
        ex, ey = ll2xy(ed['latitude'], ed['longitude'])
        depth = ed['depth']
        if depth < 0 or depth > 4.0:
            ed['incidence_pykonal_s'] = np.nan
        else:
            ed['incidence_pykonal_s'] = tracer.incidence_angle_at_station(STATION, float(ex), float(ey), float(depth))

    print("\nApplying SNR>=2.0 and rectilinearity>=0.7 QC (no incidence cut)...")
    qc_passed = {}
    n_fail_snr, n_fail_rect, n_fail_missing = 0, 0, 0
    for eid, ed in organized_waveforms.items():
        snr_h = ed.get('snr_horizontal', np.nan)
        rect = ed.get('rectilinearity_jurkevics', np.nan)
        if np.isnan(snr_h) or np.isnan(rect):
            n_fail_missing += 1
            continue
        if snr_h < 2.0:
            n_fail_snr += 1
            continue
        if rect < 0.7:
            n_fail_rect += 1
            continue
        qc_passed[eid] = ed
    print(f"  Passed: {len(qc_passed)} / {len(organized_waveforms)} "
          f"(failed: missing={n_fail_missing}, snr={n_fail_snr}, rectilinearity={n_fail_rect})")

    os.makedirs(OUT_WAVEFORMS_DIR, exist_ok=True)
    rows = []
    for eid, ed in qc_passed.items():
        waveform_filename = f"event_{str(ed['datetime']).replace(':', '-')}.mseed"
        waveform_rel_path = os.path.join(os.path.basename(OUT_WAVEFORMS_DIR), waveform_filename)
        ed['traces'].write(os.path.join(OUT_WAVEFORMS_DIR, waveform_filename), format='MSEED')
        rows.append({
            'event_id': eid, 'waveform_file': waveform_rel_path, 'station': ed['station'],
            'datetime': str(ed['datetime']), 's_arrival_time': ed['s_arrival_time'], 'p_arrival_time': ed['p_arrival_time'],
            'latitude': ed['latitude'], 'longitude': ed['longitude'], 'depth': ed['depth'],
            'back_azimuth': ed.get('back_azimuth'), 'magnitude': ed.get('magnitude'),
            'snr_horizontal': ed.get('snr_horizontal'), 'rectilinearity_jurkevics': ed.get('rectilinearity_jurkevics'),
            'incidence_p_jurkevics': ed.get('incidence_eigenvalue_jurkevics'),
            'incidence_eigenvalue_s': ed.get('incidence_eigenvalue_jurkevics_s'),
            'incidence_pykonal_s': ed.get('incidence_pykonal_s'),
        })
    pd.DataFrame(rows).to_csv(OUT_METADATA_CSV, index=False)
    print(f"Saved {len(rows)} events: metadata to {OUT_METADATA_CSV}, waveforms to {OUT_WAVEFORMS_DIR}/")


if __name__ == '__main__':
    main()
