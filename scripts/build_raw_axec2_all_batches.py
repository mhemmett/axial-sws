"""
Full-catalog raw QC rebuild for AXEC2, 2015-2021 (all 497 raw batches, ~124,144 events),
generalizing build_raw_axec2_batch1.py. Clean full recompute for every event - no reuse of
the old passing_waveforms_data_mldd_2015_2021_axec2 cache at all, since that cache's event
set was already filtered by the old (incorrect) P-wave incidence cut.

For each raw batch (axial_mldd_2015_2021_axec2_batch_<N>.mseed, N=1..497, ~250 events each):
  - organize waveforms, taper/detrend/bandpass (5-40 Hz)
  - compute SNR, rectilinearity, back-azimuth, P-Jurkevics incidence (reference only),
    PyKonal-FMM incidence (Eigenvalue-S is skipped - not needed for this production run,
    saves time across 124k events)
  - apply SNR>=2.0 and rectilinearity>=0.7 QC (no incidence pre-filter)
  - append to a single running metadata CSV + per-event mseed directory

Resumable: a marker file per batch (.done_batches/batch_<N>.done) lets this be safely
restarted without reprocessing already-completed batches.

Run with:
    python3 build_raw_axec2_all_batches.py [--start N] [--end M]
"""

import argparse
import glob
import os
import re
import sys
import time
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
)
from pykonal_raytracer import BaillardRayTracer
from baillard_raytraced_incidence import ll2xy

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(HERE, '..', 'data', 'mldd_catalog_2015_2021.csv')
STATIONS_FILE = os.path.join(HERE, '..', 'data', 'stations_axial.llz')
BATHY_FILE = os.path.join(HERE, '..', 'data', 'AXIAL_MODEL_3P_VELOCITY.S.mod.buf')
OUT_DIR = os.path.join(HERE, 'raw_axec2_all_batches_data')
OUT_WAVEFORMS_DIR = os.path.join(OUT_DIR, 'waveforms')
OUT_METADATA_CSV = os.path.join(OUT_DIR, 'raw_axec2_all_batches_metadata.csv')
DONE_MARKERS_DIR = os.path.join(OUT_DIR, '.done_batches')

STATION = 'AXEC2'
STA_LAT, STA_LON = 45.93967, -129.9738
BATCH_SIZE = 250


def find_raw_batches():
    files = glob.glob(os.path.join(HERE, 'axial_mldd_2015_2021_axec2_batch_*.mseed'))
    batches = []
    for f in files:
        m = re.search(r'batch_(\d+)\.mseed$', f)
        if m:
            batches.append((int(m.group(1)), f))
    return sorted(batches)


def process_batch(batch_num, mseed_path, catalog, tracer):
    start_idx = (batch_num - 1) * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    batch_catalog = catalog.iloc[start_idx:end_idx].copy()
    if len(batch_catalog) == 0:
        return 0

    extended_catalog = create_extended_catalog(batch_catalog, pre_p_time=1.0, post_s_time=2.0)
    extended_catalog['id'] = extended_catalog.index
    extended_catalog = extended_catalog.rename(columns={
        'event_depth': 'dep', 'event_lat': 'lat', 'event_lon': 'lon', 'event_magnitude': 'mag'
    })

    stream = obspy.read(mseed_path)
    event_streams = organize_stream_by_events(stream, extended_catalog)
    organized_waveforms = organize_waveform_data(event_streams, extended_catalog)

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

    if not organized_waveforms:
        return 0

    stations_df = pd.read_csv(STATIONS_FILE, delim_whitespace=True, header=None,
                               names=['Longitude (°W)', 'Latitude (°N)', 'Elevation (m)', 'Station ID'])
    stations_df['Elevation (m)'] = stations_df['Elevation (m)'] * 1000

    organized_waveforms = calculate_snr_for_organized_waveforms(organized_waveforms)
    organized_waveforms = calculate_back_azimuth_for_organized_waveforms(organized_waveforms, stations_df)
    organized_waveforms = calculate_rectilinearity_jurkevics_for_organized_waveforms(organized_waveforms)
    organized_waveforms = calculate_incidence_angle_eigenvalue_jurkevics_for_organized_waveforms(organized_waveforms)

    for eid, ed in organized_waveforms.items():
        ex, ey = ll2xy(ed['latitude'], ed['longitude'])
        depth = ed['depth']
        if depth < 0 or depth > 4.0:
            ed['incidence_pykonal_s'] = np.nan
        else:
            ed['incidence_pykonal_s'] = tracer.incidence_angle_at_station(STATION, float(ex), float(ey), float(depth))

    qc_passed = {}
    for eid, ed in organized_waveforms.items():
        snr_h = ed.get('snr_horizontal', np.nan)
        rect = ed.get('rectilinearity_jurkevics', np.nan)
        if np.isnan(snr_h) or np.isnan(rect):
            continue
        if snr_h < 2.0 or rect < 0.7:
            continue
        qc_passed[eid] = ed

    rows = []
    for eid, ed in qc_passed.items():
        waveform_filename = f"batch{batch_num}_event_{str(ed['datetime']).replace(':', '-')}.mseed"
        waveform_rel_path = os.path.join('waveforms', waveform_filename)
        ed['traces'].write(os.path.join(OUT_WAVEFORMS_DIR, waveform_filename), format='MSEED')
        rows.append({
            'batch': batch_num, 'event_id': f"{batch_num}_{eid}", 'waveform_file': waveform_rel_path, 'station': ed['station'],
            'datetime': str(ed['datetime']), 's_arrival_time': ed['s_arrival_time'], 'p_arrival_time': ed['p_arrival_time'],
            'latitude': ed['latitude'], 'longitude': ed['longitude'], 'depth': ed['depth'],
            'back_azimuth': ed.get('back_azimuth'), 'magnitude': ed.get('magnitude'),
            'snr_horizontal': ed.get('snr_horizontal'), 'rectilinearity_jurkevics': ed.get('rectilinearity_jurkevics'),
            'incidence_p_jurkevics': ed.get('incidence_eigenvalue_jurkevics'),
            'incidence_pykonal_s': ed.get('incidence_pykonal_s'),
        })

    if rows:
        header_needed = not os.path.exists(OUT_METADATA_CSV)
        pd.DataFrame(rows).to_csv(OUT_METADATA_CSV, mode='a', header=header_needed, index=False)

    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=497)
    args = parser.parse_args()

    os.makedirs(OUT_WAVEFORMS_DIR, exist_ok=True)
    os.makedirs(DONE_MARKERS_DIR, exist_ok=True)

    print("Loading full catalog...")
    catalog = pd.read_csv(CATALOG_FILE)
    station_catalog = catalog[catalog['station'] == STATION].copy().reset_index(drop=True)
    station_catalog['datetime'] = pd.to_datetime(station_catalog['event_datetime'], format='ISO8601')
    station_catalog['datetime'] = station_catalog['datetime'].apply(lambda x: UTCDateTime(x))
    station_catalog['p_time'] = station_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_p'], axis=1)
    station_catalog['s_time'] = station_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_s'], axis=1)
    print(f"Total {STATION} events in raw catalog: {len(station_catalog)}")

    print("Precomputing PyKonal FMM travel-time field for AXEC2...")
    tracer = BaillardRayTracer(bathy_file=BATHY_FILE)
    sx, sy = ll2xy(STA_LAT, STA_LON)
    tracer.precompute_station(STATION, float(sx), float(sy))

    batches = find_raw_batches()
    batches = [(n, f) for n, f in batches if args.start <= n <= args.end]
    print(f"Processing {len(batches)} batches ({args.start}-{args.end})...")

    t_start = time.time()
    total_passed = 0
    n_done = 0
    for batch_num, mseed_path in batches:
        marker = os.path.join(DONE_MARKERS_DIR, f'batch_{batch_num}.done')
        if os.path.exists(marker):
            n_done += 1
            continue

        t0 = time.time()
        n_passed = process_batch(batch_num, mseed_path, station_catalog, tracer)
        total_passed += n_passed
        n_done += 1

        with open(marker, 'w') as f:
            f.write(f"{n_passed} events passed, {time.time()-t0:.1f}s\n")

        elapsed = time.time() - t_start
        print(f"  batch {batch_num}: {n_passed} passed QC ({time.time()-t0:.1f}s) "
              f"[{n_done}/{len(batches)} done, {elapsed/60:.1f} min elapsed]")

    print(f"\nDone. Total events passed QC: {total_passed}. Metadata at {OUT_METADATA_CSV}")


if __name__ == '__main__':
    main()
