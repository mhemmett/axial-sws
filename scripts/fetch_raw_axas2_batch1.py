"""
Retrieve raw batch 1 (first 250 catalog events) for AXAS2 from IRIS, matching the same
convention as the pre-existing axial_mldd_2015_2021_axec2_batch_1.mseed for AXEC2 (which
was already cached; no equivalent exists for AXAS2, so this fetches it fresh).

Run with:
    python3 fetch_raw_axas2_batch1.py
"""

import os
import pandas as pd
from obspy import UTCDateTime

from get_all_traces import get_station_traces_batch

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG_FILE = os.path.join(HERE, '..', 'data', 'mldd_catalog_2015_2021.csv')
OUT_MSEED = os.path.join(HERE, 'axial_mldd_2015_2021_axas2_batch_1')  # .mseed appended by writer

STATION = 'AXAS2'
BATCH_SIZE = 250


def main():
    print("Loading catalog...")
    catalog = pd.read_csv(CATALOG_FILE)
    axas2_catalog = catalog[catalog['station'] == STATION].copy().reset_index(drop=True)
    axas2_catalog['datetime'] = pd.to_datetime(axas2_catalog['event_datetime'], format='ISO8601')
    axas2_catalog['datetime'] = axas2_catalog['datetime'].apply(lambda x: UTCDateTime(x))
    axas2_catalog['p_time'] = axas2_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_p'], axis=1)
    axas2_catalog['s_time'] = axas2_catalog.apply(lambda row: UTCDateTime(row['datetime']) + row['arrival_time_s'], axis=1)
    print(f"Total {STATION} events in raw catalog: {len(axas2_catalog)}")

    batch1_catalog = axas2_catalog.iloc[:BATCH_SIZE].copy()
    print(f"Batch 1: {len(batch1_catalog)} events "
          f"({batch1_catalog['datetime'].iloc[0]} to {batch1_catalog['datetime'].iloc[-1]})")

    # Extended windows [P-1s, S+2s], matching create_extended_catalog's defaults used elsewhere
    batch1_catalog['starttime'] = batch1_catalog['p_time'].apply(lambda t: t - 1.0)
    batch1_catalog['endtime'] = batch1_catalog['s_time'].apply(lambda t: t + 2.0)
    batch1_catalog['station_col'] = STATION

    print(f"Retrieving waveforms from IRIS for {len(batch1_catalog)} events...")
    traces = get_station_traces_batch(batch1_catalog, OUT_MSEED, 'starttime', 'endtime', 'station_col', batch_size=BATCH_SIZE)
    print(f"Retrieved {len(traces)} traces, saved to {OUT_MSEED}.mseed")


if __name__ == '__main__':
    main()
