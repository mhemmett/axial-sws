"""
Build mldd_catalog_2022_2026.csv by merging the MLDD-RT earthquake catalog
(mldd_realtime) with phase picks (hypoDD_nll_RT.pha).

Output columns match mldd_catalog_2015_2021.csv:
  station, arrival_time_p, event_datetime, arrival_time_s,
  event_lat, event_lon, event_depth, event_magnitude
"""

import re
from datetime import datetime, timezone, timedelta
import pandas as pd

TARGET_STATIONS = {"AXAS1", "AXAS2", "AXCC1", "AXEC1", "AXEC2", "AXEC3"}

CATALOG_FILE = "../data/mldd_realtime"
PHASE_FILE   = "../data/hypoDD_nll_RT.pha"
OUTPUT_FILE  = "../data/mldd_catalog_2022_2026.csv"


# ---------------------------------------------------------------------------
# 1. Parse the MLDD-RT catalog
# ---------------------------------------------------------------------------

def parse_catalog(path):
    events = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or not re.match(r"^\d{4}$", parts[0]):
                continue
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            hour, minute     = int(parts[3]), int(parts[4])
            sec_f            = float(parts[5])
            lat, lon, dep    = float(parts[6]), float(parts[7]), float(parts[8])
            mag              = float(parts[12])
            event_id         = int(parts[13])

            sec_int = int(sec_f)
            usec    = int(round((sec_f - sec_int) * 1e6))
            if sec_int >= 60:
                extra = sec_int // 60
                sec_int = sec_int % 60
                dt = datetime(year, month, day, hour, minute, sec_int, usec,
                              tzinfo=timezone.utc) + timedelta(minutes=extra)
            else:
                dt = datetime(year, month, day, hour, minute, sec_int, usec,
                              tzinfo=timezone.utc)

            events[event_id] = {
                "event_datetime": dt,
                "event_lat":      lat,
                "event_lon":      lon,
                "event_depth":    dep,
                "event_magnitude": mag,
            }
    print(f"Catalog: {len(events)} events parsed")
    return events


# ---------------------------------------------------------------------------
# 2. Parse the phase file
# ---------------------------------------------------------------------------

def strip_station_prefix(raw):
    """Remove leading network/location 'O' characters, keep station code."""
    return raw.lstrip("O")


def parse_phases(path, target_stations):
    """
    Returns dict: event_id -> {station -> {"P": float|None, "S": float|None}}
    Arrival times are travel times in seconds relative to origin.
    """
    phases = {}
    current_id = None

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            if line.startswith("#"):
                # Event header
                parts = line.split()
                current_id = int(parts[-1])
                phases[current_id] = {}

            elif current_id is not None:
                parts = line.split()
                if len(parts) < 4:
                    continue
                station = strip_station_prefix(parts[0])
                if station not in target_stations:
                    continue
                travel_time = float(parts[1])
                phase_type  = parts[3].upper()  # P or S

                if station not in phases[current_id]:
                    phases[current_id][station] = {"P": None, "S": None}
                # Keep first occurrence if duplicates exist
                if phases[current_id][station][phase_type] is None:
                    phases[current_id][station][phase_type] = travel_time

    n_events = sum(1 for v in phases.values() if v)
    print(f"Phase file: {len(phases)} event headers; {n_events} with target-station picks")
    return phases


# ---------------------------------------------------------------------------
# 3. Merge and build output DataFrame
# ---------------------------------------------------------------------------

def build_catalog(catalog, phases):
    rows = []
    matched = 0

    for event_id, cat in catalog.items():
        if event_id not in phases:
            continue
        matched += 1
        for station, picks in phases[event_id].items():
            rows.append({
                "station":           station,
                "arrival_time_p":    picks["P"],
                "event_datetime":    cat["event_datetime"],
                "arrival_time_s":    picks["S"],
                "event_lat":         cat["event_lat"],
                "event_lon":         cat["event_lon"],
                "event_depth":       cat["event_depth"],
                "event_magnitude":   cat["event_magnitude"],
            })

    print(f"Catalog events matched to phase file: {matched}")
    df = pd.DataFrame(rows, columns=[
        "station", "arrival_time_p", "event_datetime", "arrival_time_s",
        "event_lat", "event_lon", "event_depth", "event_magnitude",
    ])
    return df


# ---------------------------------------------------------------------------
# 4. Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    catalog = parse_catalog(CATALOG_FILE)
    phases  = parse_phases(PHASE_FILE, TARGET_STATIONS)
    df      = build_catalog(catalog, phases)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Total rows: {len(df)}")
    print(f"Stations: {sorted(df['station'].unique())}")
    print(f"Date range: {df['event_datetime'].min()} to {df['event_datetime'].max()}")
    print(f"\nSample rows:")
    print(df.head(10).to_string(index=False))
