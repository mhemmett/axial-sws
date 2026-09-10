"""Stage 4 splitting analysis for the AXEC2 agentified pilot.
Mirrors run_production_axec2_all_batches.py EXACTLY, on the 243 QC-passed events.
Writes RAW (pre-final-filter) results to sws_agent/axec2_pilot_splitting_raw.csv.
"""
import os, sys, warnings
import numpy as np
import pandas as pd
import obspy

warnings.filterwarnings('ignore')

REPO = "/Users/mhemmett/Seismology/axial-sws"
sys.path.insert(0, os.path.join(REPO, "swspy"))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import swspy  # noqa
from splitting_functions import perform_splitting_on_organized_waveforms

RAW_DATA_DIR = os.path.join(REPO, "scripts", "raw_axec2_all_batches_data")
QC_CSV = os.path.join(REPO, "sws_agent", "axec2_pilot_qc_passed.csv")
OUT_CSV = os.path.join(REPO, "sws_agent", "axec2_pilot_splitting_raw.csv")

# Production window/method params (verbatim from run_production_axec2_all_batches.py)
FIRST_WINDOW_START = 2
LAST_WINDOW_START = 1
FIRST_WINDOW_END = 1.5
LAST_WINDOW_END = 2.5
N_WIN = 7
S_PICK_UNCERTAINTY = 0.0395
INCIDENCE_FIELD = 'incidence_pykonal_s'


def build_organized_waveforms(df):
    organized = {}
    for _, row in df.iterrows():
        wf = row['waveform_file']
        path = os.path.join(RAW_DATA_DIR, wf)
        st = obspy.read(path)
        organized[row['event_id']] = {
            'traces': st,
            'station': row['station'],
            'datetime': row['datetime'],
            's_arrival_time': row['s_arrival_time'],
            'p_arrival_time': row['p_arrival_time'],
            'back_azimuth': row['back_azimuth'],
            'snr_horizontal': row['snr_horizontal'],
            'rectilinearity_jurkevics': row['rectilinearity_jurkevics'],
            'incidence_eigenvalue_jurkevics': row['incidence_p_jurkevics'],
            'incidence_pykonal_s': row['incidence_pykonal_s'],
            'magnitude': row['magnitude'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'depth': row['depth'],
        }
    return organized


def main():
    df = pd.read_csv(QC_CSV)
    print(f"QC-passed events in: {len(df)}")
    organized = build_organized_waveforms(df)
    print(f"Organized waveforms built: {len(organized)}")

    # Confirm the default the function would use if we did NOT pass incidence_field
    import inspect
    sig = inspect.signature(perform_splitting_on_organized_waveforms)
    print(f"FUNCTION DEFAULT incidence_field = {sig.parameters['incidence_field'].default!r}")
    print(f"FUNCTION DEFAULT cluster_eps     = {sig.parameters['cluster_eps'].default!r}")
    print(f"FUNCTION DEFAULT cluster_min_samples = {sig.parameters['cluster_min_samples'].default!r}")
    print(f"EXPLICIT arg incidence_field     = {INCIDENCE_FIELD!r}  (overrides default)")

    results = perform_splitting_on_organized_waveforms(
        organized, FIRST_WINDOW_START, LAST_WINDOW_START, FIRST_WINDOW_END, LAST_WINDOW_END,
        N_WIN, S_PICK_UNCERTAINTY, mode='swspy',
        coord_system='LQT', sws_method='EV_and_XC', incidence_field=INCIDENCE_FIELD,
        plot_results=False
    )

    rows = []
    for eid, ed in organized.items():
        r = results.get(eid)
        res = r['result'] if (r is not None and 'result' in r) else {}
        rows.append({
            'event_id': eid,
            'datetime': ed['datetime'],
            'phi': res.get('phi'),
            'dt': res.get('dt'),
            'phi_error': res.get('phi_error'),
            'dt_error': res.get('dt_error'),
            'quality': res.get('quality'),
            'success': res.get('success', False) if r is not None else False,
            'back_azimuth': ed['back_azimuth'],
            'incidence_pykonal_s': ed['incidence_pykonal_s'],
            'latitude': ed['latitude'],
            'longitude': ed['longitude'],
            'depth': ed['depth'],
        })
    out = pd.DataFrame(rows, columns=['event_id','datetime','phi','dt','phi_error','dt_error',
                                      'quality','success','back_azimuth','incidence_pykonal_s',
                                      'latitude','longitude','depth'])
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_CSV}")

    # ---- summary ----
    succ = out[out['success'] == True]
    fail = out[out['success'] != True]
    print(f"\n=== SUMMARY ===")
    print(f"in={len(out)}  success={len(succ)}  fail/null={len(fail)}")

    def circ_stats(deg):
        a = np.deg2rad(2 * np.asarray(deg, float))
        C, S = np.nanmean(np.cos(a)), np.nanmean(np.sin(a))
        mean = (np.rad2deg(np.arctan2(S, C)) / 2.0) % 180.0
        R = np.hypot(C, S)
        spread = np.rad2deg(np.sqrt(-2 * np.log(R))) / 2.0  # circular std in phi (deg)
        return mean, spread

    if len(succ):
        phi = succ['phi'].dropna().values % 180.0
        cm, cs = circ_stats(phi)
        dt = succ['dt'].dropna().values
        qw = succ['quality'].dropna().values
        print(f"phi (n={len(phi)}): circular mean={cm:.1f} deg, circular spread={cs:.1f} deg, "
              f"raw min/med/max={phi.min():.1f}/{np.median(phi):.1f}/{phi.max():.1f}")
        print(f"dt  (n={len(dt)}): min={dt.min():.4f} med={np.median(dt):.4f} max={dt.max():.4f} s")
        print(f"Q_w (n={len(qw)}): min={qw.min():.3f} med={np.median(qw):.3f} max={qw.max():.3f} mean={qw.mean():.3f}")
        for lo, hi in [(0.8,1.01),(0.5,0.8),(0.0,0.5)]:
            n = np.sum((qw>=lo)&(qw<hi))
            print(f"    Q_w [{lo:.1f},{hi if hi<=1 else 1.0:.1f}): {n}")

if __name__ == '__main__':
    main()
