#!/usr/bin/env python3
"""
2x2 baz/dt-weighted fast-direction rose plots for north array stations AX01-AX14.

Layout per station (split at 2023-09-01):
  Row 0: 2022  |  2023 Jan–Sep 1
  Row 1: 2023 Sep 1–Dec 31  |  2024

Each panel: central dt-weighted rose + 6 back-azimuth sub-roses in a ring.
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from obspy.core.utcdatetime import UTCDateTime

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = "../results"
OUTPUT_DIR  = "../results"
STATIONS    = [f"ax{i:02d}" for i in range(1, 15)]

SPLIT = pd.Timestamp("2023-09-01T00:00:00")


def _fill_rose_dt(ax, phi_values, dt_values, nbins, color, edgecolor, linewidth, spine_lw=None):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor("none")
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.yaxis.set_visible(False)
    ax.spines["polar"].set_visible(True)
    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(spine_lw if spine_lw else linewidth)
    ax.grid(False)

    if len(phi_values) == 0:
        ax.set_ylim(0, 1)
        return

    doubled_angles  = []
    doubled_weights = []
    for phi, w in zip(phi_values, dt_values):
        p = phi % 360
        doubled_angles.extend([np.deg2rad(p), np.deg2rad((p + 180) % 360)])
        doubled_weights.extend([w, w])
    doubled_angles  = np.array(doubled_angles)
    doubled_weights = np.array(doubled_weights)

    bins = np.linspace(0, 2 * np.pi, nbins + 1)
    counts, edges = np.histogram(doubled_angles, bins=bins, weights=doubled_weights)
    centers = (edges[:-1] + edges[1:]) / 2
    width   = 2 * np.pi / nbins

    ax.bar(centers, counts, width=width, bottom=0,
           color=color, edgecolor=edgecolor, linewidth=linewidth, alpha=1.0)
    ax.set_ylim(0, counts.max() * 1.2 if counts.max() > 0 else 1)


def plot_2x2_rose(results_df, station_label,
                  output_filename=None,
                  nbins=36, edgecolor="black", linewidth=1.5,
                  figsize=None,
                  time_column="event_datetime"):

    def _to_dt(x):
        if isinstance(x, UTCDateTime):
            return x.datetime
        ts = pd.Timestamp(x)
        if ts.tzinfo is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        return ts.to_pydatetime()

    df = results_df.copy()
    df["_dt"]   = pd.to_datetime(df[time_column].apply(_to_dt), utc=False)
    df["_year"] = df["_dt"].dt.year

    d2022 = df[df["_year"] == 2022]
    d2023 = df[df["_year"] == 2023]
    d2024 = df[df["_year"] == 2024]

    panels = [
        (f"2022",                d2022),
        (f"2023 Jan–Sep 1",      d2023[d2023["_dt"] < SPLIT]),
        (f"2023 Sep 1–Dec 31",   d2023[d2023["_dt"] >= SPLIT]),
        (f"2024",                d2024),
    ]

    _rgb_start = np.array(mcolors.to_rgb("#ADD8E6"))  # light blue
    _rgb_end   = np.array(mcolors.to_rgb("#800080"))  # purple
    panel_colors = [
        mcolors.to_hex(_rgb_start),
        mcolors.to_hex(0.33 * _rgb_end + 0.67 * _rgb_start),
        mcolors.to_hex(0.67 * _rgb_end + 0.33 * _rgb_start),
        mcolors.to_hex(_rgb_end),
    ]

    NCOLS = 2
    NROWS = 2

    PANEL_W_IN    = 12
    PANEL_H_IN    = 16
    FIG_W         = PANEL_W_IN * NCOLS if figsize is None else figsize[0]
    FIG_H         = PANEL_H_IN * NROWS if figsize is None else figsize[1]
    PANEL_W       = FIG_W / NCOLS
    PANEL_H       = FIG_H / NROWS
    PANEL_SCALE   = min(PANEL_W, PANEL_H)
    MAIN_ROSE_IN  = PANEL_SCALE * 0.38
    SMALL_ROSE_IN = PANEL_SCALE * 0.16
    RING_R_IN     = PANEL_SCALE * 0.37
    FS_MAIN       = max(10, int(PANEL_H * 1.2))
    FS_SMALL      = max(7,  int(PANEL_H * 0.75))
    FS_SUPTITLE   = max(20, int(PANEL_H * 2.2))

    baz_bins = np.arange(0, 361, 60)
    baz_ctrs = (baz_bins[:-1] + baz_bins[1:]) / 2

    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor="white")
    fig.suptitle(f"{station_label.upper()} — dt-weighted fast direction",
                 fontsize=FS_SUPTITLE, fontweight="bold", y=1.01)

    for i_panel, (label, df_p) in enumerate(panels):
        col   = i_panel % NCOLS
        row   = i_panel // NCOLS
        cx_in = (col + 0.5) * PANEL_W
        cy_in = FIG_H - (row + 0.5) * PANEL_H

        color = panel_colors[i_panel]
        df_p  = df_p.dropna(subset=["phi", "dt"])

        ax_main = fig.add_axes(
            [(cx_in - MAIN_ROSE_IN / 2) / FIG_W,
             (cy_in - MAIN_ROSE_IN / 2) / FIG_H,
             MAIN_ROSE_IN / FIG_W,
             MAIN_ROSE_IN / FIG_H],
            projection="polar",
        )
        _fill_rose_dt(ax_main,
                      df_p["phi"].values, np.abs(df_p["dt"].values),
                      nbins=nbins, color=color, edgecolor=edgecolor,
                      linewidth=linewidth, spine_lw=linewidth * 1.2)
        ax_main.set_title(f"{label}\nN={len(df_p)}",
                          va="bottom", fontsize=FS_MAIN, fontweight="bold", pad=14)

        for baz_lo, baz_hi, baz_c in zip(baz_bins[:-1], baz_bins[1:], baz_ctrs):
            baz_rad = np.deg2rad(baz_c)
            sc_x_in = cx_in + RING_R_IN * np.sin(baz_rad)
            sc_y_in = cy_in + RING_R_IN * np.cos(baz_rad)

            ax_s = fig.add_axes(
                [(sc_x_in - SMALL_ROSE_IN / 2) / FIG_W,
                 (sc_y_in - SMALL_ROSE_IN / 2) / FIG_H,
                 SMALL_ROSE_IN / FIG_W,
                 SMALL_ROSE_IN / FIG_H],
                projection="polar",
            )
            baz_mask = (df_p["back_azimuth"] >= baz_lo) & (df_p["back_azimuth"] < baz_hi)
            df_bin   = df_p.loc[baz_mask].dropna(subset=["phi", "dt"])
            _fill_rose_dt(ax_s,
                          df_bin["phi"].values, np.abs(df_bin["dt"].values),
                          nbins=18, color=color, edgecolor=edgecolor,
                          linewidth=0.6, spine_lw=0.8)
            ax_s.set_title(f"{int(baz_lo)}-{int(baz_hi)}°\nN={len(df_bin)}",
                           va="bottom", fontsize=FS_SMALL, pad=5)

    fig.patch.set_alpha(0.0)
    if output_filename:
        fig.savefig(output_filename, dpi=200, bbox_inches="tight", transparent=True)
        print(f"  Saved: {output_filename}")
    return fig


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for sta in STATIONS:
        csv_path = os.path.join(RESULTS_DIR, f"splitting_results_north_array_{sta}_all_batches.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {sta}: {csv_path} not found")
            continue

        print(f"Processing {sta}...")
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["phi", "dt", "back_azimuth"])

        out = os.path.join(OUTPUT_DIR, f"rose_2x2_north_array_{sta}_baz_dt_weighted.png")
        fig = plot_2x2_rose(df, station_label=sta, output_filename=out)
        plt.close(fig)

    print("Done.")
