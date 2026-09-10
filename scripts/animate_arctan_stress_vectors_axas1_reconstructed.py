#!/usr/bin/env python3
"""
animate_arctan_stress_vectors_axas1_reconstructed.py

AXAS1-only companion to animate_arctan_stress_vectors.py -- per explicit user request, this is
the ONLY gif produced for AXAS1 (the main script handles AXEC2/AXCC1 only). Per the later
explicit user request "use these functions to remake our gifs," this now fits the SAME atan2
vector-sum model (fit_atan2_vectorsum, imported from animate_arctan_stress_vectors.py, which
itself is the same model as atan2_uplift_vs_phi_axec2_axcc1_axas1_90day.py) instead of the
earlier plain-arctan-at-zero form -- see animate_arctan_stress_vectors.py's module docstring
for the full model/history and why the vector diagram no longer needs per-frame root-finding
or a running-max monotonicity patch (infl_mag(u) is exactly linear in u by construction).

AXAS1-specific: fit on the real (post-2017) data only (u0=0 fixed -- AXAS1's real record
starts right at its turnover), then invert_atan2_vectorsum reconstructs 2015-2017 from the
real SWS phi measurements that already exist for that gap (the windowcheck catalog has no
gap, only the uplift record does), bounded to u_bound=1.1 (per explicit user assumption that
the reconstructed domain begins around u_z=-1.1 m, matching this script's own earlier
plain-arctan reconstruction).

Produces:
    arctan_stress_vectors_AXAS1_reconstructed.gif

Run with:
    python3 animate_arctan_stress_vectors_axas1_reconstructed.py
"""

import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

from rose_7period_regions_windowcheck_grade3 import (
    GRADE, load_station_raw, apply_grade, _circular_mean_and_se_deg,
)
from rose_7period_6stations_newdata_snr_grades import _draw_rose
from axec2_uplift_phi_cosine_vs_time import (
    rolling_phi_stats_daily_then_roll, ERUPTION_START, ERUPTION_END,
)
from arctan_uplift_vs_phi_axec2_axcc1_axas1_90day import STATIONS, UPLIFT_ROLLING_DAYS
from animate_arctan_stress_vectors import (
    ALPHA_FIXED_AZ_DEG, ALPHA_FIXED_RAD, U_Z0, _azimuth, EXTENSION_SHIFT_DEG,
    compute_optimal_wrap, fit_atan2_vectorsum, invert_atan2_vectorsum, break_wrapped_line,
    compute_full_range_ticks, load_station_xy, load_bathy_gray, phi_window, draw_rose_mean_line,
    TARGET_MAX_EXTENT_KM, ROSE_WINDOW_DAYS, COLOR_ROSE,
    COLOR_BG, COLOR_INFL, COLOR_COMBINED,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, 'arctan_stress_vectors_AXAS1_reconstructed.gif')

ROLL_WINDOW_DAYS = 90
ROLL_MIN_DAYS = 9

N_FRAMES = 90
FPS = 15

COLOR_RECON = '#2ca02c'

# Per explicit user assumption: the reconstructed AXAS1 domain begins at u_z=-1.1 m (matching
# this script's own earlier plain-arctan reconstruction, which found -1.097 to 0.967 m) --
# bounds invert_atan2_vectorsum's search so it can't wander to an implausible u_z (an earlier,
# more permissive bound let it reach u_z=-63 m for the atan2 PDF script's own AXAS1 page).
U_BOUND = 1.1


def main():
    station = next(s for s in STATIONS if s['name'] == 'AXAS1')
    name = station['name']

    print(f'Loading {name}...')
    raw = load_station_raw(name)
    df = apply_grade(raw, GRADE)

    pre = df[df['t'] < ERUPTION_START]
    baseline_phi, _ = _circular_mean_and_se_deg(pre['phi_az'].values)

    roll_all = rolling_phi_stats_daily_then_roll(df, baseline_phi, window_days=ROLL_WINDOW_DAYS,
                                                 min_days=ROLL_MIN_DAYS)
    valid_post = roll_all.dropna(subset=['mean_phi']).copy()
    valid_post = valid_post[valid_post['t'] >= ERUPTION_END]

    _dd, infl_raw, _infl_roll_30, _rt, _rd = station['infl_module'].load_daily_series()
    inflation_roll = infl_raw.rolling(f'{UPLIFT_ROLLING_DAYS}D', center=True,
                                      min_periods=UPLIFT_ROLLING_DAYS // 2).mean()
    infl_df = inflation_roll.dropna().reset_index()
    infl_df.columns = ['t', 'inflation_m']

    merged = pd.merge_asof(valid_post.sort_values('t'), infl_df.sort_values('t'), on='t',
                           direction='nearest', tolerance=pd.Timedelta('20D'))
    merged = merged.dropna(subset=['inflation_m'])

    x_real = merged['inflation_m'].values
    y_ext_real = merged['mean_phi'].values + EXTENSION_SHIFT_DEG[name]
    wrap = compute_optimal_wrap(y_ext_real)
    y_real = (y_ext_real - wrap) % 180.0

    u0 = U_Z0[name]
    C1, C2, A, beta, r, model = fit_atan2_vectorsum(x_real, y_real, u0)
    beta_az = _azimuth(beta)
    print(f'  {name}: C1={C1:.2f}, C2={C2:.2f}, A={A:.3f}, background vector azimuth (alpha, '
         f'fixed)={ALPHA_FIXED_AZ_DEG:.0f}°, inflation vector azimuth (beta)={beta_az:.1f}°, '
         f'turnover u_z={u0:.3f} (exact steepest point), r={r:.2f}, N={len(x_real)}')

    def model_real_wrapped(u):
        return model(u, C1, C2, A, beta) % 180.0

    def true_phi(uz):
        return float((model_real_wrapped(np.array([uz]))[0] + wrap) % 180.0)

    # -- Gap reconstruction: real 2015-2017 SWS phi, inverted through the fit above.
    ashes_first_date = infl_df['t'].min()
    gap_mask = (valid_post['t'] >= ERUPTION_END) & (valid_post['t'] < ashes_first_date)
    gap = valid_post[gap_mask].sort_values('t')
    y_gap = (gap['mean_phi'].values + EXTENSION_SHIFT_DEG[name] - wrap) % 180.0
    x_recon = np.array([invert_atan2_vectorsum(yt, C1, C2, A, beta, u0, u_bound=U_BOUND)
                        for yt in y_gap])
    ok = ~np.isnan(x_recon)
    x_recon, y_recon = x_recon[ok], y_gap[ok]
    t_recon = gap['t'].values[ok]
    print(f'  Gap reconstruction: {len(gap)} real 2015-2017 SWS windows, {ok.sum()} invertible '
         f'(u_bound=±{U_BOUND:.2f} m)')
    if ok.sum():
        print(f'  Reconstructed u_z range: {x_recon.min():.3f} to {x_recon.max():.3f} m')

    u_min = float(min(x_real.min(), x_recon.min())) if len(x_recon) else float(x_real.min())
    u_max = float(x_real.max())

    # The fitted model's Y/X ARE a literal vector sum already in real compass terms -- see
    # animate_arctan_stress_vectors.py's module docstring. infl_mag(u) is EXACT and LINEAR in
    # u, hence automatically, exactly monotonic non-decreasing -- no per-frame solving, no
    # running-max patch, no flip risk, unlike the previous (plain-arctan) version.
    def infl_mag(u):
        return (u - u0) / A - A * np.cos(ALPHA_FIXED_RAD - beta)

    bg_vec = A * np.array([np.cos(ALPHA_FIXED_RAD), np.sin(ALPHA_FIXED_RAD)])

    def vecs(uz):
        infl_vec = infl_mag(uz) * np.array([np.cos(beta), np.sin(beta)])
        return bg_vec, infl_vec, bg_vec + infl_vec

    u_sweep = np.linspace(u_min, u_max, N_FRAMES)

    # Per-station DISPLAY scale factor (see animate_arctan_stress_vectors.py's make_gif) --
    # maps this station's own actual geometric extent over the sweep onto the same target
    # zoom (TARGET_MAX_EXTENT_KM) used for every station, regardless of A's fitted scale.
    raw_max_extent = np.hypot(*bg_vec)
    for uz in u_sweep:
        _, infl_vec, comb_vec = vecs(uz)
        raw_max_extent = max(raw_max_extent, np.hypot(*infl_vec), np.hypot(*comb_vec))
    scale_factor = TARGET_MAX_EXTENT_KM / raw_max_extent if raw_max_extent > 0 else 1.0
    lim = TARGET_MAX_EXTENT_KM * 1.15

    x_line = np.linspace(u_min, u_max, 300)
    y_line = model_real_wrapped(x_line)

    # -- Real station location and bathymetry crop (see animate_arctan_stress_vectors.py).
    x0, y0 = load_station_xy(name)
    bathy_gray, bathy_extent = load_bathy_gray(x0, y0, lim)

    # -- uz -> real calendar date lookup, including the RECONSTRUCTED-gap (x_recon, t_recon)
    # pairs alongside the real (inflation_m, t) ones -- the rose panel only needs a real event
    # timestamp for each frame's uz, not the uplift record itself, so frames swept through the
    # reconstructed 2015-2017 domain still resolve to a real date this way.
    lookup_infl = np.concatenate([merged['inflation_m'].values, x_recon])
    lookup_t = np.concatenate([merged['t'].values, t_recon])

    def nearest_date(uz):
        idx = int(np.argmin(np.abs(lookup_infl - uz)))
        return pd.Timestamp(lookup_t[idx]).tz_localize('UTC')

    fig = plt.figure(figsize=(16, 5.5))
    axRose = fig.add_subplot(1, 3, 1, projection='polar')
    axL = fig.add_subplot(1, 3, 2)
    axR = fig.add_subplot(1, 3, 3)

    axR.scatter(x_real, y_real, s=14, color='#0072B2', alpha=0.5, zorder=2,
               label='Real post-2017 data')
    axR.scatter(x_recon, y_recon, s=26, color=COLOR_RECON, alpha=0.85, marker='D', zorder=3,
               label='Reconstructed 2015-2017 (from real SWS $\\phi$)')
    y_line_plot = break_wrapped_line(y_line)
    axR.plot(x_line, y_line_plot, color='black', lw=1.5, linestyle='--', zorder=3,
            label=r'$\phi(u_z)=C_1+C_2\mathrm{atan2}(Y,X)$'
                 r'$,\ Y=A\sin\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\sin\beta,\ $'
                 r'$X=A\cos\alpha+(\frac{u_z-u_0}{A}-A\cos(\alpha-\beta))\cos\beta$'
                 f'\n$C_1$={C1:.2f}, $C_2$={C2:.2f}, A={A:.3f} (all free)\n'
                 f'background vector azimuth α={ALPHA_FIXED_AZ_DEG:.0f}° (fixed), inflation '
                 f'vector azimuth β={beta_az:.1f}°')
    axR.axvline(u0, color='gray', lw=0.6, linestyle=':', zorder=1,
               label=f'Turnover ($u_z$={u0:.3f}, exact steepest point)')
    dotR, = axR.plot([], [], marker='o', markersize=10, color=COLOR_COMBINED, zorder=5)
    guideR, = axR.plot([], [], color=COLOR_COMBINED, lw=1.2, linestyle='--', zorder=4)
    axR.set_xlabel('De-tided uplift $u_z$ (m) -- real (2017 onward) or reconstructed (2015-2017)')
    axR.set_ylabel(f'Principal extension direction $\\phi$ (deg, {ROLL_WINDOW_DAYS}-day rolling window)')

    y_lo, y_hi, tick_positions, tick_labels = compute_full_range_ticks(wrap)
    axR.set_ylim(y_lo, y_hi)
    axR.set_yticks(tick_positions)
    axR.set_yticklabels([str(v) for v in tick_labels])
    axR.set_title(f'{name}: uplift vs. fast direction\n(atan2 vector-sum fit, r = {r:.2f}, gap reconstructed)',
                 fontsize=10, fontweight='bold')
    axR.legend(loc='lower left', fontsize=6.5, framealpha=0.9)
    axR.grid(alpha=0.3)
    x_axis_min, x_axis_max = axR.get_xlim()

    axL.imshow(bathy_gray, origin='upper', extent=bathy_extent, aspect='auto', cmap='gray',
              alpha=0.5, zorder=0)
    axL.set_xlim(x0 - lim, x0 + lim)
    axL.set_ylim(y0 - lim, y0 + lim)
    axL.set_aspect('equal')
    axL.plot(x0, y0, marker='^', markersize=10, mfc='#FFD700', mec='k', mew=1, zorder=12)
    axL.annotate(name, (x0, y0), textcoords='offset points', xytext=(8, 8),
                fontsize=10, fontweight='bold', zorder=13)
    axL.set_title('Background + inflation stress vectors (real fit geometry, gap reconstructed)',
                 fontsize=10, fontweight='bold')
    axL.set_xlabel('East (km)')
    axL.set_ylabel('North (km)')

    axL.annotate('', xy=(0.05, 0.92), xytext=(0.05, 0.80), xycoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='0.3', lw=1.2), zorder=10)
    axL.annotate('N', (0.05, 0.94), xycoords='axes fraction', ha='center', fontsize=8,
                color='0.3', zorder=10)

    arrow_bg = axL.annotate('', xy=(x0 + bg_vec[0] * scale_factor, y0 + bg_vec[1] * scale_factor),
                            xytext=(x0, y0),
                            arrowprops=dict(arrowstyle='-|>', color=COLOR_BG, lw=2,
                                            linestyle='--'), zorder=14)
    arrow_infl = axL.annotate('', xy=(x0, y0), xytext=(x0, y0),
                              arrowprops=dict(arrowstyle='-|>', color=COLOR_INFL, lw=2,
                                              linestyle='--'), zorder=14)
    arrow_comb = axL.annotate('', xy=(x0, y0), xytext=(x0, y0),
                              arrowprops=dict(arrowstyle='-|>', color=COLOR_COMBINED, lw=2.5),
                              zorder=15)

    angle_text = axL.text(0.97, 0.03, '', transform=axL.transAxes, ha='right', va='bottom',
                          fontsize=11, fontweight='bold', color=COLOR_COMBINED, zorder=16)

    legend_handles = [
        Line2D([0], [0], color=COLOR_BG, lw=2, linestyle='--',
              label=f'Background stress (azimuth α={ALPHA_FIXED_AZ_DEG:.0f}° fixed, mag A={A:.3f})'),
        Line2D([0], [0], color=COLOR_INFL, lw=2, linestyle='--',
              label=f'Inflation stress (azimuth β={beta_az:.1f}° fixed, mag grows linearly '
                    f'in $u_z$)'),
        Line2D([0], [0], color=COLOR_COMBINED, lw=2.5,
              label='Combined (real vector sum -- its own geometric angle is NOT the '
                    'displayed φ, see animate_arctan_stress_vectors.py docstring)'),
    ]
    detail_legend = axL.legend(handles=legend_handles, loc='upper center',
                               bbox_to_anchor=(0.5, -0.18), fontsize=7.5, framealpha=0.9)
    axL.add_artist(detail_legend)

    simple_legend_handles = [
        Line2D([0], [0], color=COLOR_BG, lw=2, linestyle='--',
              label='Regional tectonic stress vector'),
        Line2D([0], [0], color=COLOR_INFL, lw=2, linestyle='--',
              label='Increasing inflation stress vector'),
        Line2D([0], [0], color=COLOR_COMBINED, lw=2.5, label='Effective stress vector'),
    ]
    axL.legend(handles=simple_legend_handles, loc='upper right', fontsize=7.5, framealpha=0.9)

    def draw_rose_frame(center):
        axRose.clear()
        phi_vals, lo, hi = phi_window(df, center)
        _draw_rose(axRose, phi_vals, np.ones(len(phi_vals)), COLOR_ROSE)
        draw_rose_mean_line(axRose, phi_vals)
        axRose.set_title(f'{name}: fast direction rose\n(raw $\\phi$, {ROSE_WINDOW_DAYS}-day '
                         f'window, N={len(phi_vals)})\n{lo.date()} to {hi.date()}',
                         fontsize=9, fontweight='bold')

    def update(frame_idx):
        uz = u_sweep[frame_idx]
        bg, infl, comb = vecs(uz)

        arrow_bg.xy = (x0 + bg[0] * scale_factor, y0 + bg[1] * scale_factor)
        arrow_infl.xy = (x0 + infl[0] * scale_factor, y0 + infl[1] * scale_factor)
        arrow_comb.xy = (x0 + comb[0] * scale_factor, y0 + comb[1] * scale_factor)

        # The displayed angle is the model's OWN true phi(u_z) (matching the right panel's dot
        # exactly), NOT derived from the arrow geometry above.
        true_deg = true_phi(uz)
        angle_text.set_text(f'{true_deg:.1f}°')

        phi_now = model_real_wrapped(np.array([uz]))[0]
        dotR.set_data([uz], [phi_now])
        guideR.set_data([x_axis_min, x_axis_max], [phi_now, phi_now])

        draw_rose_frame(nearest_date(uz))

        return arrow_bg, arrow_infl, arrow_comb, dotR, guideR, angle_text

    fig.text(0.5, 0.01,
             'Fast direction shifted $-90°$ to represent extension, not compression (right '
             'panel); $\\phi$ axis wraps mod 180.',
             ha='center', fontsize=7.5, color='0.35')
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    anim = FuncAnimation(fig, update, frames=N_FRAMES, blit=False)
    anim.save(OUT_PATH, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f'  Saved {OUT_PATH}')


if __name__ == '__main__':
    main()
