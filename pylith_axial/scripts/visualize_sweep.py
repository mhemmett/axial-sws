"""
visualize_sweep.py — Post-processing and figures for PyLith sweep output.

Reads HDF5 output from run_sweep.py scenarios and produces:
  1. Surface displacement maps (Ux, Uy, Uz) vs source pressure
  2. Fault slip magnitude and rake on east/west faults vs pressure
  3. Hanging-wall Cauchy stress (σxx, σyy, σzz, σxz) on eastern fault
  4. Coulomb stress change on fault (using post-processed stress + fault geometry)
  5. Pore-pressure effect: slip and hanging-wall stress vs Pf/σv
  6. Summary figure: key scalar diagnostics vs ΔP for pre/syn/pore modes

Outputs saved to pylith_axial/figures/

Usage:
  python visualize_sweep.py --mode pre
  python visualize_sweep.py --mode all
"""

import os, argparse, glob
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJ    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR  = os.path.join(PROJ, 'output')
FIGDIR  = os.path.join(PROJ, 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Fault geometry constants (for Coulomb stress calculation)
STRIKE_RAD   = np.radians(330.0)   # N30W
DIP_RAD      = np.radians(66.0)
# Eastern fault dip direction (toward E, az=60°)
E_DIPAZ_RAD  = np.radians(60.0)
W_DIPAZ_RAD  = np.radians(240.0)

# Static friction coefficient for Coulomb stress
MU_FRICTION  = 0.6


# ── HDF5 readers ──────────────────────────────────────────────────────────────

def load_h5(path, field):
    """Return (coords, data) from an HDF5 file. data shape: (ntime, npts, ncomp)."""
    with h5py.File(path, 'r') as f:
        coords = f['geometry/vertices'][:]
        data   = f[f'vertex_fields/{field}'][:]
    return coords, data


def load_cell_h5(path, field):
    """Return (cell_centers, data) from cell-field HDF5. data: (ntime, ncells, ncomp)."""
    with h5py.File(path, 'r') as f:
        verts  = f['geometry/vertices'][:]       # (nvert, 3)
        cells  = f['viz/topology/cells'][:]      # (ncells, 4)  ← correct path
        data   = f[f'cell_fields/{field}'][:]    # (ntime, ncells, ncomp)
    centers = verts[cells].mean(axis=1)          # (ncells, 3)
    return centers, data

# Unit provenance for cauchy_stress (must match plot_stress_3d.py):
#   FIXED 2026-07-22: cfg/pylithapp.cfg now sets `reader.coordsys.units = km`,
#   so PyLith's mesh reader correctly scales node coordinates by 1000x on read
#   (previously unset -> coordinates were misread as metres, `scales.length_scale
#   = 1.0*km` is only the nondimensionalization scale, not a coordinate-unit
#   conversion, so it didn't fix this). Before the fix, the simulated domain was
#   1000x too small, so strains/cauchy_stress came out 1000x too large, and this
#   STRESS_SCALE undid that inflation (verified then: median bulk stress
#   perturbation ~2.3% of source dP, physically sensible for a Mogi source).
#   With the mesh fix in place there's no inflation left to undo, so
#   STRESS_SCALE is now 1.0 -- PRE-FIX output/*.h5 files are STALE (generated
#   at the wrong mesh scale) and must be re-run before comparing against
#   anything produced after this change; the 2.3% check above should be
#   re-verified against a post-fix re-run rather than assumed to still hold.
STRESS_SCALE = 1.0


def find_scenario_file(label, suffix):
    """Find output HDF5 file for a scenario label and file suffix (e.g. 'solution')."""
    pattern = os.path.join(OUTDIR, f'{label}-{suffix}.h5')
    matches = glob.glob(pattern)
    if not matches:
        return None
    return matches[0]


# ── Fault normal and slip vectors ─────────────────────────────────────────────

def fault_normal(dip_az_rad):
    """Outward unit normal to fault (pointing toward hanging wall)."""
    # Normal = dip-direction rotated 90° upward from horizontal
    n = np.array([
        np.sin(dip_az_rad) * np.sin(DIP_RAD),
        np.cos(dip_az_rad) * np.sin(DIP_RAD),
        np.cos(DIP_RAD)
    ])
    return n / np.linalg.norm(n)


def fault_strike_vec():
    """Along-strike unit vector (N30W = az 330°)."""
    return np.array([np.sin(STRIKE_RAD), np.cos(STRIKE_RAD), 0.0])


def fault_updip_vec(dip_az_rad):
    """Up-dip unit vector (perpendicular to strike, pointing up the fault plane)."""
    s = fault_strike_vec()
    n = fault_normal(dip_az_rad)
    return np.cross(s, n)   # up-dip = strike × normal (right-hand rule)


def coulomb_stress(stress_tensor, dip_az_rad, pore_pressure=0.0):
    """
    Compute Coulomb failure function on fault plane.
    stress_tensor: (6,) Voigt notation [σxx,σyy,σzz,σxy,σyz,σxz]
    Returns: delta_CFS = |tau| - mu*(sigma_n - Pf)
    """
    s = stress_tensor
    S = np.array([[s[0], s[3], s[5]],
                  [s[3], s[1], s[4]],
                  [s[5], s[4], s[2]]])
    n = fault_normal(dip_az_rad)
    d = fault_updip_vec(dip_az_rad)
    strike = fault_strike_vec()

    traction  = S @ n
    sigma_n   = np.dot(traction, n)          # normal stress (+ = compression)
    tau_dip   = np.dot(traction, d)           # dip-slip shear
    tau_strike= np.dot(traction, strike)      # strike-slip shear
    tau       = np.sqrt(tau_dip**2 + tau_strike**2)
    delta_cfs = tau - MU_FRICTION * (sigma_n - pore_pressure)
    return delta_cfs, sigma_n, tau


# ── Scenario loading ───────────────────────────────────────────────────────────

def parse_label(label):
    """Extract dP_MPa and Pf_frac from scenario label string."""
    import re
    dP = 0.0; Pf = 0.0
    m = re.search(r'dP([\d.]+)GPa', label)
    if m: dP = float(m.group(1)) * 1000.   # → MPa
    m = re.search(r'Pf([\d.]+)sv', label)
    if m: Pf = float(m.group(1))
    return dP, Pf


def load_scenario(label):
    """Load key fields for one scenario. Returns dict or None if files missing."""
    sol_path  = find_scenario_file(label, 'solution')
    fe_path   = find_scenario_file(label, 'fault_east')
    fw_path   = find_scenario_file(label, 'fault_west')
    mat_path  = find_scenario_file(label, 'material_crust')

    if not sol_path:
        return None

    result = {'label': label}
    dP, Pf = parse_label(label)
    result['dP_MPa'] = dP
    result['Pf_frac'] = Pf

    # Surface displacement (z ≈ 0 nodes)
    coords, disp = load_h5(sol_path, 'displacement')
    top_mask = np.abs(coords[:, 2] - 0.1) < 0.2   # near free surface
    result['surf_coords'] = coords[top_mask]
    result['surf_disp']   = disp[-1, top_mask, :]   # last time step, [N,3]

    # Max vertical displacement at surface
    result['max_uz'] = float(disp[-1, top_mask, 2].max())
    result['min_uz'] = float(disp[-1, top_mask, 2].min())

    # Fault east slip
    if fe_path:
        fe_coords, fe_slip = load_h5(fe_path, 'slip')
        result['fe_coords'] = fe_coords
        result['fe_slip']   = fe_slip[-1]       # (N,3): [left_lateral, reverse, opening]
        slip_mag = np.linalg.norm(fe_slip[-1], axis=-1)
        result['fe_slip_mag_mean'] = float(slip_mag.mean())
        result['fe_slip_mag_max']  = float(slip_mag.max())

    # Fault west slip
    if fw_path:
        fw_coords, fw_slip = load_h5(fw_path, 'slip')
        result['fw_coords']        = fw_coords
        result['fw_slip']          = fw_slip[-1]
        result['fw_slip_mag_mean'] = float(np.linalg.norm(fw_slip[-1], axis=-1).mean())
        result['fw_slip_mag_max']  = float(np.linalg.norm(fw_slip[-1], axis=-1).max())

    # Hanging-wall stress: cells within 1 km of eastern fault, on hanging-wall side
    if mat_path:
        try:
            cc, stress = load_cell_h5(mat_path, 'cauchy_stress')
            # Eastern fault: hanging wall is toward +x (east)
            # Centroid east of fault plane, at mid-depth
            n_e = fault_normal(E_DIPAZ_RAD)
            # Project cell centers onto fault normal to find hanging-wall side
            # Use fault centre at (8.5, 4.0, -1.6) km approx
            fc = np.array([8.5, 4.0, -1.6])
            hw_mask = (np.dot(cc - fc, n_e) > 0) & (np.linalg.norm(cc - fc, axis=1) < 2.0)
            if hw_mask.sum() > 0:
                hw_stress = stress[-1, hw_mask, :] * STRESS_SCALE   # (N,6), Pa-equivalent
                result['hw_sigma_xx'] = float(hw_stress[:, 0].mean())
                result['hw_sigma_yy'] = float(hw_stress[:, 1].mean())
                result['hw_sigma_zz'] = float(hw_stress[:, 2].mean())
                cfs_vals = [coulomb_stress(hw_stress[i], E_DIPAZ_RAD)[0]
                            for i in range(len(hw_stress))]
                result['hw_cfs_mean'] = float(np.mean(cfs_vals))
        except Exception as e:
            print(f'  Warning: stress load failed for {label}: {e}')

    return result


# ── Plotting helpers ───────────────────────────────────────────────────────────

def colorbar(ax, im, label):
    div = make_axes_locatable(ax)
    cax = div.append_axes('right', size='4%', pad=0.05)
    plt.colorbar(im, cax=cax, label=label)


def surface_disp_map(ax, coords, disp_comp, title, cmap='RdBu_r', vmax=None):
    """Scatter plot of surface displacement component."""
    x, y = coords[:, 0], coords[:, 1]
    v = vmax or np.percentile(np.abs(disp_comp), 98)
    im = ax.scatter(x, y, c=disp_comp, cmap=cmap, s=6,
                    vmin=-v, vmax=v, rasterized=True)
    ax.set_aspect('equal'); ax.set_title(title, fontsize=8)
    ax.set_xlabel('East (km)', fontsize=7); ax.set_ylabel('North (km)', fontsize=7)
    colorbar(ax, im, 'm')
    return im


# ── Main figure routines ───────────────────────────────────────────────────────

def fig_surface_disp(scenarios_data, mode, pdf):
    """Surface displacement maps for each scenario (one row per scenario)."""
    n = len(scenarios_data)
    if n == 0: return
    fig, axes = plt.subplots(n, 3, figsize=(12, 2.5*n))
    if n == 1: axes = axes[np.newaxis, :]
    fig.suptitle(f'Surface displacement — {mode} mode', fontsize=10, fontweight='bold')
    labels_col = ['Ux (East)', 'Uy (North)', 'Uz (Vertical)']
    for row, sc in enumerate(scenarios_data):
        coords = sc['surf_coords']; disp = sc['surf_disp']
        for col in range(3):
            ax = axes[row, col]
            title = f'{sc["label"]} — {labels_col[col]}' if row == 0 else labels_col[col]
            surface_disp_map(ax, coords, disp[:, col], title)
            if row == 0:
                ax.set_title(f'ΔP={sc["dP_MPa"]:.0f} MPa\n{labels_col[col]}', fontsize=7)
            else:
                ax.set_title(f'ΔP={sc["dP_MPa"]:.0f} MPa', fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


def fig_scalar_vs_pressure(scenarios_data, mode, pdf):
    """Key scalar diagnostics vs source pressure."""
    dP   = np.array([sc['dP_MPa']        for sc in scenarios_data])
    uz   = np.array([sc.get('max_uz', 0) for sc in scenarios_data])
    slip = np.array([sc.get('fe_slip_mag_max', 0) for sc in scenarios_data])
    cfs  = np.array([sc.get('hw_cfs_mean', np.nan) for sc in scenarios_data])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f'Scalar diagnostics vs source pressure — {mode} mode',
                 fontsize=10, fontweight='bold')

    axes[0].plot(dP, uz*1e3, 'o-', color='steelblue', lw=1.5)
    axes[0].set_xlabel('Source ΔP (MPa)'); axes[0].set_ylabel('Max surface uplift (mm)')
    axes[0].set_title('Surface uplift'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(dP, slip, 'o-', color='firebrick', lw=1.5)
    axes[1].set_xlabel('Source ΔP (MPa)'); axes[1].set_ylabel('Peak fault slip (m)')
    axes[1].set_title('Eastern fault peak slip'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(dP, cfs/1e6, 'o-', color='darkorange', lw=1.5)
    axes[2].axhline(0, color='k', lw=0.8, ls='--')
    axes[2].set_xlabel('Source ΔP (MPa)')
    axes[2].set_ylabel('Mean ΔCFS on hanging wall (MPa)')
    axes[2].set_title('Coulomb stress change (hanging wall)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


def fig_fault_slip_map(scenarios_data, mode, pdf, fault='east'):
    """Fault slip maps for each scenario."""
    key_coords = f'f{"e" if fault=="east" else "w"}_coords'
    key_slip   = f'f{"e" if fault=="east" else "w"}_slip'
    data = [(sc['label'], sc['dP_MPa'],
             sc.get(key_coords), sc.get(key_slip))
            for sc in scenarios_data if sc.get(key_coords) is not None]
    if not data: return

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(3.5*n, 4), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle(f'{fault.capitalize()} ring fault — slip magnitude  [{mode} mode]',
                 fontsize=10, fontweight='bold')
    all_mags = [np.linalg.norm(sl, axis=-1) for _, _, _, sl in data if sl is not None]
    vmax = np.percentile(np.concatenate(all_mags), 99) if all_mags else 1.0

    for ax, (lbl, dP, coords, slip) in zip(axes, data):
        if slip is None: continue
        mag = np.linalg.norm(slip, axis=-1)
        # Plot in fault-plane coordinates: along-strike (x) vs down-dip (z)
        im = ax.scatter(coords[:, 0], coords[:, 2], c=mag,
                        cmap='hot_r', s=10, vmin=0, vmax=vmax, rasterized=True)
        ax.set_title(f'ΔP={dP:.0f} MPa', fontsize=8)
        ax.set_xlabel('East (km)', fontsize=7)
        colorbar(ax, im, 'm')
    axes[0].set_ylabel('Depth (km)', fontsize=7)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


def fig_pore_pressure(scenarios_data, pdf):
    """Pore pressure sweep: slip and stress vs Pf/σv."""
    Pf   = np.array([sc['Pf_frac']                  for sc in scenarios_data])
    slip = np.array([sc.get('fe_slip_mag_max', 0)    for sc in scenarios_data])
    cfs  = np.array([sc.get('hw_cfs_mean', np.nan)   for sc in scenarios_data])
    uz   = np.array([sc.get('max_uz', 0)             for sc in scenarios_data])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle('Pore fluid pressure sweep (ΔP=150 MPa fixed)',
                 fontsize=10, fontweight='bold')

    axes[0].plot(Pf, slip, 'o-', color='firebrick', lw=1.5)
    axes[0].set_xlabel('Pf / σv'); axes[0].set_ylabel('Peak fault slip (m)')
    axes[0].set_title('Eastern fault slip vs pore pressure'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(Pf, cfs/1e6, 'o-', color='darkorange', lw=1.5)
    axes[1].axhline(0, color='k', lw=0.8, ls='--')
    axes[1].set_xlabel('Pf / σv'); axes[1].set_ylabel('Mean ΔCFS (MPa)')
    axes[1].set_title('Coulomb stress (hanging wall)'); axes[1].grid(True, alpha=0.3)

    axes[2].plot(Pf, uz*1e3, 'o-', color='steelblue', lw=1.5)
    axes[2].set_xlabel('Pf / σv'); axes[2].set_ylabel('Max uplift (mm)')
    axes[2].set_title('Surface uplift'); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


def fig_hanging_wall_stress(scenarios_data, mode, pdf):
    """Hanging-wall mean Cauchy stress components vs source pressure."""
    dP  = np.array([sc['dP_MPa'] for sc in scenarios_data])
    sxx = np.array([sc.get('hw_sigma_xx', np.nan) for sc in scenarios_data]) / 1e6
    syy = np.array([sc.get('hw_sigma_yy', np.nan) for sc in scenarios_data]) / 1e6
    szz = np.array([sc.get('hw_sigma_zz', np.nan) for sc in scenarios_data]) / 1e6

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(f'Hanging-wall mean Cauchy stress vs ΔP — {mode} mode',
                 fontsize=10, fontweight='bold')
    ax.plot(dP, sxx, 'o-', label='σxx (E–W)', color='steelblue')
    ax.plot(dP, syy, 's-', label='σyy (N–S)', color='firebrick')
    ax.plot(dP, szz, '^-', label='σzz (vert)', color='forestgreen')
    ax.axhline(0, color='k', lw=0.7, ls='--')
    ax.set_xlabel('Source ΔP (MPa)'); ax.set_ylabel('Mean stress (MPa)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    pdf.savefig(fig, dpi=150, bbox_inches='tight'); plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def get_scenario_labels(mode):
    """Find all output labels matching a mode prefix."""
    if mode == 'all':
        prefixes = ('pre_', 'syn_', 'pore_')
    else:
        prefixes = (f'{mode}_',)
    labels = set()
    for f in glob.glob(os.path.join(OUTDIR, '*-solution.h5')):
        base = os.path.basename(f).replace('-solution.h5', '')
        if any(base.startswith(p) for p in prefixes):
            labels.add(base)
    return sorted(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='pre',
                   choices=['pre', 'syn', 'pore', 'all'],
                   help='Which sweep mode to visualize')
    args = p.parse_args()

    modes = ['pre', 'syn', 'pore'] if args.mode == 'all' else [args.mode]

    for mode in modes:
        labels = get_scenario_labels(mode)
        if not labels:
            print(f'No output found for mode={mode}. Run the sweep first.')
            continue

        print(f'\nLoading {len(labels)} scenarios for mode={mode}...')
        scenarios_data = []
        for lbl in labels:
            sc = load_scenario(lbl)
            if sc:
                scenarios_data.append(sc)
                print(f'  {lbl}: max_uz={sc.get("max_uz",0)*1e3:.2f} mm  '
                      f'slip_max={sc.get("fe_slip_mag_max",0):.3f} m')
            else:
                print(f'  {lbl}: missing output files')

        if not scenarios_data:
            continue

        outpdf = os.path.join(FIGDIR, f'pylith_sweep_{mode}.pdf')
        print(f'Writing {outpdf}...')
        with PdfPages(outpdf) as pdf:
            fig_scalar_vs_pressure(scenarios_data, mode, pdf)
            fig_hanging_wall_stress(scenarios_data, mode, pdf)
            fig_surface_disp(scenarios_data, mode, pdf)
            fig_fault_slip_map(scenarios_data, mode, pdf, fault='east')
            fig_fault_slip_map(scenarios_data, mode, pdf, fault='west')
            if mode == 'pore':
                fig_pore_pressure(scenarios_data, pdf)
        print(f'Saved {outpdf}')


if __name__ == '__main__':
    main()
