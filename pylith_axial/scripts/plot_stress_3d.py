#!/usr/bin/env python3
"""
3D Von Mises stress field figures for all PyLith scenarios.

Renders isosurfaces at 10–100 MPa (10 MPa intervals), fault surfaces,
Mogi source sphere, and a transparent bathymetry cap.  One PNG per scenario
is written to pylith_axial/figures/stress_3d/.
"""

import os, glob, sys
import numpy as np
import h5py
import pyvista as pv
from matplotlib import colormaps

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE, "output")
FIGDIR = os.path.join(BASE, "figures", "stress_3d")
os.makedirs(FIGDIR, exist_ok=True)

# ── Colormap: 10 discrete bands 0–100 MPa ────────────────────────────────────
CLIM     = (0, 100)
N_COLORS = 10

# ── Mogi source geometry (from generate_mesh.py) ─────────────────────────────
MOGI_CENTER = (7.57, 4.55, -3.33)  # km
MOGI_RADIUS = 0.43                  # km

# ── Z-stretch: domain is 10 km deep but 16–18 km wide;
#    scale Z×2 for visual clarity
Z_SCALE = 2.0


def scale_verts(v):
    """Apply Z_SCALE vertical exaggeration in-place on a (N,3) array copy."""
    v2 = v.copy()
    v2[:, 2] *= Z_SCALE
    return v2


def scale_pt(xyz):
    return (xyz[0], xyz[1], xyz[2] * Z_SCALE)


# ── Helper: load fault mesh ───────────────────────────────────────────────────

def load_surface_h5(path):
    """Return PyVista PolyData (triangles) from a PyLith surface HDF5 output."""
    with h5py.File(path, "r") as f:
        verts = f["geometry/vertices"][:]
        cells = f["viz/topology/cells"][:]
    verts = scale_verts(verts)
    n = len(cells)
    conn = np.hstack([np.full((n, 1), 3, dtype=np.int64), cells]).flatten()
    return pv.PolyData(verts, conn)


# ── Build shared geometry once ───────────────────────────────────────────────
print("Building shared geometry …")

_ref = None
for _cand in ["pre_dP0.10GPa", "syn_dP0.10GPa"]:
    if os.path.exists(os.path.join(OUTDIR, f"{_cand}-fault_east.h5")):
        _ref = _cand
        break
if _ref is None:
    sys.exit("No reference scenario found for fault geometry.")

fault_east = load_surface_h5(os.path.join(OUTDIR, f"{_ref}-fault_east.h5"))
fault_west = load_surface_h5(os.path.join(OUTDIR, f"{_ref}-fault_west.h5"))

# Mogi source sphere (Z-scaled)
mogi_sphere = pv.Sphere(
    radius=MOGI_RADIUS,
    center=scale_pt(MOGI_CENTER),
    theta_resolution=40,
    phi_resolution=40,
)

# Bathymetry: top surface cap from the base tet mesh (z > -0.15 km)
with h5py.File(os.path.join(OUTDIR, f"{_ref}-material_crust.h5"), "r") as f:
    _verts_raw = f["geometry/vertices"][:]
    _cells_raw = f["viz/topology/cells"][:]

_verts_sc = scale_verts(_verts_raw)
_n = len(_cells_raw)
_conn = np.hstack([np.full((_n, 1), 4, dtype=np.int64), _cells_raw]).flatten()
_grid_base = pv.UnstructuredGrid(_conn, np.full(_n, pv.CellType.TETRA), _verts_sc)
_surf = _grid_base.extract_surface(algorithm="dataset_surface")
# clip to the top cap (Z-scaled: original z=-0.15 → z=-0.30 scaled)
bathymetry = _surf.clip("z", origin=(0, 0, -0.15 * Z_SCALE), invert=False)
print(f"  bathymetry cap: {bathymetry.n_cells} surface cells")

# Domain centre for camera framing (Z-scaled)
_DOM_CTR = (8.0, 8.0, -5.0 * Z_SCALE)

# Camera: SE-high position looking at domain centre
# Domain is ~16×18 km in XY, 20 km tall (after Z-scale).
# Put the observer ~55 km away.
CAMERA_POS   = (38.0, -18.0, 35.0)
CAMERA_FOCUS = _DOM_CTR
CAMERA_UP    = (0.0, 0.0, 1.0)


# ── Von Mises computation ─────────────────────────────────────────────────────

# Unit provenance for cauchy_stress (must match run_sweep.py):
#   FIXED 2026-07-22: cfg/pylithapp.cfg now sets `reader.coordsys.units = km`,
#   so PyLith's mesh reader correctly scales node coordinates by 1000x on
#   read (previously unset -> coordinates were misread as metres). With that
#   fixed there's no geometric rescale artifact left, so PyLith's raw
#   cauchy_stress output is true Pa and no STRESS_SCALE correction is needed
#   -- only convert to MPa (divide by 1e6). Outputs generated before this fix
#   used the old (unfixed) mesh scale and should be re-run for comparison.
STRESS_SCALE = 1.0


def von_mises_mpa(stress_pa):
    s = stress_pa * STRESS_SCALE   # recover true Pa
    vm = np.sqrt(0.5 * (
        (s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2
        + 6 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)
    ))
    return vm / 1e6   # true Pa -> MPa


# ── Per-scenario render ───────────────────────────────────────────────────────

def render_scenario(name):
    mat_file = os.path.join(OUTDIR, f"{name}-material_crust.h5")
    if not os.path.exists(mat_file):
        print(f"  SKIP {name}: material file missing")
        return

    out_png = os.path.join(FIGDIR, f"{name}.png")
    if os.path.exists(out_png):
        print(f"  skip (exists) {name}")
        return

    # Load mesh + stress. Legacy elastic outputs wrote cauchy_stress as a
    # cell field; the current poroelastic outputs write it as a vertex field.
    with h5py.File(mat_file, "r") as f:
        verts = f["geometry/vertices"][:]
        cells = f["viz/topology/cells"][:]
        if "cauchy_stress" in f.get("cell_fields", {}):
            stress = f["cell_fields/cauchy_stress"][0]
            stress_is_vertex = False
        else:
            stress = f["vertex_fields/cauchy_stress"][0]
            stress_is_vertex = True

    vm = von_mises_mpa(stress)

    # Apply Z-scale to vertices
    verts_sc = scale_verts(verts)

    # Build PyVista tet grid
    nc = len(cells)
    conn = np.hstack([np.full((nc, 1), 4, dtype=np.int64), cells]).flatten()
    grid = pv.UnstructuredGrid(conn, np.full(nc, pv.CellType.TETRA), verts_sc)

    if stress_is_vertex:
        grid.point_data["vm_mpa"] = vm
        grid_pt = grid
    else:
        grid.cell_data["vm_mpa"] = vm
        # Interpolate cell data to points for smooth contouring
        grid_pt = grid.cell_data_to_point_data()

    # Isosurfaces at 10, 20, … 100 MPa
    iso_levels = list(range(10, 110, 10))
    iso_mesh = grid_pt.contour(isosurfaces=iso_levels, scalars="vm_mpa")

    # ── Scalar bar config (defined before add_mesh) ────────────────────────────
    sargs = dict(
        title="Von Mises (MPa)",
        title_font_size=17,
        label_font_size=13,
        n_labels=11,
        vertical=True,
        position_x=0.88,
        position_y=0.08,
        width=0.035,
        height=0.75,
        fmt="%.0f",
        color="white",
    )

    # ── Plotter ───────────────────────────────────────────────────────────────
    pl = pv.Plotter(off_screen=True, window_size=(1800, 1200))
    pl.set_background("#0d1117")

    # ── Stress isosurfaces ────────────────────────────────────────────────────
    if iso_mesh.n_cells > 0:
        pl.add_mesh(
            iso_mesh,
            scalars="vm_mpa",
            cmap="plasma_r",
            clim=CLIM,
            n_colors=N_COLORS,
            opacity=0.42,
            smooth_shading=True,
            show_scalar_bar=True,
            scalar_bar_args=sargs,
        )
    else:
        # Zero-stress scenario — anchor the colorbar with a hidden point
        _pt = pv.PolyData(np.array([[*scale_pt(MOGI_CENTER)]]))
        _pt.point_data["vm_mpa"] = np.array([50.0])
        pl.add_mesh(_pt, scalars="vm_mpa", cmap="plasma_r",
                    clim=CLIM, n_colors=N_COLORS, point_size=0.0,
                    opacity=0.0, show_scalar_bar=True,
                    scalar_bar_args=sargs)

    # ── Fault surfaces — coral/orange so they contrast with everything ─────────
    pl.add_mesh(fault_east, color="#FF6B35", opacity=0.90,
                smooth_shading=True, show_edges=False)
    pl.add_mesh(fault_west, color="#FF6B35", opacity=0.90,
                smooth_shading=True, show_edges=False)

    # ── Mogi source sphere ────────────────────────────────────────────────────
    pl.add_mesh(mogi_sphere, color="#FFE14D", opacity=0.80,
                smooth_shading=True)

    # ── Bathymetry cap — very transparent ice blue ────────────────────────────
    pl.add_mesh(bathymetry, color="#b8d8f0", opacity=0.09,
                smooth_shading=True)

    # ── Domain bounding box (light edges for orientation) ─────────────────────
    _bounds = [0, 16, -1, 17, -10 * Z_SCALE, 0.1 * Z_SCALE]
    _box = pv.Box(bounds=_bounds)
    pl.add_mesh(_box, style="wireframe", color="#334455",
                line_width=0.8, opacity=0.5)

    # ── Compass labels at top corners ─────────────────────────────────────────
    _z_top = 0.1 * Z_SCALE + 0.5
    pl.add_point_labels(
        np.array([[16.0, 8.5, _z_top], [0.0, 8.5, _z_top],
                  [8.0, 17.0, _z_top], [8.0, -1.0, _z_top]]),
        ["E", "W", "N", "S"],
        font_size=14, text_color="white", point_size=0,
        always_visible=True, shadow=True,
        show_points=False, shape=None,
    )

    # ── Depth scale label ─────────────────────────────────────────────────────
    pl.add_text(f"Z ×{Z_SCALE:.0f} (vertical exaggeration)",
                position="lower_right", font_size=10, color="#aaaaaa")

    # ── Title and legend text ─────────────────────────────────────────────────
    if name.startswith("pre_"):
        dP = name.replace("pre_dP", "").replace("GPa", "")
        subtitle = f"Pre-eruptive inflation   ΔP = {dP} GPa"
    elif name.startswith("syn_"):
        dP = name.replace("syn_dP", "").replace("GPa", "")
        subtitle = f"Syn-eruptive deflation   ΔP = {dP} GPa"
    elif name.startswith("pore_"):
        pf = name.replace("pore_Pf", "").replace("sv", "")
        subtitle = f"Pore pressure   Pf = {pf} σᵥ"
    else:
        subtitle = name

    pl.add_text(f"Axial Seamount — {subtitle}",
                position="upper_left", font_size=16, color="white",
                shadow=True)
    pl.add_text(
        "Von Mises stress  0–100 MPa  (10 MPa isosurfaces)\n"
        "Orange: ring faults   Yellow: Mogi source   Blue: seafloor",
        position="lower_left", font_size=11, color="#cccccc", shadow=True,
    )

    # ── Axes widget ───────────────────────────────────────────────────────────
    pl.add_axes(
        xlabel="E (km)", ylabel="N (km)", zlabel="Z (km, ×2)",
        color="white", line_width=2,
    )

    # ── Camera ────────────────────────────────────────────────────────────────
    pl.camera_position = [CAMERA_POS, CAMERA_FOCUS, CAMERA_UP]
    pl.camera.zoom(0.80)      # slight zoom-out to ensure nothing clips

    pl.screenshot(out_png, transparent_background=False)
    pl.close()
    print(f"  saved → {os.path.basename(out_png)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def get_all_scenarios():
    paths = glob.glob(os.path.join(OUTDIR, "*-material_crust.h5"))
    return sorted(
        os.path.basename(p).replace("-material_crust.h5", "")
        for p in paths
    )


if __name__ == "__main__":
    # Delete existing figures so they are regenerated with updated camera
    import glob as _glob
    existing = _glob.glob(os.path.join(FIGDIR, "*.png"))
    for f in existing:
        os.remove(f)

    scenarios = get_all_scenarios()
    print(f"\nFound {len(scenarios)} scenarios.  Writing figures to:\n  {FIGDIR}\n")

    for i, name in enumerate(scenarios, 1):
        print(f"[{i:3d}/{len(scenarios)}] {name}")
        try:
            render_scenario(name)
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

    print(f"\nDone.  {len(scenarios)} figures in {FIGDIR}/")
