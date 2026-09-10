"""
run_sweep.py — Stress-informed two-pass sweep driver for the poroelastic
Axial Seamount PyLith model.

PHYSICS UPGRADE: the model is now quasi-static POROELASTICITY with a
fluid-pressure Mogi source (see cfg/). Slip on the ring faults is no longer a
heuristic function of dP; it is DERIVED from the stress produced by inflation:

  Pass 1 ("pre", no fault): run a no-fault poroelastic inflation solve with a
    prescribed cavity fluid pressure. PyLith writes the Cauchy stress field
    (output/<label>-material_crust.h5).

  Resolve: project that stress onto each ring-fault plane to get the shear
    traction and the Coulomb stress change.

  Pass 2 ("syn", with fault): convert the resolved fault shear traction into a
    prescribed slip via a stress-to-slip relation, write a scenario cfg with
    that slip, and run the full poroelastic + FaultCohesiveKin model.

The sweep is parameterized over the injected cavity FLUID PRESSURE AMPLITUDE
(replaces the old mechanical dP_MPa). Solver settings are swapped per pass
(no-fault vs with-fault poroelastic field-split preconditioners shipped with
PyLith).

Usage:
  python run_sweep.py --pylith /path/to/pylith [--dry-run]

  NOTE: This script will REFUSE to run until two physical inputs that the user
  has not yet specified are filled in (see the SWEEP_PF_AMPLITUDES and
  STRESS_TO_SLIP sections, marked `REQUIRES USER INPUT`). Guessing them would
  silently fix the scientific result, so the script raises instead.
"""

import os
import sys
import subprocess
import argparse
import json
import math

import numpy as np
import h5py

# ── PyLith environment ─────────────────────────────────────────────────────────
PYLITH_DIR = '/Users/mhemmett/Downloads/pylith-5.0.1-macOS-12.0-arm64'
PYLITH_ENV = {
    **os.environ,
    'PYTHONHOME':        PYLITH_DIR,
    'DYLD_LIBRARY_PATH': f"{PYLITH_DIR}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}",
    'PATH':              f"{PYLITH_DIR}/bin:{os.environ.get('PATH', '')}",
    'PYTHONPATH':        f"{PYLITH_DIR}/lib/python3.12/site-packages",
}
PYLITH_BIN    = f'{PYLITH_DIR}/bin/nemesis'
PYLITH_SCRIPT = f'{PYLITH_DIR}/bin/pylith'

# Solver setting files shipped with PyLith (poroelastic field-split).
SOLVER_NOFAULT = f'{PYLITH_DIR}/src/pylith-5.0.1/share/settings/solver_poroelasticity.cfg'
SOLVER_FAULT   = f'{PYLITH_DIR}/src/pylith-5.0.1/share/settings/solver_poroelasticity_fault.cfg'

# ── Project paths ──────────────────────────────────────────────────────────────
PROJ   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG    = os.path.join(PROJ, 'cfg')
OUTDIR = os.path.join(PROJ, 'output')
TMPDIR = os.path.join(PROJ, 'scenarios')
os.makedirs(TMPDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

# Base cfg stack for the FULL (with-fault) problem: ring faults + all 4 dike segment
# interfaces, matching pylithapp.cfg's `interfaces = [fault_east, fault_west, dike1,
# dike2, dike3, dike4]`.
BASE_CFGS_FULL = [
    os.path.join(CFG, 'pylithapp.cfg'),
    os.path.join(CFG, 'materials.cfg'),
    os.path.join(CFG, 'faults.cfg'),
    os.path.join(CFG, 'dike.cfg'),
    os.path.join(CFG, 'bc.cfg'),
    os.path.join(CFG, 'output.cfg'),
]
# Pass-1 stack omits faults.cfg AND dike.cfg (no interfaces at all) -- used by the
# generic SWEEP_PF_AMPLITUDES sweep.
BASE_CFGS_NOFAULT = [
    os.path.join(CFG, 'pylithapp.cfg'),
    os.path.join(CFG, 'materials.cfg'),
    os.path.join(CFG, 'bc.cfg'),
    os.path.join(CFG, 'output.cfg'),
]
# Pass-1 stack for the Baillard scenarios: dike.cfg included (so `interfaces =
# [dike1, dike2, dike3, dike4]` has something to activate) but faults.cfg (ring
# faults) deliberately omitted -- PyLith errors ("unknown component") if a cfg file
# configures a fault name that isn't in the active `interfaces` list, so the ring
# faults' own cfg file cannot be present at all while they're excluded from Pass 1.
BASE_CFGS_NOFAULT_DIKE = [
    os.path.join(CFG, 'pylithapp.cfg'),
    os.path.join(CFG, 'materials.cfg'),
    os.path.join(CFG, 'dike.cfg'),
    os.path.join(CFG, 'bc.cfg'),
    os.path.join(CFG, 'output.cfg'),
]

# ── Stress-scaling provenance (must match the viz scripts) ─────────────────────
# FIXED 2026-07-22: cfg/pylithapp.cfg now sets reader.coordsys.units = km, so
# the mesh geometry is read at its correct physical scale (no more 1000x
# shrink). With that fixed, there is no rescale artifact left to correct for,
# so STRESS_SCALE stays 1.0 -- but note this is now correct for a DIFFERENT
# reason than before the fix (there used to be an argued-invariant rescale;
# now there's simply no rescale at all). Runs/outputs predating the fix used
# the old (unfixed) mesh scale and should be re-run before being compared
# against anything produced after this change.
STRESS_SCALE = 1.0

# ── Fault geometry (from mesh/generate_mesh.py) ────────────────────────────────
# Both faults strike N30W (az 330) and dip 66 deg; eastern dips toward az 60,
# western toward az 240. Unit normal points in the dip-azimuth direction, up by
# cos(dip): n = (sin(dipaz) sin(dip), cos(dipaz) sin(dip), cos(dip)).
STRIKE_DEG = 330.0
DIP_DEG    = 66.0
# Centroids of each fault (km), used to select the nearby crust cells whose
# stress is resolved onto the plane (mirrors generate_mesh.py fe_c / fw_c).
HALF_LEN, DIP_WIDTH, TOP_Z = 4.0, 3.25, -0.1


def _fault_normal(dip_az_deg):
    """Unit normal to a fault plane (z-up, km frame), pointing up-dip-outward."""
    az = math.radians(dip_az_deg)
    dip = math.radians(DIP_DEG)
    return np.array([
        math.sin(az) * math.sin(dip),
        math.cos(az) * math.sin(dip),
        math.cos(dip),
    ])


def _fault_centroid(cx, cy, dip_az_deg):
    az = math.radians(dip_az_deg)
    dip = math.radians(DIP_DEG)
    d = np.array([math.sin(az) * math.cos(dip),
                  math.cos(az) * math.cos(dip),
                  -math.sin(dip)])
    return np.array([cx, cy, TOP_Z]) + 0.5 * DIP_WIDTH * d


FAULTS = {
    'east': {'normal': _fault_normal(60.0),  'centroid': _fault_centroid(8.5, 4.0, 60.0)},
    'west': {'normal': _fault_normal(240.0), 'centroid': _fault_centroid(7.0, 4.0, 240.0)},
}

# Effective friction for the Coulomb stress change. The standard
# effective-stress value (Byerlee-ish for hydrothermally altered basalt) is
# 0.6; this is a documented modeling choice, not a free unknown like the two
# values below. If the user wants a different value, change here.
MU_FRICTION = 0.6


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRES USER INPUT (1/2): injected cavity fluid-pressure amplitudes
# ═══════════════════════════════════════════════════════════════════════════════
# The sweep is over the fluid overpressure injected at the Mogi cavity wall
# (Dirichlet `pressure` BC, cfg/bc.cfg -> mogi_deep_bc.initial_amplitude).
# The old mechanical sweep used 0..100 MPa of WALL TRACTION, which is NOT the
# same physical quantity as an injected pore-fluid overpressure. The correct
# range for a poroelastic source has not been specified.
#
# User-specified single scenario: inject 50 MPa overpressure at the cavity.
# Single-element list -> the sweep runs exactly once.
SWEEP_PF_AMPLITUDES = [50.0e6]                     # user-specified: 50 MPa


# ═══════════════════════════════════════════════════════════════════════════════
# Baillard-geometry pre/syn scenarios (separate from the generic sweep above)
# ═══════════════════════════════════════════════════════════════════════════════
# Runs the mesh's Yang-spheroid + Okada-dike source (mesh/generate_mesh.py) through
# PyLith using EXACTLY scripts/baillard_simple_model.py's own pressure/opening
# values: pre-eruption = spheroid inflation only (dike closed); syn-eruption =
# spheroid deflation + the dike opened 2 m (matching that script's DP_INFLATE /
# DP_DEFLATE / DIKE_OPENING). Run via `--baillard-simple`; does not touch or
# invalidate SWEEP_PF_AMPLITUDES's existing sweep over the old spherical-cavity
# geometry.
BAILLARD_SCENARIOS = [
    {'name': 'baillard_pre', 'pf_amp_Pa': +0.05e9, 'dike_opening_m': 0.0},
    {'name': 'baillard_syn', 'pf_amp_Pa': -0.05e9, 'dike_opening_m': 2.0},
]


# ═══════════════════════════════════════════════════════════════════════════════
# REQUIRES USER INPUT (2/2): stress-to-slip relation
# ═══════════════════════════════════════════════════════════════════════════════
# Pass 2 prescribes kinematic slip derived from the Pass-1 resolved fault shear
# traction. Converting a shear traction (Pa) into a slip (m) requires a physical
# relation, e.g. one of:
#   (a) elastic-dislocation:  slip = (stress_drop * characteristic_length) / mu
#       -> needs a characteristic_length (m) and a stress-drop definition.
#   (b) fault compliance:     slip = compliance * shear_traction
#       -> needs a compliance (m/Pa).
#   (c) fixed stress drop:    a target stress drop (Pa) over a slip patch.
# Each needs at least one parameter the user has NOT specified. Picking one
# silently sets the slip magnitude -> the headline scientific result. So this is
# left unset and the script refuses to run Pass 2 until it is provided.
#
# User-specified: compliance method with compliance = 5e-11 m/Pa (midpoint of
# the suggested 1e-11 .. 1e-10 m/Pa range). slip = compliance * shear_traction.
STRESS_TO_SLIP = {'mode': 'compliance', 'compliance_m_per_Pa': 5.0e-11}

# ── Dike opening timing (baillard_syn only; see cfg/dike.cfg) ──────────────────
# SIMPLIFIED (2026-07-23, explicit user direction): don't model the dike opening as
# evolving/propagating in time at all -- just open the whole structure (all 4 segments)
# instantaneously, in pylithapp.cfg's default single quasi-static step (start_time=-1yr,
# end_time=0yr, initial_dt=1yr -- one step by construction). This replaces an earlier
# KinSrcRamp-based multi-step approach that got a working ramp but then hit a SEPARATE,
# reproducible divergence on the run's LAST time step specific to having 4 separate fault
# interfaces (confirmed independent of ramp/rise-time placement via a diagnostic run) --
# an unresolved solver/preconditioner issue, not a ramp-timing one. See cfg/dike.cfg's
# header for the same reasoning. RUN_START_YR is kept only as the dike's
# initiation_time reference point (0.0*s relative to it, i.e. "on" from t=start_time).
RUN_START_YR = -1.0


# ── Pass 1: write the no-fault inflation scenario cfg ──────────────────────────
def write_pass1_cfg(label, pf_amp_Pa, dike_opening_m=None):
    """Pass 1: no-fault poroelastic inflation at a given injected fluid pressure.

    dike_opening_m=None (default, used by the generic SWEEP_PF_AMPLITUDES sweep):
    no interfaces at all, exactly as before. dike_opening_m=<value> (used by the
    Baillard pre/syn scenarios): the dike IS included in Pass 1 (with its fixed,
    prescribed opening -- NOT stress-derived) so its contribution is part of the
    ambient stress field that resolve_fault_stress() later reads for the ring
    faults, matching Baillard's own linear-superposition treatment. The ring
    faults (fault_east/fault_west) are still excluded from Pass 1 either way,
    since those are what Pass 1 exists to resolve stress for.
    """
    cfg_path = os.path.join(TMPDIR, f'{label}_pass1.cfg')
    outbase = os.path.join(OUTDIR, f'{label}_pass1')

    if dike_opening_m is None:
        interfaces_line = 'interfaces = []'
        subfields_block = """[pylithapp.problem.solution.subfields]
# No fault -> no Lagrange multiplier subfield.
lagrange_multiplier_fault.basis_order = 0
"""
        dike_block = ''
    else:
        interfaces_line = 'interfaces = [dike1, dike2, dike3, dike4]'
        # lagrange_multiplier_fault.basis_order left at pylithapp.cfg's base value
        # (2) since the dike segments are real FaultCohesiveKin interfaces here.
        subfields_block = ''
        # Same fixed, INSTANTANEOUS opening applied uniformly to all 4 segments (see
        # cfg/dike.cfg for why they're 4 separate interfaces, not one merged surface,
        # and for why this is instantaneous rather than ramped).
        dike_block = '\n'.join(f"""
[pylithapp.problem.interfaces.dike{i}.observers.observer]
writer.filename = {outbase}-dike{i}.h5

# Dike opening included in Pass 1 (fixed/prescribed, not stress-derived).
[pylithapp.problem.interfaces.dike{i}.eq_ruptures.rupture]
db_auxiliary_field.data = [{RUN_START_YR:.4f}*year, 0.0*m, 0.0*m, {dike_opening_m:.6f}*m]
""" for i in (1, 2, 3, 4))

    content = f"""# Auto-generated Pass-1 (no-fault inflation) cfg: {label}
# Injected cavity fluid pressure = {pf_amp_Pa / 1e6:.3f} MPa

[pylithapp.problem]
defaults.name = {label}_pass1
# Ring faults excluded from Pass 1 (dike may be included -- see below).
{interfaces_line}

{subfields_block}
[pylithapp.problem.solution_observers.domain]
writer.filename = {outbase}-solution.h5

[pylithapp.problem.materials.crust.observers.observer]
writer.filename = {outbase}-material_crust.h5

# Injected fluid pressure at the Mogi cavity wall.
[pylithapp.problem.bc.mogi_deep_bc]
db_auxiliary_field.data = [{pf_amp_Pa / 1e6:.6f}*MPa]
{dike_block}"""
    with open(cfg_path, 'w') as f:
        f.write(content)
    return cfg_path


# ── Resolve Coulomb stress from a Pass-1 result onto a fault plane ─────────────
def resolve_fault_stress(material_h5, fault_normal, fault_centroid,
                         select_radius_km=1.0):
    """Average the Cauchy stress over crust cells near the fault and resolve it
    onto the plane. Returns (sigma_normal, tau_shear, coulomb) in Pa (true,
    after STRESS_SCALE)."""
    # Legacy elastic outputs wrote cauchy_stress as a cell field; the current
    # poroelastic outputs write it as a vertex field (see plot_stress_3d.py).
    with h5py.File(material_h5, 'r') as f:
        verts = f['geometry/vertices'][:]             # (nverts, 3) km
        cells = f['viz/topology/cells'][:]            # (ncells, 4) tet vertex idx
        if 'cauchy_stress' in f.get('cell_fields', {}):
            stress = f['cell_fields/cauchy_stress'][0]  # (ncells, 6) Voigt
            centers = verts[cells].mean(axis=1)          # cell centroids (km)
        else:
            stress = f['vertex_fields/cauchy_stress'][0]  # (nverts, 6) Voigt
            centers = verts                                 # vertex positions (km)

    d = np.linalg.norm(centers - fault_centroid, axis=1)
    sel = d < select_radius_km
    if not np.any(sel):
        raise RuntimeError(
            f'No crust cells/vertices within {select_radius_km} km of fault '
            f'centroid {fault_centroid}; widen select_radius_km.')

    s = stress[sel].mean(axis=0) * STRESS_SCALE      # true Pa, Voigt order
    # PyLith Voigt order: [xx, yy, zz, yz, xz, xy].
    sigma = np.array([
        [s[0], s[5], s[4]],
        [s[5], s[1], s[3]],
        [s[4], s[3], s[2]],
    ])
    n = fault_normal / np.linalg.norm(fault_normal)
    t = sigma @ n                       # traction vector on the plane
    sigma_n = float(n @ t)              # normal stress (tension positive)
    tau_vec = t - sigma_n * n
    tau = float(np.linalg.norm(tau_vec))
    # Coulomb stress change: tau - mu * sigma_n_effective.
    # Compression-negative convention -> clamping sign handled by caller.
    coulomb = tau + MU_FRICTION * sigma_n   # sigma_n tension-positive
    return sigma_n, tau, coulomb, tau_vec, n


def slip_from_stress(tau_shear_Pa):
    """Convert a resolved shear traction (Pa) into a prescribed slip (m) using
    the user-specified stress-to-slip relation."""
    if STRESS_TO_SLIP is None:
        raise RuntimeError(
            'STRESS_TO_SLIP is not set. The stress-to-slip relation and its '
            'parameter(s) must be supplied by the user before Pass 2 can run.')
    mode = STRESS_TO_SLIP['mode']
    if mode == 'compliance':
        return STRESS_TO_SLIP['compliance_m_per_Pa'] * tau_shear_Pa
    if mode == 'dislocation':
        # slip = stress_drop * char_length / mu_shear ; here we take the resolved
        # shear traction as the stress drop and use the shared 30 GPa shear mod.
        mu_shear = 30.0e9
        return tau_shear_Pa * STRESS_TO_SLIP['char_length_m'] / mu_shear
    raise ValueError(f'Unknown STRESS_TO_SLIP mode: {mode!r}')


# ── Pass 2: write the full (with-fault) scenario cfg ───────────────────────────
def write_pass2_cfg(label, pf_amp_Pa, slip_e_m, slip_w_m, dike_opening_m=0.0):
    """Pass 2: full poroelastic + fault model with stress-derived ring-fault slip
    and a fixed (not stress-derived) dike opening. dike_opening_m=0.0 (default)
    is a no-op for the generic SWEEP_PF_AMPLITUDES sweep, which doesn't use the
    dike at all."""
    cfg_path = os.path.join(TMPDIR, f'{label}_pass2.cfg')
    outbase = os.path.join(OUTDIR, label)
    dike_observer_block = '\n'.join(
        f"[pylithapp.problem.interfaces.dike{i}.observers.observer]\n"
        f"writer.filename = {outbase}-dike{i}.h5\n"
        for i in (1, 2, 3, 4))
    dike_rupture_block = '\n'.join(
        f"[pylithapp.problem.interfaces.dike{i}.eq_ruptures.rupture]\n"
        f"db_auxiliary_field.data = [{RUN_START_YR:.4f}*year, 0.0*m, 0.0*m, {dike_opening_m:.6f}*m]\n"
        for i in (1, 2, 3, 4))
    content = f"""# Auto-generated Pass-2 (full model) cfg: {label}
# Injected cavity fluid pressure = {pf_amp_Pa / 1e6:.3f} MPa
# Stress-derived slip: east = {slip_e_m:.4f} m, west = {slip_w_m:.4f} m
# Dike opening (fixed, not stress-derived, instantaneous -- see cfg/dike.cfg): {dike_opening_m:.4f} m
# (Normal faulting during inflation -> reverse slip set negative.)

[pylithapp.problem]
defaults.name = {label}

[pylithapp.problem.solution_observers.domain]
writer.filename = {outbase}-solution.h5

[pylithapp.problem.materials.crust.observers.observer]
writer.filename = {outbase}-material_crust.h5

[pylithapp.problem.interfaces.fault_east.observers.observer]
writer.filename = {outbase}-fault_east.h5

[pylithapp.problem.interfaces.fault_west.observers.observer]
writer.filename = {outbase}-fault_west.h5

{dike_observer_block}
# Injected fluid pressure at the Mogi cavity wall.
[pylithapp.problem.bc.mogi_deep_bc]
db_auxiliary_field.data = [{pf_amp_Pa / 1e6:.6f}*MPa]

# Stress-informed prescribed slip (ring faults).
[pylithapp.problem.interfaces.fault_east.eq_ruptures.rupture]
db_auxiliary_field.data = [0.0*s, 0.0*m, {slip_e_m:.6f}*m, 0.0*m]

[pylithapp.problem.interfaces.fault_west.eq_ruptures.rupture]
db_auxiliary_field.data = [0.0*s, 0.0*m, {slip_w_m:.6f}*m, 0.0*m]

# Fixed (not stress-derived), INSTANTANEOUS dike opening (see cfg/dike.cfg for why).
# Same opening applied uniformly to all 4 segments (see cfg/dike.cfg for why they're
# 4 separate interfaces, not one merged surface).
{dike_rupture_block}"""
    with open(cfg_path, 'w') as f:
        f.write(content)
    return cfg_path


# The shipped poroelastic solver cfgs use an `ml` (algebraic-multigrid)
# displacement preconditioner, which was extremely slow on this mesh. We force a
# direct LU displacement solve. Because the shipped solver cfg is passed AFTER
# the base cfgs, this override must be passed LAST so it wins.
def _write_lu_override():
    path = os.path.join(TMPDIR, 'solver_lu_override.cfg')
    with open(path, 'w') as f:
        f.write(
            "# Force a direct LU displacement solve (overrides shipped `ml`).\n"
            "[pylithapp.petsc]\n"
            "fieldsplit_displacement_pc_type  = lu\n"
            "fieldsplit_displacement_ksp_type = preonly\n")
    return path


LU_OVERRIDE = _write_lu_override()


def _run_pylith(cfgs, solver_cfg, label, dry_run, log_path=None):
    cmd = [PYLITH_BIN, PYLITH_SCRIPT] + cfgs + [solver_cfg, LU_OVERRIDE]
    if dry_run:
        print(f'  [DRY RUN] {" ".join(os.path.basename(c) for c in cfgs)} '
              f'+ {os.path.basename(solver_cfg)}')
        return True
    if log_path:
        print(f'  {label}: logging to {log_path}')
        with open(log_path, 'w') as logf:
            result = subprocess.run(cmd, cwd=PROJ, env=PYLITH_ENV,
                                    stdout=logf, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(cmd, cwd=PROJ, env=PYLITH_ENV)
    if result.returncode != 0:
        print(f'  {label}: FAILED (exit {result.returncode})')
        return False
    return True


def run_scenario(pf_amp_Pa, dry_run=False, pass1_only=False):
    label = f'axial_Pf{pf_amp_Pa / 1e6:05.1f}MPa'
    print(f'\n=== {label}  (injected Pf = {pf_amp_Pa / 1e6:.1f} MPa) ===')

    # Pass 1: no-fault inflation.
    cfg1 = write_pass1_cfg(label, pf_amp_Pa)
    log1 = os.path.join(OUTDIR, f'{label}_pass1.log')
    if not _run_pylith(BASE_CFGS_NOFAULT + [cfg1], SOLVER_NOFAULT,
                       f'{label}/pass1', dry_run, log_path=log1):
        return False

    if dry_run:
        # Skip stress resolution / Pass 2 in a dry run (no Pass-1 output exists).
        print('  [DRY RUN] would resolve fault stress and run Pass 2')
        return True

    if pass1_only:
        print('  [PASS1-ONLY] skipping stress resolution and Pass 2')
        return True

    # Resolve Coulomb stress onto each fault plane.
    mat_h5 = os.path.join(OUTDIR, f'{label}_pass1-material_crust.h5')
    resolved = {}
    for name, geom in FAULTS.items():
        sn, tau, coulomb, tau_vec, n = resolve_fault_stress(
            mat_h5, geom['normal'], geom['centroid'])
        resolved[name] = {'sigma_n_Pa': sn, 'tau_Pa': tau, 'coulomb_Pa': coulomb}
        print(f'  fault_{name}: sigma_n={sn / 1e6:+.2f} MPa  '
              f'tau={tau / 1e6:.2f} MPa  dCFS={coulomb / 1e6:+.2f} MPa')

    # Convert resolved shear traction -> slip. Inflation drives normal faulting,
    # so reverse slip is set negative (matches faults.cfg convention).
    slip_e = -slip_from_stress(resolved['east']['tau_Pa'])
    slip_w = -slip_from_stress(resolved['west']['tau_Pa'])
    print(f'  derived slip: east={slip_e:.4f} m  west={slip_w:.4f} m')

    # Persist the resolved-stress / slip provenance.
    with open(os.path.join(OUTDIR, f'{label}-parameters.json'), 'w') as f:
        json.dump({'label': label, 'pf_amplitude_Pa': pf_amp_Pa,
                   'mu_friction': MU_FRICTION,
                   'stress_to_slip': STRESS_TO_SLIP,
                   'resolved': resolved,
                   'slip_east_m': slip_e, 'slip_west_m': slip_w}, f, indent=2)

    # Pass 2: full model with derived slip.
    cfg2 = write_pass2_cfg(label, pf_amp_Pa, slip_e, slip_w)
    log2 = os.path.join(OUTDIR, f'{label}_pass2.log')
    if not _run_pylith(BASE_CFGS_FULL + [cfg2], SOLVER_FAULT,
                       f'{label}/pass2', dry_run, log_path=log2):
        return False
    print(f'  -> output/{label}-*.h5')
    return True


def run_baillard_scenario(name, pf_amp_Pa, dike_opening_m, dry_run=False, pass1_only=False):
    """Run one named Baillard-geometry scenario (pre or syn), reusing the same
    two-pass resolve-onto-ring-faults machinery as run_scenario(), but with the
    dike's fixed opening threaded through both passes."""
    label = name
    print(f'\n=== {label}  (injected Pf = {pf_amp_Pa / 1e6:+.1f} MPa, '
          f'dike opening = {dike_opening_m:.2f} m) ===')

    # Pass 1 needs dike.cfg loaded (BASE_CFGS_NOFAULT_DIKE) so the dike interface's
    # label/edge/ref_dir/etc. are defined, but NOT faults.cfg (ring faults) --
    # see BASE_CFGS_NOFAULT_DIKE's comment for why the ring faults' cfg file must
    # be entirely absent while they're excluded from Pass 1's active interfaces.
    # Uses SOLVER_FAULT (not SOLVER_NOFAULT) since the dike is a real
    # FaultCohesiveKin interface here -- it needs the Lagrange-multiplier
    # field-split block and cohesive-DOF reordering that solver provides, which
    # the plain no-fault solver settings don't configure.
    cfg1 = write_pass1_cfg(label, pf_amp_Pa, dike_opening_m=dike_opening_m)
    log1 = os.path.join(OUTDIR, f'{label}_pass1.log')
    if not _run_pylith(BASE_CFGS_NOFAULT_DIKE + [cfg1], SOLVER_FAULT,
                       f'{label}/pass1', dry_run, log_path=log1):
        return False

    if dry_run:
        print('  [DRY RUN] would resolve fault stress and run Pass 2')
        return True
    if pass1_only:
        print('  [PASS1-ONLY] skipping stress resolution and Pass 2')
        return True

    mat_h5 = os.path.join(OUTDIR, f'{label}_pass1-material_crust.h5')
    resolved = {}
    for fname, geom in FAULTS.items():
        sn, tau, coulomb, tau_vec, n = resolve_fault_stress(
            mat_h5, geom['normal'], geom['centroid'])
        resolved[fname] = {'sigma_n_Pa': sn, 'tau_Pa': tau, 'coulomb_Pa': coulomb}
        print(f'  fault_{fname}: sigma_n={sn / 1e6:+.2f} MPa  '
              f'tau={tau / 1e6:.2f} MPa  dCFS={coulomb / 1e6:+.2f} MPa')

    slip_e = -slip_from_stress(resolved['east']['tau_Pa'])
    slip_w = -slip_from_stress(resolved['west']['tau_Pa'])
    print(f'  derived slip: east={slip_e:.4f} m  west={slip_w:.4f} m')

    with open(os.path.join(OUTDIR, f'{label}-parameters.json'), 'w') as f:
        json.dump({'label': label, 'pf_amplitude_Pa': pf_amp_Pa,
                   'dike_opening_m': dike_opening_m,
                   'mu_friction': MU_FRICTION,
                   'stress_to_slip': STRESS_TO_SLIP,
                   'resolved': resolved,
                   'slip_east_m': slip_e, 'slip_west_m': slip_w}, f, indent=2)

    cfg2 = write_pass2_cfg(label, pf_amp_Pa, slip_e, slip_w, dike_opening_m=dike_opening_m)
    log2 = os.path.join(OUTDIR, f'{label}_pass2.log')
    if not _run_pylith(BASE_CFGS_FULL + [cfg2], SOLVER_FAULT,
                       f'{label}/pass2', dry_run, log_path=log2):
        return False
    print(f'  -> output/{label}-*.h5')
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pylith', default=PYLITH_DIR,
                   help='PyLith install dir (default: hard-coded path)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--pass1-only', action='store_true',
                   help='Run only the no-fault Pass 1 solve (diagnostic).')
    p.add_argument('--baillard-simple', action='store_true',
                   help='Run the two named Baillard pre/syn scenarios (BAILLARD_SCENARIOS) '
                        'instead of the generic SWEEP_PF_AMPLITUDES sweep.')
    args = p.parse_args()

    if args.baillard_simple:
        print(f'Baillard-geometry pre/syn scenarios  |  {len(BAILLARD_SCENARIOS)} scenarios')
        ok = sum(run_baillard_scenario(s['name'], s['pf_amp_Pa'], s['dike_opening_m'],
                                       args.dry_run, args.pass1_only)
                for s in BAILLARD_SCENARIOS)
        print(f'\nCompleted {ok}/{len(BAILLARD_SCENARIOS)}')
        return

    if SWEEP_PF_AMPLITUDES is None:
        sys.exit(
            'STOP: SWEEP_PF_AMPLITUDES is not set. The range of injected cavity '
            'fluid-pressure amplitudes for the poroelastic Mogi source has not '
            'been specified. Set it in run_sweep.py (see REQUIRES USER INPUT 1/2) '
            'before running the sweep.')
    if STRESS_TO_SLIP is None and not args.dry_run and not args.pass1_only:
        sys.exit(
            'STOP: STRESS_TO_SLIP is not set. The stress-to-slip relation needed '
            'to turn Pass-1 resolved fault stress into prescribed slip has not '
            'been specified. Set it in run_sweep.py (see REQUIRES USER INPUT 2/2) '
            'before running Pass 2.')

    amps = SWEEP_PF_AMPLITUDES
    print(f'Poroelastic two-pass sweep  |  {len(amps)} fluid-pressure amplitudes')
    ok = sum(run_scenario(a, args.dry_run, args.pass1_only) for a in amps)
    print(f'\nCompleted {ok}/{len(amps)}')


if __name__ == '__main__':
    main()
