#!/usr/bin/env nemesis
"""
Generate a tet mesh for Axial Seamount ring-fault / Yang-spheroid / Okada-dike PyLith
simulation.

Geometry (x=East km, y=North km, z=Up km):
  Domain   : [0,16] x [-1,17] x [-10, 0.1]
  Fault E  : N30W strike (az 330), 66 deg dip toward E (dip az 60),
             center (8.5, 4.0), 8 km along-strike x 3.25 km down-dip, top at z=-0.1
  Fault W  : same strike, 66 deg dip toward W (dip az 240), center (7.0, 4.0)
  Spheroid : Yang (1988) prolate spheroid pressure source (Baillard geometry, matches
             scripts/baillard_simple_model.py's SPH_* constants and
             Axial_Deformation/axial_comb_V0.m's sph_x0/y0/z0/a/b/theta/phi, which are
             fixed across every MATLAB scenario): center (8.84, 5.38) km, depth 3.81 km,
             semi-major axis 2.2 km, semi-minor axis 0.38 km, major axis dips 77 deg from
             horizontal toward azimuth 286 deg (CW from North). Replaces the old single
             spherical Mogi cavity (S1) -- see git history for the prior sphere geometry.
  Dike     : Real multi-segment propagating rupture (user-specified 2026-07-22 session
             geometry, NOT from baillard_simple_model.py's simplified single-segment
             fissure or from any literature/repo trace data -- confirmed via repo search
             that no North-Rift-Zone/2015-dike-trace data exists anywhere in this repo,
             so these are the user's own directly-specified values). 5 waypoints define 4
             connected vertical planar segments, depth 0 to -0.5 km throughout:
               P0 = (9.620, 5.519) km -- 0.3 km east of AXEC1 (9.320, 5.519)
               P1 = (9.620, 4.519) km -- P0, 1 km south (az 180)
               P2 = (8.572, 10.464) km -- reversed, az 350, until 5 km past AXEC1
                    (segment length 6.037 km)
               P3 = (7.072, 10.464) km -- P2, 1.5 km west
               P4 = (9.072, 13.928) km -- P3, 4 km at az 30 (N-NE, into the North Rift Zone)
             Total path length ~12.54 km. Verified (see session notes) to NOT
             geometrically overlap the spheroid (min ellipsoid-distance value 13.3, >>1)
             or either ring fault (min separation 0.96 km from fault_east, 3.48 km from
             fault_west centroid).

Physical groups (must match cfg label / label_value):
  dim=3  tag=1   -> volume material (MaterialGroup, no string name needed)
  dim=2  tag=10  -> boundary_xneg
  dim=2  tag=11  -> boundary_xpos
  dim=2  tag=12  -> boundary_yneg
  dim=2  tag=13  -> boundary_ypos
  dim=2  tag=14  -> boundary_zneg
  dim=2  tag=15  -> boundary_zpos  (free surface)
  dim=2  tag=20  -> fault_east
  dim=2  tag=21  -> fault_west
  dim=1  tag=120 -> fault_east_edge  (buried perimeter of eastern fault)
  dim=1  tag=121 -> fault_west_edge  (buried perimeter of western fault)
  dim=2  tag=30  -> mogi_deep        (Yang spheroid cavity wall; label kept from the old
                                       sphere so cfg/bc.cfg's mogi_deep_bc needs no changes)
  dim=2  tag=40  -> dike1            (P0->P1 segment, Okada dike opening surface)
  dim=2  tag=41  -> dike2            (P1->P2 segment)
  dim=2  tag=42  -> dike3            (P2->P3 segment)
  dim=2  tag=43  -> dike4            (P3->P4 segment)
  dim=1  tag=140 -> dike1_edge       (buried perimeter of dike1)
  dim=1  tag=141 -> dike2_edge       (buried perimeter of dike2)
  dim=1  tag=142 -> dike3_edge       (buried perimeter of dike3)
  dim=1  tag=143 -> dike4_edge       (buried perimeter of dike4)

  NOTE on why the dike is 4 SEPARATE interfaces, not one continuous "dike" surface (as an
  earlier version of this file had): PyLith's automatic cohesive-cell insertion
  (DMPlexLabelCohesiveCheck) requires a fault interface to have a single consistent normal
  orientation across its whole surface. This dike's path has two sharp junctions -- P1
  (where it reverses from heading south to heading almost due north again, only a ~10 deg
  interior angle -- effectively a fold-back) and P2 (an 80 deg corner) -- and treating all
  4 segments as ONE continuous FaultCohesiveKin interface crashed with "Invalid cohesive
  label" exactly at those two junctions (confirmed via the PyLith error log: "Impinging
  point" warnings and the fatal error both localized to P1's and P2's coordinates). Modeling
  each straight segment as its own independent interface sidesteps the problem entirely --
  each is a simple planar quad with no internal kinks.

Run with:
  $PYLITH_DIR/bin/nemesis generate_mesh.py
  python generate_mesh.py
"""

import math, sys, os
import gmsh

# Use PyLith gmsh_utils if available
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    '../../../Downloads/pylith-5.0.1-macOS-12.0-arm64'
                                    '/lib/python3.12/site-packages'))
    from pylith.meshio.gmsh_utils import MaterialGroup, BoundaryGroup, create_group
    HAS_PYLITH = True
except ImportError:
    HAS_PYLITH = False
    def create_group(name, tag, dim, entities, recursive=False, exclude=None):
        gmsh.model.add_physical_group(dim, entities, tag)
        gmsh.model.set_physical_name(dim, tag, name)

# ── Geometry constants (km) ───────────────────────────────────────────────────
STRIKE_RAD = math.radians(330.0)   # N30W
DIP_RAD    = math.radians(66.0)
HALF_LEN   = 4.0     # km half-length along strike
DIP_WIDTH  = 3.25    # km down-dip width (fault_west; see DIP_WIDTH_EAST below)
TOP_Z      = -0.1    # km, top of fault

# fault_east's full 3.25 km down-dip width geometrically overlaps the Yang spheroid
# below (verified numerically: with DIP_WIDTH=3.25 the fault rectangle enters the
# spheroid across down-dip depth ~1.75-2.73 km, along-strike offset ~1.15-1.43 km from
# its center). Trimmed to 1.5 km down-dip -- comfortably clear of the spheroid
# (nearest approach still corresponds to ellipsoid-distance value ~1.19, i.e. >1 =
# outside -- see git history/session notes for the check) -- while keeping the full
# 8 km along-strike length and the shallower (0-1.5 km down-dip) part of the fault.
DIP_WIDTH_EAST = 1.5   # km down-dip width, fault_east only (shrunk to avoid the spheroid)

X0, X1 = 0.0,  16.0
Y0, Y1 = -1.0, 17.0
Z0, Z1 = -10.0, 0.1

# Yang (1988) prolate spheroid (Baillard geometry -- see module docstring)
SPH_X0, SPH_Y0   = 8.84, 5.38     # km, centre (East, North)
SPH_DEPTH        = 3.81           # km, depth below seafloor
SPH_Z0           = -SPH_DEPTH     # km, z-up
SPH_A, SPH_B     = 2.2, 0.38      # km, semi-major / semi-minor axes
SPH_DIP          = 77.0           # deg from horizontal (90 = vertical)
SPH_AZIMUTH      = 286.0          # deg CW from North

# Real multi-segment propagating dike (see module docstring for provenance/derivation).
# 5 waypoints -> 4 connected vertical planar segments.
DIKE_WAYPOINTS = [
    (9.620, 5.519),    # P0: 0.3 km east of AXEC1
    (9.620, 4.519),    # P1: 1 km south of P0
    (8.572, 10.464),   # P2: reversed, az 350, 5 km past AXEC1 (segment length 6.037 km)
    (7.072, 10.464),   # P3: 1.5 km west of P2
    (9.072, 13.928),   # P4: 4 km at az 30 (N-NE) from P3, into the North Rift Zone
]
DIKE_ZTOP  = 0.0            # km, z-up (depth 0 = surface)
DIKE_ZBOT  = -0.5           # km, z-up (depth 0.5 km)

DX_FAULT  = 0.30   # km, element size at fault/source
DX_FAR    = 2.0    # km, far-field element size
DIST_MIN  = 0.5    # km
DIST_MAX  = 4.0    # km


def fault_corners(cx, cy, dip_az_deg, dip_width=DIP_WIDTH):
    """Return (p1,p2,p3,p4) corners of fault rectangle (z-up, km)."""
    s = (math.sin(STRIKE_RAD), math.cos(STRIKE_RAD), 0.0)
    az = math.radians(dip_az_deg)
    d = (math.sin(az)*math.cos(DIP_RAD),
         math.cos(az)*math.cos(DIP_RAD),
         -math.sin(DIP_RAD))
    p1 = (cx + HALF_LEN*s[0], cy + HALF_LEN*s[1], TOP_Z)
    p2 = (cx - HALF_LEN*s[0], cy - HALF_LEN*s[1], TOP_Z)
    p3 = (p2[0]+dip_width*d[0], p2[1]+dip_width*d[1], TOP_Z+dip_width*d[2])
    p4 = (p1[0]+dip_width*d[0], p1[1]+dip_width*d[1], TOP_Z+dip_width*d[2])
    return p1, p2, p3, p4


def dike_corners(start_xy, end_xy, z_top, z_bot):
    """Return (p1,p2,p3,p4) corners of a VERTICAL planar rectangle spanning a dike
    fissure trace (z-up, km) -- Okada's classic vertical tensile dike has no dip tilt."""
    (x1, y1), (x2, y2) = start_xy, end_xy
    p1 = (x1, y1, z_top)
    p2 = (x2, y2, z_top)
    p3 = (x2, y2, z_bot)
    p4 = (x1, y1, z_bot)
    return p1, p2, p3, p4


def make_fault_surface(corners):
    """Create and return a planar quadrilateral surface tag."""
    pts   = [gmsh.model.occ.add_point(*c) for c in corners]
    lines = [gmsh.model.occ.add_line(pts[i], pts[(i+1) % 4]) for i in range(4)]
    loop  = gmsh.model.occ.add_curve_loop(lines)
    return gmsh.model.occ.add_plane_surface([loop])


def surfaces_near(cx, cy, cz, radius, exclude_tags=()):
    """Return surface tags whose centroid lies within radius of (cx,cy,cz)."""
    result = []
    for _, tag in gmsh.model.get_entities(2):
        if tag in exclude_tags:
            continue
        mx, my, mz = gmsh.model.occ.get_center_of_mass(2, tag)
        d = math.sqrt((mx-cx)**2 + (my-cy)**2 + (mz-cz)**2)
        if d < radius:
            result.append(tag)
    return result


def bbox_surfaces(xmin, xmax, ymin, ymax, zmin, zmax, tol=0.05):
    """Return (dim,tag) pairs for surfaces whose centroid is inside the bounding box."""
    return [t for _, t in
            gmsh.model.get_entities_in_bounding_box(
                xmin-tol, ymin-tol, zmin-tol,
                xmax+tol, ymax+tol, zmax+tol, dim=2)]


def perimeter_curves(surf_tags, domain_curve_set):
    """Return all 1D curves bounding surf_tags that are NOT in domain_curve_set."""
    curves = set()
    for tag in surf_tags:
        _, down = gmsh.model.get_adjacencies(2, tag)
        curves.update(down)
    return sorted(curves - domain_curve_set)


def main():
    gmsh.initialize()
    gmsh.model.add("axial_seamount")
    occ = gmsh.model.occ

    # ── 1. Domain box ──────────────────────────────────────────────────────────
    v_box = occ.add_box(X0, Y0, Z0, X1-X0, Y1-Y0, Z1-Z0)

    # ── 2. Hollow out the Yang prolate-spheroid cavity ────────────────────────
    # Build as a sphere of radius = semi-minor axis, stretch along its own (initially
    # vertical) axis to the semi-major/semi-minor ratio, then rotate that axis to the
    # dip/azimuth orientation before cutting from the domain box.
    v_s1 = occ.add_sphere(SPH_X0, SPH_Y0, SPH_Z0, SPH_B)
    occ.dilate([(3, v_s1)], SPH_X0, SPH_Y0, SPH_Z0, 1.0, 1.0, SPH_A / SPH_B)

    az_rad, dip_rad = math.radians(SPH_AZIMUTH), math.radians(SPH_DIP)
    axis_dir = (math.sin(az_rad) * math.cos(dip_rad),
                math.cos(az_rad) * math.cos(dip_rad),
                math.sin(dip_rad))
    # Rotate from the initial vertical axis (0,0,1) to axis_dir: rotate about
    # cross((0,0,1), axis_dir) = (-axis_dir_y, axis_dir_x, 0) by the angle between them.
    rot_axis = (-axis_dir[1], axis_dir[0], 0.0)
    rot_norm = math.hypot(rot_axis[0], rot_axis[1])
    if rot_norm > 1e-9:
        rot_axis = (rot_axis[0] / rot_norm, rot_axis[1] / rot_norm, 0.0)
        rot_angle = math.acos(max(-1.0, min(1.0, axis_dir[2])))
        occ.rotate([(3, v_s1)], SPH_X0, SPH_Y0, SPH_Z0,
                   rot_axis[0], rot_axis[1], rot_axis[2], rot_angle)

    crust_out, _ = occ.cut([(3, v_box)], [(3, v_s1)],
                            removeObject=True, removeTool=True)
    v_crust = crust_out[0][1]

    # ── 3. Create fault + dike plane surfaces ─────────────────────────────────
    s_fe = make_fault_surface(fault_corners(8.5, 4.0,  60.0, dip_width=DIP_WIDTH_EAST))  # eastern (shrunk to avoid the spheroid)
    s_fw = make_fault_surface(fault_corners(7.0, 4.0, 240.0))  # western
    s_dk_segments = [
        make_fault_surface(dike_corners(DIKE_WAYPOINTS[i], DIKE_WAYPOINTS[i + 1], DIKE_ZTOP, DIKE_ZBOT))
        for i in range(len(DIKE_WAYPOINTS) - 1)
    ]

    # ── 4. Fragment domain along each fault/dike segment (sequential, removeTool=True) ─
    res1, _ = occ.fragment([(3, v_crust)], [(2, s_fe)],
                            removeObject=True, removeTool=True)
    v_tags_cur = [t for d, t in res1 if d == 3]

    res2, _ = occ.fragment([(3, v) for v in v_tags_cur], [(2, s_fw)],
                            removeObject=True, removeTool=True)
    v_tags_cur = [t for d, t in res2 if d == 3]

    for s_dk in s_dk_segments:
        res_dk, _ = occ.fragment([(3, v) for v in v_tags_cur], [(2, s_dk)],
                                 removeObject=True, removeTool=True)
        v_tags_cur = [t for d, t in res_dk if d == 3]
    v_tags_final = v_tags_cur

    occ.synchronize()

    # ── 5. Identify surfaces by bounding box ──────────────────────────────────
    tol = 0.05
    s_xneg = bbox_surfaces(X0, X0, Y0, Y1, Z0, Z1)
    s_xpos = bbox_surfaces(X1, X1, Y0, Y1, Z0, Z1)
    s_yneg = bbox_surfaces(X0, X1, Y0, Y0, Z0, Z1)
    s_ypos = bbox_surfaces(X0, X1, Y1, Y1, Z0, Z1)
    s_zneg = bbox_surfaces(X0, X1, Y0, Y1, Z0, Z0)
    s_zpos = bbox_surfaces(X0, X1, Y0, Y1, Z1, Z1)

    domain_surfs = set(s_xneg + s_xpos + s_yneg + s_ypos + s_zneg + s_zpos)

    # Collect curves belonging to domain boundary surfaces
    domain_curves = set()
    for tag in domain_surfs:
        _, down = gmsh.model.get_adjacencies(2, tag)
        domain_curves.update(down)

    # Spheroid cavity wall: centroid within 1.1x the semi-major axis of the source centre.
    # No true ellipsoid surface point is farther than SPH_A from the centre, so this is
    # already a generous margin -- NOTE: a much larger radius (previously 1.5x = 3.3 km)
    # was tried first and incorrectly swept up fault_east's own (trimmed) surface, whose
    # centroid sits ~3.27 km from the spheroid centre despite no longer geometrically
    # overlapping it (see DIP_WIDTH_EAST above) -- keep this radius comfortably below
    # that fault distance.
    s_mogi1 = surfaces_near(SPH_X0, SPH_Y0, SPH_Z0, SPH_A * 1.1, exclude_tags=tuple(domain_surfs))
    mogi_surfs = set(s_mogi1)

    # Remaining interior surfaces (fault_east, fault_west, and the 4 dike segments) are
    # classified together in ONE joint nearest-object assignment, not via any radius-based
    # candidate-gathering step. Two radius-based attempts were tried and both failed for
    # this geometry: (1) per-segment "within radius R of segment centroid" -- segment 2
    # (P1->P2, 6.037 km long) needs R~3.3 km to reach its own far end, which is ALSO close
    # enough to reach segment 3's centroid (3.24 km away), so segment 2 silently claimed
    # ALL of segment 3's surfaces when processed first (confirmed empirically: "Dike3
    # surfaces: []"); (2) a single big radius around the whole path's bounding region --
    # that radius (~5.4 km, needed to span the 12.5 km path) also reached fault_east's
    # centroid (~4.9 km away), sweeping in fault surfaces too. Both failure modes share the
    # same root cause: a per-object radius sized to reach a large/far object can always
    # accidentally reach a smaller/nearer neighboring object. The fix: skip radius
    # candidate-gathering entirely and directly compute, for every remaining surface, its
    # distance to EACH of the 6 real candidates (4 dike segment lines via 2D
    # point-to-line-segment distance -- exact for these vertical planar quads -- plus the 2
    # fault centroids via 3D distance, same as originally used), then assign to whichever
    # of the 6 is closest. No radius, so no possibility of one object's reach overlapping
    # another's territory.
    def _point_to_segment_dist_2d(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            return math.hypot(px - x1, py - y1)
        t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
        cx, cy = x1 + t * dx, y1 + t * dy
        return math.hypot(px - cx, py - cy)

    de = (math.sin(math.radians(60))*math.cos(DIP_RAD),
          math.cos(math.radians(60))*math.cos(DIP_RAD),
          -math.sin(DIP_RAD))
    dw = (math.sin(math.radians(240))*math.cos(DIP_RAD),
          math.cos(math.radians(240))*math.cos(DIP_RAD),
          -math.sin(DIP_RAD))
    fe_c = (8.5 + 0.5*DIP_WIDTH_EAST*de[0], 4.0 + 0.5*DIP_WIDTH_EAST*de[1], TOP_Z + 0.5*DIP_WIDTH_EAST*de[2])
    fw_c = (7.0 + 0.5*DIP_WIDTH*dw[0], 4.0 + 0.5*DIP_WIDTH*dw[1], TOP_Z + 0.5*DIP_WIDTH*dw[2])

    n_seg = len(DIKE_WAYPOINTS) - 1
    dike_seg_surfs = [[] for _ in range(n_seg)]
    s_fault_e, s_fault_w = [], []
    dike_z_mid = 0.5 * (DIKE_ZTOP + DIKE_ZBOT)
    for _, tag in gmsh.model.get_entities(2):
        if tag in domain_surfs or tag in mogi_surfs:
            continue
        cx, cy, cz = gmsh.model.occ.get_center_of_mass(2, tag)
        dike_dists = [_point_to_segment_dist_2d(cx, cy, *DIKE_WAYPOINTS[i], *DIKE_WAYPOINTS[i + 1])
                     for i in range(n_seg)]
        dist_e = math.sqrt((cx-fe_c[0])**2 + (cy-fe_c[1])**2 + (cz-fe_c[2])**2)
        dist_w = math.sqrt((cx-fw_c[0])**2 + (cy-fw_c[1])**2 + (cz-fw_c[2])**2)
        best_dike = min(range(n_seg), key=lambda i: dike_dists[i])
        all_dists = dike_dists + [dist_e, dist_w]
        best_overall = min(range(len(all_dists)), key=lambda i: all_dists[i])
        if best_overall < n_seg:
            dike_seg_surfs[best_overall].append(tag)
        elif best_overall == n_seg:
            s_fault_e.append(tag)
        else:
            s_fault_w.append(tag)

    dike_seg_surfs = [sorted(s) for s in dike_seg_surfs]
    dike_surfs = set(tag for seg in dike_seg_surfs for tag in seg)
    s_dike = sorted(dike_surfs)   # union, used only for mesh sizing / diagnostics below

    # Buried-perimeter edge curves (all perimeter curves not on the domain boundary),
    # computed PER dike segment (each is now its own interface with its own perimeter,
    # including the edge where it meets its neighbor -- see module docstring).
    e_curves = perimeter_curves(s_fault_e, domain_curves)
    w_curves = perimeter_curves(s_fault_w, domain_curves)
    dk_curves_per_seg = [perimeter_curves(seg_surfs, domain_curves) for seg_surfs in dike_seg_surfs]

    print(f"Volume tags        : {v_tags_final}")
    print(f"Boundary xneg/xpos : {s_xneg} / {s_xpos}")
    print(f"Boundary yneg/ypos : {s_yneg} / {s_ypos}")
    print(f"Boundary zneg/zpos : {s_zneg} / {s_zpos}")
    print(f"Spheroid surfaces  : {s_mogi1}")
    for i, (seg_surfs, seg_curves) in enumerate(zip(dike_seg_surfs, dk_curves_per_seg)):
        print(f"Dike{i+1} surfaces     : {seg_surfs}  perimeter curves: {seg_curves}")
    print(f"Fault E surfaces   : {s_fault_e}  perimeter curves: {e_curves}")
    print(f"Fault W surfaces   : {s_fault_w}  perimeter curves: {w_curves}")

    # ── 6. Physical groups ────────────────────────────────────────────────────
    # Volume: use MaterialGroup so name is "material-id:1" (PyLith convention)
    if HAS_PYLITH:
        MaterialGroup(tag=1, entities=v_tags_final).create_physical_group()
    else:
        gmsh.model.add_physical_group(3, v_tags_final, 1)
        gmsh.model.set_physical_name(3, 1, "material-id:1")

    # Domain boundary surfaces — BoundaryGroup, no recursion
    # (DirichletBC only needs the surface, not sub-entities)
    create_group("boundary_xneg", 10, 2, s_xneg, recursive=False)
    create_group("boundary_xpos", 11, 2, s_xpos, recursive=False)
    create_group("boundary_yneg", 12, 2, s_yneg, recursive=False)
    create_group("boundary_ypos", 13, 2, s_ypos, recursive=False)
    create_group("boundary_zneg", 14, 2, s_zneg, recursive=False)
    create_group("boundary_zpos", 15, 2, s_zpos, recursive=False)

    # Fault surfaces — BoundaryGroup dim=2, no recursion (edges added separately below)
    create_group("fault_east",      20, 2, s_fault_e, recursive=False)
    create_group("fault_west",      21, 2, s_fault_w, recursive=False)

    # Dike segments — 4 SEPARATE physical groups/interfaces (see module docstring for why
    # a single merged "dike" group failed PyLith's cohesive-cell insertion at the sharp
    # P1/P2 junctions).
    for i, seg_surfs in enumerate(dike_seg_surfs):
        create_group(f"dike{i+1}", 40 + i, 2, seg_surfs, recursive=False)

    # Fault/dike edge groups (perimeter of buried interfaces) — dim=1, separate tags
    # Convention: edge_tag = interface_tag + 100
    create_group("fault_east_edge", 120, 1, e_curves,  recursive=False)
    create_group("fault_west_edge", 121, 1, w_curves,  recursive=False)
    for i, seg_curves in enumerate(dk_curves_per_seg):
        create_group(f"dike{i+1}_edge", 140 + i, 1, seg_curves, recursive=False)

    # Spheroid cavity wall — Neumann/pressure BC surface, no recursion. Kept the
    # "mogi_deep"/tag 30 label from the old sphere so cfg/bc.cfg needs no changes.
    create_group("mogi_deep",    30, 2, s_mogi1, recursive=False)

    # ── 7. Mesh sizing ────────────────────────────────────────────────────────
    gmsh.option.set_number("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.set_numbers(f_dist, "SurfacesList",
                                      s_fault_e + s_fault_w + s_dike + s_mogi1)

    f_thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.set_number(f_thresh, "InField",  f_dist)
    gmsh.model.mesh.field.set_number(f_thresh, "SizeMin",  DX_FAULT)
    gmsh.model.mesh.field.set_number(f_thresh, "SizeMax",  DX_FAR)
    gmsh.model.mesh.field.set_number(f_thresh, "DistMin",  DIST_MIN)
    gmsh.model.mesh.field.set_number(f_thresh, "DistMax",  DIST_MAX)
    gmsh.model.mesh.field.set_as_background_mesh(f_thresh)

    # ── 8. Generate mesh ──────────────────────────────────────────────────────
    print("Generating 3D tet mesh ...")
    gmsh.model.mesh.generate(3)

    # Quality improvement (mirrors GenerateMesh.improve_quality, 5 iterations)
    for i in range(5):
        gmsh.option.set_number("Mesh.OptimizeThreshold", 0.4 + 0.04*i)
        gmsh.model.mesh.optimize("Relocate3D")
        gmsh.model.mesh.optimize("")

    # ── 9. Write (format 4.1 binary — PyLith 5 standard) ─────────────────────
    gmsh.option.set_number("Mesh.MshFileVersion", 4.1)
    gmsh.option.set_number("Mesh.Binary", 1)
    outfile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "axial_seamount.msh")
    gmsh.write(outfile)

    # Verify: print element type summary
    etypes, etags, _ = gmsh.model.mesh.get_elements()
    type_names = {e: gmsh.model.mesh.get_element_properties(e)[0] for e in etypes}
    print(f"\nMesh written → {outfile}")
    print("Element summary:")
    for et, en in zip(etypes, etags):
        print(f"  type {et} ({type_names[et]}): {len(en)} elements")

    if "--gui" in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    main()
