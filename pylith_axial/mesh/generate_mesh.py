#!/usr/bin/env nemesis
"""
Generate a tet mesh for Axial Seamount ring-fault / Mogi-source PyLith simulation.

Geometry (x=East km, y=North km, z=Up km):
  Domain  : [0,16] x [-1,17] x [-10, 0.1]
  Fault E : N30W strike (az 330), 66 deg dip toward E (dip az 60),
            center (8.5, 4.0), 8 km along-strike x 3.25 km down-dip, top at z=-0.1
  Fault W : same strike, 66 deg dip toward W (dip az 240), center (7.0, 4.0)
  Mogi S1 : center (7.57, 4.55, -3.33), radius 0.43 km  [deep]
  Mogi S2 : center (7.53, 6.60, -1.25), radius 0.28 km  [shallow]

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
  dim=2  tag=30  -> mogi_deep        (S1 cavity wall)
  dim=2  tag=31  -> mogi_shallow     (S2 cavity wall)

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
DIP_WIDTH  = 3.25    # km down-dip width
TOP_Z      = -0.1    # km, top of fault

X0, X1 = 0.0,  16.0
Y0, Y1 = -1.0, 17.0
Z0, Z1 = -10.0, 0.1

S1 = (7.57, 4.55, -3.33); R1 = 0.43   # deep Mogi source (S2 excluded: intersects eastern fault)

DX_FAULT  = 0.30   # km, element size at fault/source
DX_FAR    = 2.0    # km, far-field element size
DIST_MIN  = 0.5    # km
DIST_MAX  = 4.0    # km


def fault_corners(cx, cy, dip_az_deg):
    """Return (p1,p2,p3,p4) corners of fault rectangle (z-up, km)."""
    s = (math.sin(STRIKE_RAD), math.cos(STRIKE_RAD), 0.0)
    az = math.radians(dip_az_deg)
    d = (math.sin(az)*math.cos(DIP_RAD),
         math.cos(az)*math.cos(DIP_RAD),
         -math.sin(DIP_RAD))
    p1 = (cx + HALF_LEN*s[0], cy + HALF_LEN*s[1], TOP_Z)
    p2 = (cx - HALF_LEN*s[0], cy - HALF_LEN*s[1], TOP_Z)
    p3 = (p2[0]+DIP_WIDTH*d[0], p2[1]+DIP_WIDTH*d[1], TOP_Z+DIP_WIDTH*d[2])
    p4 = (p1[0]+DIP_WIDTH*d[0], p1[1]+DIP_WIDTH*d[1], TOP_Z+DIP_WIDTH*d[2])
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

    # ── 2. Hollow out sphere cavities (Mogi sources) ───────────────────────────
    v_s1 = occ.add_sphere(*S1, R1)
    crust_out, _ = occ.cut([(3, v_box)], [(3, v_s1)],
                            removeObject=True, removeTool=True)
    v_crust = crust_out[0][1]

    # ── 3. Create fault plane surfaces ────────────────────────────────────────
    s_fe = make_fault_surface(fault_corners(8.5, 4.0,  60.0))  # eastern
    s_fw = make_fault_surface(fault_corners(7.0, 4.0, 240.0))  # western

    # ── 4. Fragment domain along each fault (sequential, removeTool=True) ─────
    # Fragment eastern fault first
    res1, _ = occ.fragment([(3, v_crust)], [(2, s_fe)],
                            removeObject=True, removeTool=True)
    v_tags1 = [t for d, t in res1 if d == 3]

    # Fragment western fault through all resulting volumes
    res2, _ = occ.fragment([(3, v) for v in v_tags1], [(2, s_fw)],
                            removeObject=True, removeTool=True)
    v_tags_final = [t for d, t in res2 if d == 3]

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

    # Mogi cavity walls: centroid within 1.6×radius of source centre
    s_mogi1 = surfaces_near(*S1, R1*1.6, exclude_tags=tuple(domain_surfs))
    s_mogi2 = []
    mogi_surfs = set(s_mogi1)

    # Fault surfaces: all remaining interior surfaces not classified above
    # Eastern fault centroid (expected)
    de = (math.sin(math.radians(60))*math.cos(DIP_RAD),
          math.cos(math.radians(60))*math.cos(DIP_RAD),
          -math.sin(DIP_RAD))
    dw = (math.sin(math.radians(240))*math.cos(DIP_RAD),
          math.cos(math.radians(240))*math.cos(DIP_RAD),
          -math.sin(DIP_RAD))
    fe_c = (8.5 + 0.5*DIP_WIDTH*de[0], 4.0 + 0.5*DIP_WIDTH*de[1], TOP_Z + 0.5*DIP_WIDTH*de[2])
    fw_c = (7.0 + 0.5*DIP_WIDTH*dw[0], 4.0 + 0.5*DIP_WIDTH*dw[1], TOP_Z + 0.5*DIP_WIDTH*dw[2])

    s_fault_e, s_fault_w = [], []
    for _, tag in gmsh.model.get_entities(2):
        if tag in domain_surfs or tag in mogi_surfs:
            continue
        cx, cy, cz = gmsh.model.occ.get_center_of_mass(2, tag)
        dist_e = math.sqrt((cx-fe_c[0])**2 + (cy-fe_c[1])**2 + (cz-fe_c[2])**2)
        dist_w = math.sqrt((cx-fw_c[0])**2 + (cy-fw_c[1])**2 + (cz-fw_c[2])**2)
        if dist_e <= dist_w:
            s_fault_e.append(tag)
        else:
            s_fault_w.append(tag)

    # Fault perimeter edge curves (buried — all perimeter curves not on domain boundary)
    e_curves = perimeter_curves(s_fault_e, domain_curves)
    w_curves = perimeter_curves(s_fault_w, domain_curves)

    print(f"Volume tags        : {v_tags_final}")
    print(f"Boundary xneg/xpos : {s_xneg} / {s_xpos}")
    print(f"Boundary yneg/ypos : {s_yneg} / {s_ypos}")
    print(f"Boundary zneg/zpos : {s_zneg} / {s_zpos}")
    print(f"Mogi S1 surfaces   : {s_mogi1}")
    print(f"Mogi S2 surfaces   : {s_mogi2}")
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

    # Fault edge groups (perimeter of buried faults) — dim=1, separate tags
    # Convention: edge_tag = fault_tag + 100
    create_group("fault_east_edge", 120, 1, e_curves, recursive=False)
    create_group("fault_west_edge", 121, 1, w_curves, recursive=False)

    # Mogi cavity walls — Neumann BC surfaces, no recursion
    create_group("mogi_deep",    30, 2, s_mogi1, recursive=False)

    # ── 7. Mesh sizing ────────────────────────────────────────────────────────
    gmsh.option.set_number("Mesh.MeshSizeFromPoints",         0)
    gmsh.option.set_number("Mesh.MeshSizeFromCurvature",      0)
    gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)

    f_dist = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.set_numbers(f_dist, "SurfacesList",
                                      s_fault_e + s_fault_w + s_mogi1)

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
