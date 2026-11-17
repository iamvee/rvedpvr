import argparse
import hashlib
import json
import os
import sys
import time

from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abaqus_mesh as am
from fem_mesh import TAG, build_mesh, log, tag_facets
from fibers import compute_fibers
from loading import load_ramp, surface_tracker, volume_forms
from material import function_space, strain_energy
from morph import morph_coords
from outputs import PHYS_KEYS, geometry_records, probe_geometry, write_outputs
from solver import boundary_conditions, newton_solver, residual_forms, volumetric_error


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="1", help="base geometry (20k+1: 1,21,...,381)")
    ap.add_argument("--lvedp", type=float, default=30.0, help="LV peak pressure mmHg")
    ap.add_argument("--rvedp", type=float, default=10.0, help="RV peak pressure mmHg")
    ap.add_argument("--material", choices=["ho", "fung"], default="ho",
                    help="passive law: ho (reduced Holzapfel-Ogden) or fung (Guccione)")
    ap.add_argument("--a", type=float, default=0.10, help="isotropic matrix a (kPa)")
    ap.add_argument("--b", type=float, default=8.023, help="isotropic matrix b")
    ap.add_argument("--af", type=float, default=20.0, help="fiber a_f (kPa)")
    ap.add_argument("--bf", type=float, default=10.0, help="fiber b_f")
    ap.add_argument("--c", type=float, default=0.88, help="Fung scale c_region (kPa)")
    ap.add_argument("--B1", type=float, default=18.5, help="Fung fibre coeff B1 (b_ff)")
    ap.add_argument("--B2", type=float, default=3.58, help="Fung cross-fibre coeff B2 (b_tt)")
    ap.add_argument("--B3", type=float, default=1.63, help="Fung shear coeff B3 (b_fs)")
    ap.add_argument("--K", type=float, default=None,
                    help="Fung volumetric modulus, MPa; default 1e4*c")
    ap.add_argument("--qform", choices=["paper", "ti"], default="paper",
                    help="Fung shear weighting: paper (single E23^2) or ti (2*E23^2)")
    ap.add_argument("--endo", type=float, default=60.0, help="endo helix angle deg")
    ap.add_argument("--epi", type=float, default=-60.0, help="epi helix angle deg")
    ap.add_argument("--f_lv", type=float, default=1.0, help="LV cavity scale")
    ap.add_argument("--f_rv", type=float, default=1.0, help="RV cavity scale")
    ap.add_argument("--shift_lv", type=float, default=0.0, help="LV center shift toward RV (frac)")
    ap.add_argument("--shift_rv", type=float, default=0.0, help="RV center shift toward LV (frac)")
    ap.add_argument("--sept", type=float, default=0.0,
                    help="RV endocardium shift toward the free wall, fraction of the LV->RV axis")
    ap.add_argument("--kappa", type=float, default=0.003, help="bulk MPa (compressible)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--linesearch", choices=["basic", "bt", "l2", "cp"], default="basic",
                    help="SNES line search; basic relies on load sub-stepping")
    ap.add_argument("--expcap", type=float, default=40.0,
                    help="cap on the argument of the material exponentials")
    ap.add_argument("--ksp", choices=["direct", "amg"], default="direct",
                    help="linear solver: direct (MUMPS LU) or amg (GMRES + GAMG)")
    ap.add_argument("--u-order", type=int, default=1, choices=[1, 2],
                    help="displacement element order for the compressible formulation")
    ap.add_argument("--pstab", type=float, default=0.0,
                    help="perturbed-Lagrangian stiffness for the mixed formulations, kPa; 0 = auto")
    ap.add_argument("--mesh-dir", default=None,
                    help="directory of <geom>_{node,elem,set,surf}.txt; default mesh/mesh_files")
    ap.add_argument("--formulation", choices=["compressible", "incompressible", "hybrid"],
                    default="compressible")
    ap.add_argument("--name", default=None)
    ap.add_argument("--no-frames", action="store_true",
                    help="skip frames.npz/fibers.npz (numerical-only, ~4 KB/run)")
    ap.add_argument("--no-index", action="store_true",
                    help="skip results/INDEX.md append (use for parallel batch)")
    ap.add_argument("--probe", action="store_true",
                    help="measure the reference geometry only and stop")
    return ap.parse_args()


def main():
    args = parse_args()

    params = {k: getattr(args, k) for k in PHYS_KEYS}
    if args.material != "ho":
        params.update({"material": args.material, "c": args.c, "qform": args.qform,
                       "K": args.K,
                       "B1": args.B1, "B2": args.B2, "B3": args.B3})
    h = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]
    name = args.name or f"g{args.geom}_{h}"
    outdir = os.path.join("results", name)
    os.makedirs(outdir, exist_ok=True)
    comm = MPI.COMM_WORLD
    t0 = time.time()
    log(f"[run] {name}  params={params}")

    base = (os.path.join(args.mesh_dir, f"{args.geom}_") if args.mesh_dir
            else f"mesh/mesh_files/{args.geom}_")
    coords = am.read_nodes(base + "node.txt")
    cells = am.read_elems(base + "elem.txt")
    nsets = am.read_nsets(base + "set.txt")
    surfs = am.read_surfaces(base + "surf.txt", cells)
    log(f"[mesh] {coords.shape[0]} nodes  {cells.shape[0]} tets")

    quality = {"flipped_elems": 0}
    if args.f_lv != 1.0 or args.f_rv != 1.0 or args.shift_lv or args.shift_rv or args.sept:
        coords, quality = morph_coords(coords, cells, surfs, comm, args.f_lv, args.f_rv,
                                       args.shift_lv, args.shift_rv, args.sept)
        log(f"[morph] f_lv={args.f_lv} f_rv={args.f_rv} shift=({args.shift_lv},"
            f"{args.shift_rv}) sept={args.sept}  flipped={quality['flipped_elems']}  "
            f"min_vol_ratio={quality['min_abs_vol_ratio']:.3f}")
        if quality["flipped_elems"]:
            log("  WARNING: morph inverted elements; reduce f/shift/sept magnitude")

    if args.probe:
        probe_geometry(coords, surfs, nsets, quality, params, outdir, comm)
        return

    msh = build_mesh(coords, cells, comm)
    tags = tag_facets(msh, coords, surfs)
    log("[tags] " + str({k: int((tags.values == v).sum()) for k, v in TAG.items()}))

    base_axis = coords[nsets["top_nodes"]].mean(0) - coords.mean(0)
    F0, S0, N0, fib = compute_fibers(msh, tags, coords, cells, base_axis, args.endo, args.epi)
    log(f"[fibers] helix {args.endo:g} -> {args.epi:g} deg  ({time.time()-t0:.1f}s)")

    incompressible, W, w, u, p, v, du, Vdisp = function_space(msh, args)
    psi, Fdef, Jdet = strain_energy(msh, args, u, p, F0, S0, N0, incompressible)
    R, Jac, p_lv, p_rv, N, ds = residual_forms(msh, tags, psi, Jdet, Fdef, w, v, du)
    bcs = boundary_conditions(msh, Vdisp, coords, nsets)
    vols = volume_forms(msh, u, coords, surfs, Jdet, Fdef, N, ds, comm)
    snes = newton_solver(msh, R, Jac, w, bcs, args)
    S, capture = surface_tracker(msh, w, incompressible, coords, surfs)
    p_lv_l, p_rv_l, v_lv_l, v_rv_l, v_epi_l = load_ramp(args, snes, w, p_lv, p_rv, vols, capture)
    m_ref, perpt, m_fin, ts_rows, TS_HDR = geometry_records(S, base_axis, coords, p_lv_l, p_rv_l,
                                                            v_lv_l, v_rv_l, v_epi_l)
    j_max, j_mean, j_p95 = volumetric_error(msh, Jdet, comm)
    write_outputs(comm, outdir, args, name, params, quality, p_lv_l, p_rv_l, v_lv_l, v_rv_l,
                  v_epi_l, m_ref, m_fin, perpt, ts_rows, TS_HDR, j_max, j_mean, j_p95, S, fib,
                  coords, nsets, t0)


if __name__ == "__main__":
    main()
