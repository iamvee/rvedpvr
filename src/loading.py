import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from dolfinx import fem
from dolfinx.fem import assemble_scalar, form

from fem_mesh import TAG, log

SC = PETSc.ScalarType
MMHG_TO_MPA = 133.32237e-6


def volume_forms(msh, u, coords, surfs, Jdet, Fdef, N, ds, comm):
    x = ufl.SpatialCoordinate(msh)
    pos = x + u
    lv_c = coords[np.unique(surfs["LV_surf"])].mean(0)
    rv_c = coords[np.unique(surfs["RV_surf"])].mean(0)
    epi_c = coords[np.unique(surfs["epi_surf"])].mean(0)

    def vform(tag, c):
        cc = fem.Constant(msh, SC(c))
        return form((1.0 / 3.0) * ufl.inner(pos - cc, Jdet * ufl.inv(Fdef).T * N) * ds(tag))
    Vlv_f, Vrv_f, Vepi_f = vform(TAG["LV"], lv_c), vform(TAG["RV"], rv_c), vform(TAG["EPI"], epi_c)

    def vols():
        return (abs(comm.allreduce(assemble_scalar(Vlv_f), MPI.SUM)),
                abs(comm.allreduce(assemble_scalar(Vrv_f), MPI.SUM)),
                abs(comm.allreduce(assemble_scalar(Vepi_f), MPI.SUM)))
    return vols


def surface_tracker(msh, w, incompressible, coords, surfs):
    V1v = fem.functionspace(msh, ("Lagrange", 1, (3,)))
    uh = fem.Function(V1v)
    dtree = cKDTree(V1v.tabulate_dof_coordinates()[:, :3])
    disp_src = w.sub(0) if incompressible else w
    S = {}
    for key, sname in [("epi", "epi_surf"), ("lv_endo", "LV_surf"), ("rv_endo", "RV_surf")]:
        uniq, inv = np.unique(surfs[sname].reshape(-1), return_inverse=True)
        pts = coords[uniq]
        _, rows = dtree.query(pts)
        S[key] = {"pts": pts, "tris": inv.reshape(-1, 3), "rows": rows,
                  "U": [np.zeros_like(pts)]}

    def capture():
        uh.interpolate(disp_src)
        arr = uh.x.array.reshape(-1, 3)
        for d in S.values():
            d["U"].append(arr[d["rows"]].copy())
    return S, capture


def load_ramp(args, snes, w, p_lv, p_rv, vols, capture):
    LVe, RVe = args.lvedp * MMHG_TO_MPA, args.rvedp * MMHG_TO_MPA
    v0 = vols()
    p_lv_l, p_rv_l = [0.0], [0.0]
    v_lv_l, v_rv_l, v_epi_l = [v0[0]], [v0[1]], [v0[2]]
    log(f"[V0] LV={v0[0]:.3f}  RV={v0[1]:.3f}  epi={v0[2]:.3f} mm^3")

    def solve_at(lam):
        p_lv.value = SC(lam * LVe)
        p_rv.value = SC(lam * RVe)
        try:
            snes.solve(None, w.x.petsc_vec)
            w.x.scatter_forward()
            return snes.getConvergedReason() > 0, snes.getIterationNumber()
        except PETSc.Error:
            return False, snes.getIterationNumber()

    lam_done, w_backup = 0.0, w.x.array.copy()
    min_dlam = 1.0 / (args.steps * 64)
    for s in range(1, args.steps + 1):
        target = s / args.steps
        sub, tries, subs, total_it, ok_step = target, 0, 0, 0, True
        while lam_done < target - 1e-12:
            tries += 1
            ok, nit = solve_at(sub)
            total_it += nit
            if ok:
                lam_done, w_backup = sub, w.x.array.copy()
                subs += 1
                sub = target
            else:
                w.x.array[:] = w_backup
                w.x.scatter_forward()
                sub = lam_done + 0.5 * (sub - lam_done)
                if sub - lam_done < min_dlam or tries > 80:
                    ok_step = False
                    break
        if not ok_step:
            log(f"  step {s}: diverged (sub-step floor); stopping ramp "
                f"at P_lv={lam_done*args.lvedp:.2f} mmHg")
            break
        vl, vr, ve = vols()
        p_lv_l.append(target * args.lvedp)
        p_rv_l.append(target * args.rvedp)
        v_lv_l.append(vl)
        v_rv_l.append(vr)
        v_epi_l.append(ve)
        capture()
        extra = f" ({subs} sub-steps)" if subs > 1 else ""
        log(f"  step {s}/{args.steps}  LVP={target*args.lvedp:5.2f} RVP={target*args.rvedp:5.2f}"
            f"  LV={vl:8.3f} RV={vr:8.3f}  (it={total_it}{extra})")
    return p_lv_l, p_rv_l, v_lv_l, v_rv_l, v_epi_l
