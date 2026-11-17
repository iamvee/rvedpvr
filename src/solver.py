import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from dolfinx import fem
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)

from fem_mesh import TAG, ipoints, log, vertex_coords

SC = PETSc.ScalarType


class SNESProblem:
    def __init__(self, R, Jac, w, bcs):
        self.L = fem.form(R)
        self.a = fem.form(Jac)
        self.bcs = bcs
        self.w = w

    def F(self, snes, x, b):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.w.x.petsc_vec)
        self.w.x.scatter_forward()
        with b.localForm() as bloc:
            bloc.set(0.0)
        assemble_vector(b, self.L)
        apply_lifting(b, [self.a], bcs=[self.bcs], x0=[x], alpha=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x, A, P):
        A.zeroEntries()
        assemble_matrix(A, self.a, bcs=self.bcs)
        A.assemble()


def bc_dofs_at(Vdisp, comp, pts, vtree, allv):
    _, rows = vtree.query(pts)
    verts = allv[rows].astype(np.int32)
    Vc = Vdisp.sub(comp)
    dofs = fem.locate_dofs_topological(Vc, 0, verts)
    log(f"[bc] component {comp}: {len(verts)} vertices -> {len(dofs)} constrained dofs")
    if len(dofs) == 0:
        raise RuntimeError(
            f"displacement BC on component {comp} constrained 0 dofs from {len(verts)} vertices; "
            f"the model would be free to translate/rotate and the Jacobian singular")
    return fem.dirichletbc(SC(0.0), dofs, Vc)


def residual_forms(msh, tags, psi, Jdet, Fdef, w, v, du):
    Pi = psi * ufl.dx

    N = ufl.FacetNormal(msh)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    p_lv = fem.Constant(msh, SC(0.0))
    p_rv = fem.Constant(msh, SC(0.0))
    follow = (p_lv * ufl.inner(Jdet * ufl.inv(Fdef).T * N, du) * ds(TAG["LV"])
              + p_rv * ufl.inner(Jdet * ufl.inv(Fdef).T * N, du) * ds(TAG["RV"]))
    R = ufl.derivative(Pi, w, v) + follow
    Jac = ufl.derivative(R, w)
    return R, Jac, p_lv, p_rv, N, ds


def boundary_conditions(msh, Vdisp, coords, nsets):
    vco, allv = vertex_coords(msh)
    vtree = cKDTree(vco)
    bcs = [bc_dofs_at(Vdisp, 1, coords[nsets["top_nodes"]], vtree, allv)]
    if "topNode_X_max_min_nodes" in nsets:
        bcs.append(bc_dofs_at(Vdisp, 2, coords[nsets["topNode_X_max_min_nodes"]], vtree, allv))
    if "topNode_Z_max_min_nodes" in nsets:
        bcs.append(bc_dofs_at(Vdisp, 0, coords[nsets["topNode_Z_max_min_nodes"]], vtree, allv))
    return bcs


def newton_solver(msh, R, Jac, w, bcs, args):
    problem = SNESProblem(R, Jac, w, bcs)
    snes = PETSc.SNES().create(msh.comm)
    snes.setFunction(problem.F, create_vector(problem.L))
    snes.setJacobian(problem.J, create_matrix(problem.a))
    snes.setType("newtonls")
    snes.setTolerances(rtol=1e-8, atol=1e-9, max_it=40)
    ksp = snes.getKSP()
    pc = ksp.getPC()
    if args.ksp == "amg":
        ksp.setType("gmres")
        ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=500)
        pc.setType("gamg")
    else:
        ksp.setType("preonly")
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
    opts = PETSc.Options()
    if args.ksp == "amg":
        opts["pc_gamg_type"] = "agg"
        opts["pc_gamg_agg_nsmooths"] = 1
        opts["mg_levels_ksp_type"] = "chebyshev"
        opts["mg_levels_pc_type"] = "jacobi"
    opts["snes_linesearch_type"] = args.linesearch
    opts["mat_mumps_icntl_14"] = 80
    snes.setFromOptions()
    return snes


def volumetric_error(msh, Jdet, comm):
    try:
        V_dg0 = fem.functionspace(msh, ("DG", 0))
        jf = fem.Function(V_dg0)
        jf.interpolate(fem.Expression(Jdet, ipoints(V_dg0)))
        jl = jf.x.array
        jdev = np.abs(jl - 1.0)
        j_max = float(comm.allreduce(float(jdev.max()) if jdev.size else 0.0, op=MPI.MAX))
        j_mean = float(comm.allreduce(float(jdev.mean()) if jdev.size else 0.0, op=MPI.SUM)
                       / max(comm.size, 1))
        j_p95 = float(comm.allreduce(float(np.percentile(jdev, 95)) if jdev.size else 0.0, op=MPI.MAX))
    except Exception as e:
        log(f"[integrity] could not evaluate J: {e}")
        j_max = j_mean = j_p95 = None
    return j_max, j_mean, j_p95
