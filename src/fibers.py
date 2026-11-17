import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc

from fem_mesh import TAG, ipoints

SC = PETSc.ScalarType


def compute_fibers(msh, tags, coords, cells, base_axis, endo_deg, epi_deg):
    fdim = msh.topology.dim - 1
    tdim = msh.topology.dim
    V1 = fem.functionspace(msh, ("Lagrange", 1))
    tr, tt = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    a = ufl.inner(ufl.grad(tr), ufl.grad(tt)) * ufl.dx
    L = ufl.inner(fem.Constant(msh, SC(0.0)), tt) * ufl.dx
    endo_f = tags.indices[np.isin(tags.values, [TAG["RV"], TAG["LV"]])]
    epi_f = tags.indices[tags.values == TAG["EPI"]]
    bc0 = fem.dirichletbc(SC(0.0), fem.locate_dofs_topological(V1, fdim, endo_f), V1)
    bc1 = fem.dirichletbc(SC(1.0), fem.locate_dofs_topological(V1, fdim, epi_f), V1)
    prob = LinearProblem(a, L, bcs=[bc0, bc1],
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                        "pc_factor_mat_solver_type": "mumps"})
    t = prob.solve()

    Vv = fem.functionspace(msh, ("DG", 0, (3,)))
    Vs = fem.functionspace(msh, ("DG", 0))
    gt = fem.Function(Vv)
    gt.interpolate(fem.Expression(ufl.grad(t), ipoints(Vv)))
    tc = fem.Function(Vs)
    tc.interpolate(fem.Expression(t, ipoints(Vs)))

    g = gt.x.array.reshape(-1, 3)
    tv = np.clip(tc.x.array, 0.0, 1.0)
    r = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-30)
    k = base_axis / (np.linalg.norm(base_axis) + 1e-30)
    el = k[None, :] - (r * k[None, :]).sum(1, keepdims=True) * r
    el /= (np.linalg.norm(el, axis=1, keepdims=True) + 1e-30)
    ec = np.cross(el, r)
    ec /= (np.linalg.norm(ec, axis=1, keepdims=True) + 1e-30)
    helix = endo_deg + (epi_deg - endo_deg) * tv
    alpha = np.deg2rad(helix)
    f0 = np.cos(alpha)[:, None] * ec + np.sin(alpha)[:, None] * el
    f0 /= (np.linalg.norm(f0, axis=1, keepdims=True) + 1e-30)
    F0 = fem.Function(Vv)
    F0.x.array[:] = f0.reshape(-1)

    s0 = np.cross(r, f0); s0 /= (np.linalg.norm(s0, axis=1, keepdims=True) + 1e-30)
    n0 = np.cross(f0, s0); n0 /= (np.linalg.norm(n0, axis=1, keepdims=True) + 1e-30)
    S0 = fem.Function(Vv); S0.x.array[:] = s0.reshape(-1)
    N0 = fem.Function(Vv); N0.x.array[:] = n0.reshape(-1)

    ncells = msh.topology.index_map(tdim).size_local
    cids = np.arange(ncells, dtype=np.int32)
    cent = dmesh.compute_midpoints(msh, tdim, cids)
    fib = {"centroids": cent, "f0": f0[:ncells], "s0": s0[:ncells], "n0": n0[:ncells],
           "t": tv[:ncells], "helix_deg": helix[:ncells]}
    return F0, S0, N0, fib
