import numpy as np
import ufl
from scipy.spatial import cKDTree
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem

from fem_mesh import TAG, build_mesh, tag_facets


def morph_coords(coords, cells, surfs, comm, f_lv, f_rv, shift_lv, shift_rv, sept=0.0):
    msh0 = build_mesh(coords, cells, comm)
    tags0 = tag_facets(msh0, coords, surfs)
    fdim = msh0.topology.dim - 1
    V = fem.functionspace(msh0, ("Lagrange", 1, (3,)))
    dc = V.tabulate_dof_coordinates()[:, :3]
    c_lv = coords[np.unique(surfs["LV_surf"])].mean(0)
    c_rv = coords[np.unique(surfs["RV_surf"])].mean(0)
    axis = c_rv - c_lv

    g = fem.Function(V)
    garr = g.x.array.reshape(-1, 3)
    lv_f = tags0.indices[tags0.values == TAG["LV"]]
    rv_f = tags0.indices[tags0.values == TAG["RV"]]
    epi_f = tags0.indices[tags0.values == TAG["EPI"]]
    d_lv = fem.locate_dofs_topological(V, fdim, lv_f)
    d_rv = fem.locate_dofs_topological(V, fdim, rv_f)
    d_epi = fem.locate_dofs_topological(V, fdim, epi_f)
    garr[d_lv] = (f_lv - 1.0) * (dc[d_lv] - c_lv) + shift_lv * axis
    garr[d_rv] = (f_rv - 1.0) * (dc[d_rv] - c_rv) - shift_rv * axis + sept * axis
    garr[d_epi] = 0.0
    bc = fem.dirichletbc(g, np.concatenate([d_lv, d_rv, d_epi]))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(fem.Constant(msh0, np.zeros(3)), v) * ufl.dx
    disp = LinearProblem(a, L, bcs=[bc],
                         petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                                        "pc_factor_mat_solver_type": "mumps"}).solve()
    darr = disp.x.array.reshape(-1, 3)
    _, nrows = cKDTree(dc).query(coords)
    coords_new = coords + darr[nrows]

    def svol(X):
        a_, b_, c_, d_ = (X[cells[:, i]] for i in range(4))
        return np.einsum("ij,ij->i", np.cross(b_ - a_, c_ - a_), d_ - a_) / 6.0
    s0, s1 = svol(coords), svol(coords_new)
    flipped = int(np.sum(np.sign(s0) != np.sign(s1)))
    return coords_new, {"flipped_elems": flipped,
                        "min_abs_vol_ratio": float(np.min(np.abs(s1)) /
                                                   max(np.min(np.abs(s0)), 1e-30))}
