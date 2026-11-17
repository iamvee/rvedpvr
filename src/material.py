import basix.ufl
import ufl
from dolfinx import fem
from petsc4py import PETSc

from fem_mesh import log

SC = PETSc.ScalarType


def function_space(msh, args):
    incompressible = args.formulation in ("incompressible", "hybrid")
    if incompressible:
        if args.formulation == "hybrid":
            Pu = basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,))
            Pp = basix.ufl.element("DG", "tetrahedron", 0)
        else:
            Pu = basix.ufl.element("Lagrange", "tetrahedron", 2, shape=(3,))
            Pp = basix.ufl.element("Lagrange", "tetrahedron", 1)
        W = fem.functionspace(msh, basix.ufl.mixed_element([Pu, Pp]))
        w = fem.Function(W)
        u, p = ufl.split(w)
        v = ufl.TestFunction(W)
        du = ufl.split(v)[0]
        Vdisp = W.sub(0)
    else:
        W = fem.functionspace(msh, ("Lagrange", args.u_order, (3,)))
        w = fem.Function(W)
        u = w
        v = ufl.TestFunction(W)
        du = v
        Vdisp = W
        p = None
    _elab = ("P1/DG0 (C3D4H-equivalent)" if args.formulation == "hybrid"
             else "P2/P1" if incompressible else f"P{args.u_order}")
    log(f"[dofs] {args.formulation} {_elab} total="
        f"{W.dofmap.index_map.size_global * W.dofmap.index_map_bs}")
    return incompressible, W, w, u, p, v, du, Vdisp


def strain_energy(msh, args, u, p, F0, S0, N0, incompressible):
    I = ufl.Identity(3)
    Fdef = I + ufl.grad(u)
    Jdet = ufl.det(Fdef)
    Jsafe = ufl.max_value(Jdet, 1.0e-6)
    C = Fdef.T * Fdef
    if args.material == "fung":
        c_r = fem.Constant(msh, SC(args.c * 1e-3))
        B1 = fem.Constant(msh, SC(args.B1)); B2 = fem.Constant(msh, SC(args.B2))
        B3 = fem.Constant(msh, SC(args.B3))
        Cbar = Jsafe ** (-2.0 / 3.0) * C
        Eb = 0.5 * (Cbar - I)
        E11 = ufl.dot(F0, Eb * F0); E22 = ufl.dot(S0, Eb * S0); E33 = ufl.dot(N0, Eb * N0)
        E12 = ufl.dot(F0, Eb * S0); E13 = ufl.dot(F0, Eb * N0); E23 = ufl.dot(S0, Eb * N0)
        w23, w1213 = (2.0, 2.0) if args.qform == "ti" else (1.0, 1.0)
        Q = (B1 * E11 ** 2 + B2 * (E22 ** 2 + E33 ** 2 + w23 * E23 ** 2)
             + B3 * w1213 * (E12 ** 2 + E13 ** 2))
        psi = c_r * (ufl.exp(Q) - 1)
        if incompressible:
            psi += p * (Jdet - 1)
        else:
            Kval = args.K if args.K is not None else 1e4 * (args.c * 1e-3)
            Kb = fem.Constant(msh, SC(Kval))
            psi += Kb / 2 * ((Jdet ** 2 - 1) / 2 - ufl.ln(Jdet))
    else:
        I1b = Jsafe ** (-2.0 / 3.0) * ufl.tr(C)
        I4 = ufl.dot(F0, C * F0)
        a_iso = fem.Constant(msh, SC(args.a * 1e-3))
        b_iso = fem.Constant(msh, SC(args.b))
        a_f = fem.Constant(msh, SC(args.af * 1e-3))
        b_f = fem.Constant(msh, SC(args.bf))
        _cap = fem.Constant(msh, SC(args.expcap))
        _e = lambda z: ufl.exp(ufl.min_value(z, _cap))
        psi = a_iso / (2 * b_iso) * (_e(b_iso * (I1b - 3)) - 1)
        psi_f = a_f / (2 * b_f) * (_e(b_f * (I4 - 1) ** 2) - 1)
        psi += ufl.conditional(ufl.gt(I4, 1.0), psi_f, 0.0)
        if incompressible:
            psi += p * (Jdet - 1)
            Kp = (args.pstab if args.pstab > 0
                  else 1.0e3 * (3.0 * args.a + 4.0 * args.af))
            psi -= p ** 2 / (2.0 * fem.Constant(msh, SC(Kp * 1e-3)))
        else:
            psi += fem.Constant(msh, SC(args.kappa)) / 2 * (Jdet - 1) ** 2
    log(f"[material] {args.material}")
    return psi, Fdef, Jdet
