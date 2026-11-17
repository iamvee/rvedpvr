import numpy as np

from thickness import NBANDS, _regions, wall_thickness

def _tri_area(pts, tris):
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


def _radius(pts):
    c = pts.mean(0)
    return float(np.linalg.norm(pts - c, axis=1).mean())


def _fit_sphere(pts):
    if len(pts) < 4:
        return pts.mean(0), float("nan")
    A = np.hstack([2 * pts, np.ones((len(pts), 1))])
    b = np.einsum("ij,ij->i", pts, pts)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    R = float(np.sqrt(max(sol[3] + c @ c, 0.0)))
    return c, R


def crescent_curvature(rv, reg=None, epi=None, lv=None):
    R = reg if reg is not None else _regions(epi, lv, rv)
    rv_sept = R["rv_sept"]
    out = {}
    free = rv[~rv_sept]
    if len(free) >= 8:
        cf, Rf = _fit_sphere(free)
        out["rv_free_R"] = Rf
        out["rv_free_kappa"] = float(1.0 / Rf) if Rf and np.isfinite(Rf) else float("nan")
    sept = rv[rv_sept]
    if len(sept) >= 8:
        cs, Rs = _fit_sphere(sept)
        kappa = 1.0 / Rs if Rs and np.isfinite(Rs) else float("nan")
        rv_c, lv_c = rv.mean(0), (lv.mean(0) if lv is not None else rv.mean(0))
        to_rv = np.linalg.norm(cs - rv_c)
        to_lv = np.linalg.norm(cs - lv_c)
        sign = 1.0 if to_lv < to_rv else -1.0
        out["septum_R"] = Rs
        out["septum_kappa"] = float(sign * kappa) if np.isfinite(kappa) else float("nan")
    return out


def cavity_extents(pts, long_axis=None):
    c = pts.mean(0)
    q = pts - c
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    proj = q @ vt.T
    ext = np.sort(proj.max(0) - proj.min(0))[::-1]
    out = {"ext_major": float(ext[0]), "ext_mid": float(ext[1]), "ext_minor": float(ext[2])}
    if long_axis is not None:
        ax = np.asarray(long_axis, float); ax = ax / (np.linalg.norm(ax) + 1e-30)
        s = q @ ax
        out["depth_long"] = float(s.max() - s.min())
    return out


def geometry_measures(epi, epi_t, lv, lv_t, rv, rv_t, axis=None, origin=None):
    reg = _regions(epi, lv, rv)
    stats, perpt = wall_thickness(epi, lv, rv, axis, origin, reg=reg)
    areas = {"epi": _tri_area(epi, epi_t),
             "lv_endo": _tri_area(lv, lv_t),
             "rv_endo": _tri_area(rv, rv_t)}
    radius = {"lv_R": _radius(lv), "rv_W": _radius(rv)}
    curv = crescent_curvature(rv, reg=reg, epi=epi, lv=lv)
    depth = {"lv": cavity_extents(lv, axis), "rv": cavity_extents(rv, axis)}
    return {"thickness_mm": stats, "area_mm2": areas, "radius_mm": radius,
            "curvature_mm-1": curv, "extent_mm": depth}, perpt
