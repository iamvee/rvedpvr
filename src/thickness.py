import numpy as np
from scipy.spatial import cKDTree

NBANDS = 5


_NN_K = 16


_RAY_K = 64


_TUBE_FACTUAL = 2.5


def _spacing(pts, tree):
    d = tree.query(pts, k=2)[0][:, 1]
    return float(np.median(d))


def _point_normals(pts, centroid, k=_NN_K):
    n = len(pts)
    k = min(k, n)
    idx = cKDTree(pts).query(pts, k=k)[1]
    nb = pts[idx]
    nb = nb - nb.mean(axis=1, keepdims=True)
    _, _, vt = np.linalg.svd(nb, full_matrices=False)
    normals = vt[:, 2, :]
    out = pts - centroid
    flip = np.einsum("ij,ij->i", normals, out) < 0
    normals[flip] *= -1
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-30)
    return normals


def _transmural(src, norm, tgt_pts, tgt_tree, tube_r, k=_RAY_K, tmin=0.1):
    k = min(k, len(tgt_pts))
    idx = tgt_tree.query(src, k=k)[1]
    if idx.ndim == 1:
        idx = idx[:, None]
    cand = tgt_pts[idx]
    rel = cand - src[:, None, :]
    t = np.einsum("nkc,nc->nk", rel, norm)
    perp = np.linalg.norm(rel - t[..., None] * norm[:, None, :], axis=2)
    ok = (t > tmin) & (perp < tube_r)
    tt = np.where(ok, t, np.inf)
    j = np.argmin(tt, axis=1)
    tmin_along = tt[np.arange(len(src)), j]
    hit = np.where(np.isfinite(tmin_along), idx[np.arange(len(src)), j], -1)
    return tmin_along, hit


def _regions(epi, lv, rv):
    te, tl, tr = cKDTree(epi), cKDTree(lv), cKDTree(rv)
    lv_n = _point_normals(lv, lv.mean(0))
    rv_n = _point_normals(rv, rv.mean(0))
    tube_lv = _TUBE_FACTUAL * _spacing(lv, tl)
    tube_rv = _TUBE_FACTUAL * _spacing(rv, tr)

    t_lv_epi, h_lv_epi = _transmural(lv, lv_n, epi, te, tube_lv)
    t_lv_rv, h_lv_rv = _transmural(lv, lv_n, rv, tr, tube_lv)
    lv_sept = t_lv_rv < t_lv_epi
    lv_thick = np.minimum(t_lv_epi, t_lv_rv)
    bad = ~np.isfinite(lv_thick)
    if bad.any():
        nn = np.minimum(te.query(lv[bad])[0], tr.query(lv[bad])[0])
        lv_thick[bad] = nn

    t_rv_epi, h_rv_epi = _transmural(rv, rv_n, epi, te, tube_rv)
    t_rv_lv, h_rv_lv = _transmural(rv, rv_n, lv, tl, tube_rv)
    rv_sept = t_rv_lv < t_rv_epi
    rv_thick = np.minimum(t_rv_epi, t_rv_lv)
    bad = ~np.isfinite(rv_thick)
    if bad.any():
        nn = np.minimum(te.query(rv[bad])[0], tl.query(rv[bad])[0])
        rv_thick[bad] = nn

    return {"lv_n": lv_n, "rv_n": rv_n,
            "lv_sept": lv_sept, "rv_sept": rv_sept,
            "lv_thick": lv_thick, "rv_thick": rv_thick,
            "t_lv_epi": t_lv_epi, "h_lv_epi": h_lv_epi,
            "t_lv_rv": t_lv_rv, "h_lv_rv": h_lv_rv,
            "t_rv_epi": t_rv_epi, "h_rv_epi": h_rv_epi,
            "t_rv_lv": t_rv_lv, "h_rv_lv": h_rv_lv}


def _bands(pts, thick, axis, origin, nbands=NBANDS):
    if axis is None or origin is None or pts.size == 0:
        return []
    ax = np.asarray(axis, float)
    ax = ax / (np.linalg.norm(ax) + 1e-30)
    s = (pts - np.asarray(origin, float)) @ ax
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-9:
        return [float(np.mean(thick))] + [np.nan] * (nbands - 1)
    edges = np.linspace(lo, hi, nbands + 1)
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, nbands - 1)
    return [float(thick[idx == b].mean()) if (idx == b).any() else float("nan")
            for b in range(nbands)]


def _stats(a, pts=None, axis=None, origin=None):
    a = np.asarray(a, float)
    finite = np.isfinite(a)
    a = a[finite]
    if pts is not None:
        pts = np.asarray(pts)[finite]
    if a.size == 0:
        return {"n": 0}
    d = {"n": int(a.size), "mean": float(a.mean()), "std": float(a.std()),
         "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
         "p90": float(np.percentile(a, 90)),
         "min": float(a.min()), "max": float(a.max())}
    if pts is not None:
        d["bands"] = _bands(pts, a, axis, origin)
    return d


def wall_thickness(epi, lv, rv, axis=None, origin=None, reg=None):
    R = reg if reg is not None else _regions(epi, lv, rv)
    lv_sept, rv_sept = R["lv_sept"], R["rv_sept"]
    sept_m = lv_sept & (R["h_lv_rv"] >= 0)
    lvfw_m = (~lv_sept) & (R["h_lv_epi"] >= 0)
    rvfw_m = (~rv_sept) & (R["h_rv_epi"] >= 0)
    stats = {"septum": _stats(R["t_lv_rv"][sept_m], lv[sept_m], axis, origin),
             "lv_free": _stats(R["t_lv_epi"][lvfw_m], lv[lvfw_m], axis, origin),
             "rv_free": _stats(R["t_rv_epi"][rvfw_m], rv[rvfw_m], axis, origin)}
    perpt = {"lv_endo_thick": R["lv_thick"], "lv_endo_sept": lv_sept.astype(np.int8),
             "rv_endo_thick": R["rv_thick"], "rv_endo_sept": rv_sept.astype(np.int8)}
    return stats, perpt


def measurement_lines(epi, lv, rv, reg=None):
    R = reg if reg is not None else _regions(epi, lv, rv)
    lv_sept, rv_sept = R["lv_sept"], R["rv_sept"]

    def seg(src, mask, hit, tpts):
        m = mask & (hit >= 0)
        s = src[m]; e = tpts[hit[m]]
        return s, e, np.linalg.norm(e - s, axis=1)

    return {
        "septum":  seg(lv, lv_sept,  R["h_lv_rv"],  rv),
        "lv_free": seg(lv, ~lv_sept, R["h_lv_epi"], epi),
        "rv_free": seg(rv, ~rv_sept, R["h_rv_epi"], epi),
    }
