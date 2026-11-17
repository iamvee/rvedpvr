import argparse
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_GEOMS = [str(1 + 20 * k) for k in range(20)]
HYPERTROPHIC = {"281", "301", "321", "341", "361", "381"}

try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from config import CFG
    _P, _M = CFG.protocol, CFG.material
    _SW = _M.sweep
except Exception as _e:
    print(f"[plan] WARNING: could not read STUDY_CONFIG.yaml ({_e}); using built-in defaults.")
    CFG = None

LV = float(_P.lv_peak_mmhg) if CFG else 30.0
RATIOS4 = ([round(float(r) / LV, 4) for r in _P.rv_peaks_mmhg]
           if CFG else [0.33, 0.50, 0.67, 1.00])
REF_RATIO = (round(float(_P.rv_peak_mmhg) / LV, 4) if CFG else 0.33)
if REF_RATIO not in RATIOS4:
    RATIOS4.append(REF_RATIO)
RATIOS4 = sorted(set(RATIOS4))

FIXED = {"a": float(_M.a) if CFG else 0.10,
         "b": float(_M.b) if CFG else 5.00,
         "kappa": float(_M.kappa_mpa) if CFG else 0.01,
         "lvedp": LV}

A_MAT = ({"af": tuple(_SW.af_kpa), "bf": tuple(_SW.bf),
          "endo": tuple(_SW.endo_deg), "epi": tuple(_SW.epi_deg)} if CFG else
         {"af": (5.0, 40.0), "bf": (6.0, 14.0), "endo": (35.0, 60.0), "epi": (-60.0, -35.0)})
A_SHAPE = {"f_lv": (0.92, 1.10), "f_rv": (0.90, 1.10),
           "shift_lv": (0.0, 0.03), "shift_rv": (0.0, 0.03), "sept": (0.0, 0.04)}

B_MAT = ({"af": float(_M.af_reference), "bf": float(_M.bf_reference),
          "endo": float(_M.endo_deg), "epi": float(_M.epi_deg)} if CFG else
         {"af": 20.0, "bf": 10.0, "endo": 45.0, "epi": -45.0})
B_RATIO_LO, B_RATIO_HI, B_RATIO_N = 0.01, 5.0, 25

def _widen(lo_hi, factor=1.5):
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * factor
    return (mid - half, mid + half)

C_MAT = ({"af": _widen(A_MAT["af"]), "bf": _widen(A_MAT["bf"]),
          "endo": _widen(A_MAT["endo"]), "epi": _widen(A_MAT["epi"])} if CFG else
         {"af": (1.0, 100.0), "bf": (3.0, 20.0), "endo": (20.0, 80.0), "epi": (-80.0, -20.0)})
C_MAT["af"] = (max(0.5, C_MAT["af"][0]), C_MAT["af"][1])
C_MAT["bf"] = (max(0.5, C_MAT["bf"][0]), C_MAT["bf"][1])

C_SHAPE = ({"f_lv": tuple(CFG.cohort.geometry.f_lv), "f_rv": tuple(CFG.cohort.geometry.f_rv),
            "shift_lv": (0.0, max(CFG.cohort.geometry.shift)),
            "shift_rv": (0.0, max(CFG.cohort.geometry.shift)),
            "sept": (-0.03, 0.10)} if CFG else
           {"f_lv": (0.78, 1.28), "f_rv": (0.78, 1.28),
            "shift_lv": (0.0, 0.08), "shift_rv": (0.0, 0.08), "sept": (-0.03, 0.10)})
C_RATIO = ((min(RATIOS4), max(RATIOS4)) if CFG else (0.10, 1.50))

TARGET = {"A_sets": 1000, "B_geoms": len(BASE_GEOMS), "B_ratios": B_RATIO_N, "C": 500}
OVERSAMPLE = {"A_sets": 2000, "C": 1000}

PREC = {"af": 2, "bf": 2, "endo": 1, "epi": 1, "f_lv": 3, "f_rv": 3,
        "shift_lv": 3, "shift_rv": 3, "sept": 3, "rvedp": 2}
GUARD = {"f_lv": (0.68, 1.37), "f_rv": (0.68, 1.37), "shift_lv": (0.0, 0.12),
         "shift_rv": (0.0, 0.12), "sept": (-0.06, 0.16)}


def cohort_of(geom):
    return "hypertrophic" if str(geom) in HYPERTROPHIC else "normal"


def lhs(n, k, rng):
    grid = [[(i + rng.random()) / n for i in range(n)] for _ in range(k)]
    for col in grid:
        rng.shuffle(col)
    return [[grid[j][i] for j in range(k)] for i in range(n)]


def draw(bands, row):
    keys = list(bands.keys())
    return {k: round(bands[k][0] + (bands[k][1] - bands[k][0]) * row[j], PREC.get(k, 3))
            for j, k in enumerate(keys)}


def in_guard(s):
    return all(GUARD[k][0] <= s.get(k, 0.0) <= GUARD[k][1] for k in GUARD)


def cfg(config_id, arm, geom, params, rvedp, filt, steps):
    s = dict(FIXED)
    s.update(params)
    s.update({"config_id": config_id, "_arm": arm, "geom": str(geom),
              "_cohort": cohort_of(geom), "rvedp": round(rvedp, 2),
              "_filter": filt, "steps": steps})
    return s


def make_candidates(steps, seed, over=1.0, scale=1.0):
    rng = random.Random(2000 + seed)
    TARGET["A_sets"] = int(round(TARGET["A_sets"] * scale))
    TARGET["C"] = int(round(TARGET["C"] * scale))
    out = []
    cid = 0
    nA = max(TARGET["A_sets"], int(round(OVERSAMPLE["A_sets"] * over * scale)))
    nC = max(TARGET["C"], int(round(OVERSAMPLE["C"] * over * scale)))

    a_shape = lhs(nA, len(A_SHAPE), rng)
    a_mat = lhs(nA, len(A_MAT), rng)
    a_geom_rows = lhs(nA, 1, rng)
    for i in range(nA):
        params = {**draw(A_SHAPE, a_shape[i]), **draw(A_MAT, a_mat[i])}
        if not in_guard(params):
            continue
        geom = BASE_GEOMS[int(a_geom_rows[i][0] * len(BASE_GEOMS)) % len(BASE_GEOMS)]
        setid = f"A{i:04d}"
        for r in RATIOS4:
            out.append(cfg(f"{cid:05d}", "A_normal", geom, params, r * LV, "extended", steps))
            out[-1]["_setid"] = setid
            out[-1]["_ratio"] = r
            cid += 1

    ratiosB = [round(10 ** x, 4) for x in
               [math.log10(B_RATIO_LO) + (math.log10(B_RATIO_HI) - math.log10(B_RATIO_LO)) * j
                / (B_RATIO_N - 1) for j in range(B_RATIO_N)]]
    for geom in BASE_GEOMS:
        for r in ratiosB:
            out.append(cfg(f"{cid:05d}", "B_sweep", geom, dict(B_MAT), r * LV, "none", steps))
            out[-1]["_ratio"] = r
            cid += 1

    c_rows = lhs(nC, len(C_SHAPE) + len(C_MAT) + 1, rng)
    ns, nm = len(C_SHAPE), len(C_MAT)
    for i, row in enumerate(c_rows):
        params = {**draw(C_SHAPE, row[:ns]), **draw(C_MAT, row[ns:ns + nm])}
        if not in_guard(params):
            continue
        geom = BASE_GEOMS[i % len(BASE_GEOMS)]
        ratio = round(C_RATIO[0] + (C_RATIO[1] - C_RATIO[0]) * row[-1], 3)
        out.append(cfg(f"{cid:05d}", "C_extended", geom, params, ratio * LV, "wide", steps))
        out[-1]["_ratio"] = ratio
        cid += 1
    return out


def summarize(cands):
    from collections import Counter
    arms = Counter(c["_arm"] for c in cands)
    return dict(arms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the arm-A/arm-C config targets: --scale 2 gives ~2x the unique "
                         "hearts.")
    ap.add_argument("--oversample", type=float, default=1.0,
                    help="scale the candidate over-draw (>1 aims for more headroom so the screened "
                         "queue still hits target when the realism reject rate is high)")
    ap.add_argument("--out", default=os.path.join(HERE, "designs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cands = make_candidates(args.steps, args.seed, args.oversample, args.scale)
    arms = summarize(cands)
    tot_cfg = TARGET["A_sets"] * len(RATIOS4) + TARGET["B_geoms"] * TARGET["B_ratios"] + TARGET["C"]
    meta = {"n_candidates": len(cands), "steps": args.steps, "seed": args.seed,
            "targets": dict(TARGET), "oversample_factor": args.oversample, "scale": args.scale, "ratios4": RATIOS4, "lvedp": LV,
            "arms_candidate": arms,
            "note": "candidates only; run screen.py to filter, trim and shuffle"}
    outp = os.path.join(args.out, "candidates.json")
    with open(outp, "w") as f:
        json.dump({"meta": meta, "candidates": cands}, f, indent=1)
    print(f"candidates: {len(cands)} configs -> {outp}")
    print(f"   arms (candidate): {arms}")
    print(f"   targets (post-screen): A={TARGET['A_sets']}x{len(RATIOS4)}="
          f"{TARGET['A_sets']*len(RATIOS4)}, B={TARGET['B_geoms']}x{TARGET['B_ratios']}="
          f"{TARGET['B_geoms']*TARGET['B_ratios']}, C={TARGET['C']}  => {tot_cfg} configs")
    print(f"   x1 solve per config (compressibility is a numerical setting, not a factor)"
          f"  =>  {tot_cfg} solves")


if __name__ == "__main__":
    main()
