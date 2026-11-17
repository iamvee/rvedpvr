import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "sfin"
PHYS = ("geom", "lvedp", "rvedp", "material", "a", "b", "af", "bf", "c", "B1", "B2", "B3", "qform",
        "endo", "epi", "f_lv", "f_rv", "shift_lv", "shift_rv", "sept", "kappa", "formulation", "steps")


HYPER = {"281", "301", "321", "341", "361", "381"}


def load_meta_by_id(queue_path):
    if not os.path.exists(queue_path):
        return {}
    doc = json.load(open(queue_path))
    return {r["run_id"]: r for r in doc.get("runs", [])}


def derive_meta(p):
    kap = p.get("kappa")
    if kap is None:
        comp = None
    elif abs(float(kap) - 0.01) < 1e-6:
        comp = "compressible"
    else:
        comp = "near_incompressible"
    lv = p.get("lvedp") or 0
    ratio = round(p.get("rvedp", 0) / lv, 4) if lv else None
    cohort = "hypertrophic" if str(p.get("geom")) in HYPER else "normal"
    return {"_comp": comp, "_ratio": ratio, "_cohort": cohort}


def aggregate(results_dir, meta_by_id):
    rows = []
    for name in sorted(os.listdir(results_dir)):
        if name not in meta_by_id and not name.startswith(TAG):
            continue
        d = os.path.join(results_dir, name)
        pp, mp = os.path.join(d, "params.json"), os.path.join(d, "measurements.json")
        if not (os.path.exists(pp) and os.path.exists(mp)):
            continue
        try:
            p, m = json.load(open(pp)), json.load(open(mp))
        except Exception:
            continue
        meta = meta_by_id.get(name, {})
        fb = derive_meta(p)
        row = {"name": name}
        row.update({k: p.get(k) for k in PHYS})
        for k in ("_arm", "_comp", "_pair", "_ratio", "_cohort", "config_id"):
            row[k] = meta.get(k, fb.get(k))
        lv = p.get("lvedp") or 0
        row["ratio"] = round(p.get("rvedp", 0) / lv, 4) if lv else None
        row["flipped_elems"] = p.get("morph_quality", {}).get("flipped_elems", 0)
        row["converged_steps"] = m.get("converged_steps")
        row["requested_steps"] = m.get("requested_steps")
        row["complete"] = m.get("complete")
        for cfg in ("reference", "final"):
            blk = m.get(cfg, {})
            for reg in ("septum", "lv_free", "rv_free"):
                st = blk.get("thickness_mm", {}).get(reg, {})
                row[f"{cfg}_th_{reg}_mean"] = st.get("mean")
                row[f"{cfg}_th_{reg}_std"] = st.get("std")
            for su in ("epi", "lv_endo", "rv_endo"):
                row[f"{cfg}_area_{su}"] = blk.get("area_mm2", {}).get(su)
            for cv in ("lv_cav", "rv_cav", "epi", "myo"):
                row[f"{cfg}_vol_{cv}"] = blk.get("volume_mm3", {}).get(cv)
            rad = blk.get("radius_mm", {})
            row[f"{cfg}_R_lv"] = rad.get("lv_R")
            row[f"{cfg}_W_rv"] = rad.get("rv_W")
            cur = blk.get("curvature_mm-1", {})
            row[f"{cfg}_rvfree_kappa"] = cur.get("rv_free_kappa")
            row[f"{cfg}_septum_kappa"] = cur.get("septum_kappa")
            ext = blk.get("extent_mm", {})
            row[f"{cfg}_lv_depth"] = ext.get("lv", {}).get("depth_long")
            row[f"{cfg}_rv_depth"] = ext.get("rv", {}).get("depth_long")
        _mi = m.get("myo_incompressibility", {})
        row["myo_max_rel_drift"] = _mi.get("max_rel_drift")
        row["max_abs_J_minus_1"] = _mi.get("max_abs_J_minus_1")
        row["p95_abs_J_minus_1"] = _mi.get("p95_abs_J_minus_1")
        rows.append(row)
    return rows


def main():
    results_dir = os.path.join(HERE, "results")
    meta_by_id = load_meta_by_id(os.path.join(HERE, "designs", "runqueue.json"))
    rows = aggregate(results_dir, meta_by_id)
    if not rows:
        print("[aggregate] no finished runs yet."); return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out = os.path.join(results_dir, "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    done = sum(1 for r in rows if r.get("complete"))
    print(f"[aggregate] {len(rows)} runs ({done} complete) -> {out}")


if __name__ == "__main__":
    main()
