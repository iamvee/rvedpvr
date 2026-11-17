import json
import os
import time

import numpy as np

import measure as meas
from fem_mesh import log

PHYS_KEYS = ["geom", "lvedp", "rvedp", "a", "b", "af", "bf", "endo", "epi",
             "f_lv", "f_rv", "shift_lv", "shift_rv", "sept", "kappa", "formulation", "steps"]


def write_params(outdir, params, quality):
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump({**params, "morph_quality": quality}, f, indent=2)


def append_index(name, params, quality):
    idx = os.path.join("results", "INDEX.md")
    cols = ["name"] + PHYS_KEYS + ["flipped_elems"]
    if not os.path.exists(idx):
        with open(idx, "w") as f:
            f.write("# Results index\n\nEach row is one simulation; see the "
                    "matching `results/<name>/params.json`.\n\n")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
    with open(idx) as f:
        if name in f.read():
            return
    row = [name] + [str(params[k]) for k in PHYS_KEYS] + [str(quality.get("flipped_elems", 0))]
    with open(idx, "a") as f:
        f.write("| " + " | ".join(row) + " |\n")


def probe_geometry(coords, surfs, nsets, quality, params, outdir, comm):
    def _cloud(sn):
        u, inv = np.unique(surfs[sn].reshape(-1), return_inverse=True)
        return coords[u], inv.reshape(-1, 3)
    ep, eT = _cloud("epi_surf"); lp, lT = _cloud("LV_surf"); rp, rT = _cloud("RV_surf")
    axis = coords[nsets["top_nodes"]].mean(0) - coords.mean(0)
    m_ref, _ = meas.geometry_measures(ep, eT, lp, lT, rp, rT, axis, coords.mean(0))
    if comm.rank == 0:
        json.dump({"reference": m_ref, "morph_quality": quality, "params": params},
                  open(os.path.join(outdir, "probe.json"), "w"), indent=2)
        write_params(outdir, params, quality)
    tk = m_ref["thickness_mm"]
    log(f"[probe] RVfw={tk['rv_free']['mean']:.3f} sept={tk['septum']['mean']:.3f} "
        f"LVfw={tk['lv_free']['mean']:.3f}  R_lv={m_ref['radius_mm']['lv_R']:.3f} "
        f"W_rv={m_ref['radius_mm']['rv_W']:.3f} mm  flipped={quality['flipped_elems']}")


def geometry_records(S, base_axis, coords, p_lv_l, p_rv_l, v_lv_l, v_rv_l, v_epi_l):
    ep, lp, rp = S["epi"]["pts"], S["lv_endo"]["pts"], S["rv_endo"]["pts"]
    eT, lT, rT = S["epi"]["tris"], S["lv_endo"]["tris"], S["rv_endo"]["tris"]
    geo_axis, geo_org = base_axis, coords.mean(0)
    m_ref, perpt = meas.geometry_measures(ep, eT, lp, lT, rp, rT, geo_axis, geo_org)
    m_ref["volume_mm3"] = {"lv_cav": v_lv_l[0], "rv_cav": v_rv_l[0], "epi": v_epi_l[0],
                           "myo": v_epi_l[0] - v_lv_l[0] - v_rv_l[0]}
    epf, lpf, rpf = ep + S["epi"]["U"][-1], lp + S["lv_endo"]["U"][-1], rp + S["rv_endo"]["U"][-1]
    m_fin, _ = meas.geometry_measures(epf, eT, lpf, lT, rpf, rT, geo_axis, geo_org)
    m_fin["volume_mm3"] = {"lv_cav": v_lv_l[-1], "rv_cav": v_rv_l[-1], "epi": v_epi_l[-1],
                           "myo": v_epi_l[-1] - v_lv_l[-1] - v_rv_l[-1]}
    th = m_ref["thickness_mm"]
    log(f"[thick] septum={th['septum']['mean']:.3f}+/-{th['septum']['std']:.3f}  "
        f"LVfree={th['lv_free']['mean']:.3f}+/-{th['lv_free']['std']:.3f}  "
        f"RVfree={th['rv_free']['mean']:.3f}+/-{th['rv_free']['std']:.3f} mm")

    NB = meas.NBANDS
    def _bandrow(reg):
        b = reg.get("bands", [])
        return (list(b) + [float("nan")] * NB)[:NB]
    band_hdr = ",".join(f"th_{w}_b{j}" for w in ("sept", "lvfw", "rvfw") for j in range(NB))
    TS_HDR = ("step,P_lv_mmHg,P_rv_mmHg,V_lv_mm3,V_rv_mm3,V_epi_mm3,V_myo_mm3,"
              "area_epi,area_lv_endo,area_rv_endo,th_sept_mean,th_sept_std,"
              "th_lvfw_mean,th_lvfw_std,th_rvfw_mean,th_rvfw_std,R_lv,W_rv,"
              "rvfree_kappa,septum_kappa,lv_depth,rv_depth," + band_hdr)
    ts_rows = []
    for k in range(len(p_lv_l)):
        md, _ = meas.geometry_measures(ep + S["epi"]["U"][k], eT,
                                       lp + S["lv_endo"]["U"][k], lT,
                                       rp + S["rv_endo"]["U"][k], rT, geo_axis, geo_org)
        t_, a_, r_ = md["thickness_mm"], md["area_mm2"], md["radius_mm"]
        c_, ex_ = md["curvature_mm-1"], md["extent_mm"]
        vmyo = v_epi_l[k] - v_lv_l[k] - v_rv_l[k]
        row = [k, p_lv_l[k], p_rv_l[k], v_lv_l[k], v_rv_l[k], v_epi_l[k], vmyo,
               a_["epi"], a_["lv_endo"], a_["rv_endo"],
               t_["septum"]["mean"], t_["septum"]["std"],
               t_["lv_free"]["mean"], t_["lv_free"]["std"],
               t_["rv_free"]["mean"], t_["rv_free"]["std"],
               r_["lv_R"], r_["rv_W"],
               c_.get("rv_free_kappa", float("nan")), c_.get("septum_kappa", float("nan")),
               ex_["lv"]["depth_long"], ex_["rv"]["depth_long"]]
        row += _bandrow(t_["septum"]) + _bandrow(t_["lv_free"]) + _bandrow(t_["rv_free"])
        ts_rows.append(row)
    return m_ref, perpt, m_fin, ts_rows, TS_HDR


def write_outputs(comm, outdir, args, name, params, quality, p_lv_l, p_rv_l, v_lv_l, v_rv_l,
                  v_epi_l, m_ref, m_fin, perpt, ts_rows, TS_HDR, j_max, j_mean, j_p95, S, fib,
                  coords, nsets, t0):
    if comm.rank == 0:
        np.savetxt(os.path.join(outdir, "pv.csv"),
                   np.column_stack([p_lv_l, p_rv_l, v_lv_l, v_rv_l, v_epi_l]),
                   delimiter=",", comments="",
                   header="P_lv_mmHg,P_rv_mmHg,V_lv_mm3,V_rv_mm3,V_epi_mm3")
        converged = len(p_lv_l) - 1
        v_myo = [ve - vl - vr for ve, vl, vr in zip(v_epi_l, v_lv_l, v_rv_l)]
        myo0 = v_myo[0] if v_myo else 0.0
        myo_drift = (max(abs(m - myo0) for m in v_myo) / myo0) if myo0 else None
        with open(os.path.join(outdir, "measurements.json"), "w") as f:
            json.dump({"converged_steps": converged, "requested_steps": args.steps,
                       "complete": converged == args.steps,
                       "myo_incompressibility": {
                           "vol_ref_mm3": myo0,
                           "vol_final_mm3": (v_myo[-1] if v_myo else None),
                           "max_rel_drift": myo_drift,
                           "max_abs_J_minus_1": j_max,
                           "p95_abs_J_minus_1": j_p95,
                           "mean_abs_J_minus_1": j_mean},
                       "reference": m_ref, "final": m_fin}, f, indent=2)
        np.savetxt(os.path.join(outdir, "measurements_ts.csv"), np.array(ts_rows),
                   delimiter=",", header=TS_HDR, comments="",
                   fmt=["%d"] + ["%.6g"] * (len(TS_HDR.split(",")) - 1))
        write_params(outdir, params, quality)
        if not args.no_index:
            append_index(name, params, quality)
        if not args.no_frames:
            out = {"p_lv": np.asarray(p_lv_l), "p_rv": np.asarray(p_rv_l),
                   "v_lv": np.asarray(v_lv_l), "v_rv": np.asarray(v_rv_l),
                   "v_epi": np.asarray(v_epi_l), **perpt,
                   "bc_top_uy": coords[nsets["top_nodes"]],
                   "bc_xmm_uz": coords[nsets.get("topNode_X_max_min_nodes", np.array([], int))],
                   "bc_zmm_ux": coords[nsets.get("topNode_Z_max_min_nodes", np.array([], int))]}
            for key, d in S.items():
                out[f"{key}_pts"] = d["pts"]
                out[f"{key}_tris"] = d["tris"]
                out[f"{key}_U"] = np.asarray(d["U"])
            np.savez_compressed(os.path.join(outdir, "frames.npz"), **out)
            np.savez_compressed(os.path.join(outdir, "fibers.npz"), **fib,
                                epi_pts=S["epi"]["pts"], epi_tris=S["epi"]["tris"])
        log(f"[done] {len(p_lv_l)-1} steps in {time.time()-t0:.1f}s "
            f"({'numeric' if args.no_frames else 'full'}) -> {outdir}")
