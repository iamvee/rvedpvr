
EXTENDED = {"T_sept": (1.0, 3.5), "T_lvfw": (1.3, 3.3), "T_rvfw": (0.4, 1.9),
            "R_lv": (1.6, 3.8), "W_rv": (1.4, 4.6)}
WIDE = {"T_sept": (0.6, 5.0), "T_lvfw": (0.8, 5.0), "T_rvfw": (0.25, 3.2),
        "R_lv": (1.0, 5.5), "W_rv": (1.0, 5.5)}


def values(reference):
    t, r = reference["thickness_mm"], reference["radius_mm"]
    return {"T_sept": t["septum"]["mean"], "T_lvfw": t["lv_free"]["mean"],
            "T_rvfw": t["rv_free"]["mean"], "R_lv": r["lv_R"], "W_rv": r["rv_W"]}


def accept(probe, mode="extended"):
    if probe.get("morph_quality", {}).get("flipped_elems", 0) > 0:
        return False, "flipped_elems"
    v = values(probe["reference"])
    if any((x != x) or x <= 0 for x in v.values()):
        return False, "degenerate"
    if mode == "none":
        return True, "ok"
    win = WIDE if mode == "wide" else EXTENDED
    for k, (lo, hi) in win.items():
        if not (lo <= v[k] <= hi):
            return False, f"{k}={v[k]:.2f}"
    if mode == "modified":
        if v["T_rvfw"] > 1.6:
            return False, f"rvfw_not_thinned={v['T_rvfw']:.2f}"
        if v["T_sept"] < 1.5:
            return False, f"sept_not_thick={v['T_sept']:.2f}"
    return True, "ok"
