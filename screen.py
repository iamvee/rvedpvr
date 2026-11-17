import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
REPO_LOCAL = HERE
REPO_HOST = os.environ.get("HOST_REPO", REPO_LOCAL)
RUNNER_SELF = os.environ.get("RUNNER_SELF")
KAPPA_COMPRESSIBLE = 0.003
KAPPA_NEAR_INCOMP = 0.10
_COND_COMPRESSIBLE = ("compressible", "compressible", KAPPA_COMPRESSIBLE)
_COND_NEAR_INCOMP = ("near_incompressible", "compressible", KAPPA_NEAR_INCOMP)
CONDITIONS = [_COND_COMPRESSIBLE]


def set_conditions(mode):
    global CONDITIONS
    if mode == "both":
        CONDITIONS = [_COND_COMPRESSIBLE, _COND_NEAR_INCOMP]
    elif mode == "incompressible":
        CONDITIONS = [("incompressible", "incompressible", KAPPA_COMPRESSIBLE)]
    else:
        CONDITIONS = [_COND_COMPRESSIBLE]
    return CONDITIONS


def mount_flags():
    return ["--volumes-from", RUNNER_SELF] if RUNNER_SELF else ["-v", f"{REPO_HOST}:/work"]


def geom_key(s):
    return (str(s["geom"]), s.get("f_lv", 1.0), s.get("f_rv", 1.0),
            s.get("shift_lv", 0.0), s.get("shift_rv", 0.0), s.get("sept", 0.0))


def probe_one(s, args, name):
    cmd = ["docker", "run", "--rm",
           "--cpus", str(args.per_run_cpus), "--memory", f"{args.per_run_ram_gb}g",
           "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
           "-e", "XDG_CACHE_HOME=/work/.jitcache",
           *mount_flags(), "-w", "/work", args.image,
           "python", "src/edpvr.py", "--probe", "--no-index", "--name", name,
           "--geom", str(s["geom"])]
    for k in ("f_lv", "f_rv", "shift_lv", "shift_rv", "sept"):
        if k in s:
            cmd += [f"--{k}", str(s[k])]
    err = ""
    probe_path = os.path.join(REPO_LOCAL, "results", name, "probe.json")
    for _ in range(1 + max(0, args.probe_retries)):
        try:
            os.remove(probe_path)
        except OSError:
            pass
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            err = (p.stderr or p.stdout or "").strip().replace("\n", " ")[-240:]
            continue
        try:
            return json.load(open(probe_path)), ""
        except Exception:
            err = (p.stderr or p.stdout or "").strip().replace("\n", " ")[-240:]
    return None, err or "no probe.json produced"


def probe_all(candidates, args):
    import geom_filter as gf
    uniq = OrderedDict()
    for s in candidates:
        uniq.setdefault(geom_key(s), s)
    if args.no_probe:
        print(f"[screen] --no-probe: accepting all {len(uniq)} distinct geometries "
              f"(guards already applied in plan.py)")
        return {k: (True, "no-probe") for k in uniq}, {}
    conc = max(1, min(int(args.max_ram_gb // args.per_run_ram_gb),
                      int(args.max_cpus // args.per_run_cpus)))
    print(f"[screen] probing {len(uniq)} distinct geometries from {len(candidates)} candidates "
          f"({conc} concurrent)...")
    probes, errors = {}, {}
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = {ex.submit(probe_one, s, args,
                          "screen_probe_" + hashlib.sha1(repr(k).encode()).hexdigest()[:12]): k
                for k, s in uniq.items()}
        done = 0
        for fut in as_completed(futs):
            pr, err = fut.result()
            probes[futs[fut]] = pr
            if pr is None:
                errors[futs[fut]] = err
            done += 1
            if done % 100 == 0:
                print(f"[screen]   probed {done}/{len(uniq)}")
    verdict = {}
    for k, s in uniq.items():
        pr = probes.get(k)
        mode = s.get("_filter", "extended")
        verdict[k] = (False, "probe_failed") if pr is None else gf.accept(pr, mode)
    nfail = sum(1 for v in verdict.values() if v[1] == "probe_failed")
    if nfail:
        print(f"[screen] NOTE: {nfail}/{len(uniq)} geometries failed to PROBE (docker/probe error, not "
              f"a realism rejection). If this is large, it is the likely cause of a short queue -- see "
              f"the probe_errors sample in screen_report.json.")
    return verdict, errors


def reason_kind(reason):
    return reason.split("=")[0] if "=" in reason else reason


def trim(candidates, verdict, targets):
    rep = {"A": Counter(), "B": Counter(), "C": Counter()}
    rejected = []

    def note_reject(arm, cfg, reason):
        rejected.append({"arm": arm, "geom": cfg["geom"], "reason": reason, "kind": reason_kind(reason),
                         "f_lv": cfg.get("f_lv"), "f_rv": cfg.get("f_rv"), "shift_lv": cfg.get("shift_lv"),
                         "shift_rv": cfg.get("shift_rv"), "sept": cfg.get("sept"),
                         "cohort": cfg.get("_cohort"), "example_config_id": cfg.get("config_id")})

    A = [c for c in candidates if c["_arm"] == "A_normal"]
    setmap = OrderedDict()
    for c in A:
        setmap.setdefault(c["_setid"], []).append(c)
    keptA, nsets = [], 0
    for sid, cs in setmap.items():
        ok, reason = verdict[geom_key(cs[0])]
        rep["A"]["valid_sets" if ok else "rejected_sets"] += 1
        if ok and nsets < targets["A_sets"]:
            keptA.extend(cs); nsets += 1
        elif not ok:
            note_reject("A_normal", cs[0], reason)
    rep["A"]["kept_sets"] = nsets
    rep["A"]["kept_configs"] = len(keptA)

    B = [c for c in candidates if c["_arm"] == "B_sweep"]
    keptB = []
    seenB = set()
    for c in B:
        ok, reason = verdict[geom_key(c)]
        rep["B"]["valid" if ok else "rejected"] += 1
        if ok:
            keptB.append(c)
        elif geom_key(c) not in seenB:
            seenB.add(geom_key(c)); note_reject("B_sweep", c, reason)
    rep["B"]["kept_configs"] = len(keptB)

    C = [c for c in candidates if c["_arm"] == "C_extended"]
    keptC, nC = [], 0
    for c in C:
        ok, reason = verdict[geom_key(c)]
        rep["C"]["valid" if ok else "rejected"] += 1
        if ok and nC < targets["C"]:
            keptC.append(c); nC += 1
        elif not ok:
            note_reject("C_extended", c, reason)
    rep["C"]["kept_configs"] = len(keptC)

    return keptA + keptB + keptC, rep, rejected


def expand_and_shuffle(configs, seed):
    runs = []
    for c in configs:
        for comp_label, form, kappa in CONDITIONS:
            r = dict(c)
            r["formulation"] = form
            r["kappa"] = kappa
            r["_comp"] = comp_label
            r["_pair"] = c["config_id"]
            runs.append(r)
    random.Random(seed).shuffle(runs)
    for i, r in enumerate(runs):
        r["run_id"] = f"sfin_{i:05d}"
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=os.path.join(HERE, "designs", "candidates.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "designs", "runqueue.json"))
    ap.add_argument("--image", default="rvedpvr-fenics")
    ap.add_argument("--max-cpus", type=float, default=16.0)
    ap.add_argument("--max-ram-gb", type=float, default=48.0)
    ap.add_argument("--per-run-cpus", type=float, default=1.0)
    ap.add_argument("--per-run-ram-gb", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--probe-retries", type=int, default=1,
                    help="extra probe attempts on failure (docker-in-docker probes are flaky)")
    ap.add_argument("--target-runs", type=int, default=0,
                    help="expected total runs (0 = design meta target, else 10000); shorter -> WARNING")
    ap.add_argument("--compressibility", choices=["single", "both", "incompressible"], default="single",
                    help="'single' (default) = one solve per config at kappa=0.003, the production "
                         "setting; 'both' also runs kappa=0.10 as a NUMERICAL sensitivity (do not "
                         "report it as physiology); 'incompressible' uses the mixed u-p formulation "
                         "(slow -- use for a small cross-check subset).")
    ap.add_argument("--no-topup", action="store_true",
                    help="disable the top-up loop (single screening pass only)")
    ap.add_argument("--max-topup-rounds", type=int, default=12,
                    help="cap on top-up rounds when an arm is short after realism rejects")
    args = ap.parse_args()

    doc = json.load(open(args.candidates))
    cands, meta = doc["candidates"], doc["meta"]
    targets = meta["targets"]
    set_conditions(args.compressibility)
    if not args.target_runs:
        t = meta.get("targets", {})
        n_cfg = (t.get("A_sets", 0) * len(meta.get("ratios4", [])) +
                 t.get("B_geoms", 0) * t.get("B_ratios", 0) + t.get("C", 0))
        args.target_runs = meta.get("target_runs") or (n_cfg * len(CONDITIONS)) or 10000

    print(f"[screen] compressibility mode '{args.compressibility}' -> "
          f"{len(CONDITIONS)} solve(s) per config: {[c[0] for c in CONDITIONS]}")

    verdict, errors = probe_all(cands, args)
    kept, rep, rejected = trim(cands, verdict, targets)

    def short_arms():
        s = []
        if rep["A"].get("kept_sets", 0) < targets["A_sets"]:
            s.append("A_normal")
        if rep["C"].get("kept_configs", 0) < targets["C"]:
            s.append("C_extended")
        return s
    if not args.no_probe and not args.no_topup and short_arms():
        import plan
        steps = meta.get("steps", 20)
        stall, last_kept, rnd = 0, len(kept), 0
        while short_arms() and rnd < args.max_topup_rounds and stall < 2:
            rnd += 1
            need = short_arms()
            more = plan.make_candidates(steps, args.seed + 1000 * rnd, meta.get("oversample_factor", 1.0))
            add = [c for c in more if c["_arm"] in need]
            base = 1 + max(int(c["config_id"]) for c in cands)
            for j, c in enumerate(add):
                c["config_id"] = f"{base + j:05d}"
                if "_setid" in c:
                    c["_setid"] = f"R{rnd}_{c['_setid']}"
            newv, newerr = probe_all(add, args)
            verdict.update(newv); errors.update(newerr)
            cands.extend(add)
            kept, rep, rejected = trim(cands, verdict, targets)
            print(f"[screen] top-up round {rnd}: +{len(add)} candidates for {need} -> kept "
                  f"A={rep['A'].get('kept_sets')}/{targets['A_sets']} sets, "
                  f"C={rep['C'].get('kept_configs')}/{targets['C']} ({len(kept)} configs)")
            stall = stall + 1 if len(kept) <= last_kept else 0
            last_kept = len(kept)

    runs = expand_and_shuffle(kept, args.seed)

    arms = Counter(r["_arm"] for r in runs)
    forms = Counter(r["_comp"] for r in runs)
    by_reason = Counter(r["kind"] for r in rejected)
    by_arm = Counter(r["arm"] for r in rejected)
    by_arm_reason = Counter(f"{r['arm']}:{r['kind']}" for r in rejected)
    by_geom = Counter(str(r["geom"]) for r in rejected)
    probe_err_sample = [{"reason": v, "n": 1} for v in list(errors.values())[:5]]
    rejections = {"total_distinct_geoms_rejected": len(rejected),
                  "by_reason": dict(by_reason.most_common()),
                  "by_arm": dict(by_arm), "by_arm_reason": dict(by_arm_reason.most_common()),
                  "by_base_geom": dict(by_geom.most_common()),
                  "probe_failed": int(by_reason.get("probe_failed", 0)),
                  "probe_error_examples": [e["reason"] for e in probe_err_sample]}
    report = {"n_runs": len(runs), "n_configs": len(kept), "target_runs": args.target_runs,
              "shortfall_runs": max(0, args.target_runs - len(runs)),
              "arms_runs": dict(arms), "compressibility": dict(forms),
              "trim": {k: dict(v) for k, v in rep.items()},
              "rejections": rejections, "no_probe": args.no_probe, "seed": args.seed}
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    qhash = hashlib.sha256(json.dumps(runs, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    report["queue_hash"] = qhash
    with open(args.out, "w") as f:
        json.dump({"meta": {**report, "ratios4": meta.get("ratios4"), "lvedp": meta.get("lvedp")},
                   "runs": runs}, f, indent=1)
    print(f"[screen] queue_hash {qhash[:12]} -- copy designs/runqueue.json to every machine; "
          f"they must all print this value.")
    with open(os.path.join(HERE, "designs", "screen_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    import csv as _csv
    with open(os.path.join(HERE, "designs", "rejected.csv"), "w", newline="") as f:
        cols = ["arm", "geom", "cohort", "kind", "reason", "f_lv", "f_rv", "shift_lv", "shift_rv",
                "sept", "example_config_id"]
        w = _csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rejected:
            w.writerow({k: r.get(k) for k in cols})
    print("=" * 60)
    print(f"[screen] kept {len(kept)} configs -> {len(runs)} runs ({len(CONDITIONS)} solve(s) per config)")
    print(f"[screen] arms: {dict(arms)}")
    print(f"[screen] trim: A={rep['A'].get('kept_configs')} B={rep['B'].get('kept_configs')} "
          f"C={rep['C'].get('kept_configs')}")
    print(f"[screen] dismissed {len(rejected)} distinct geometries: by reason {dict(by_reason.most_common())}")
    print(f"[screen] wrote {args.out} + screen_report.json + rejected.csv")
    if len(runs) < args.target_runs:
        print(f"[screen] WARNING: {len(runs)} < {args.target_runs} runs (short by "
              f"{args.target_runs - len(runs)}). The top-up loop hit its round cap. See screen_report.json "
              f"'rejections': if 'probe_failed' is large it is a DOCKER/probe problem (fix the runner); "
              f"otherwise raise --max-topup-rounds or plan.py --oversample.")
    else:
        print(f"[screen] target met: {len(runs)} >= {args.target_runs} runs ready to solve.")


if __name__ == "__main__":
    main()
