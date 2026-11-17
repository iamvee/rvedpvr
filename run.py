import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

EXIT_FATAL_CONFIG = 4

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_LOCAL = HERE
REPO_HOST = os.environ.get("HOST_REPO", REPO_LOCAL)
RUNNER_SELF = os.environ.get("RUNNER_SELF")
MAT_KEYS = ("material", "a", "b", "af", "bf", "c", "B1", "B2", "B3", "qform", "endo", "epi",
            "lvedp", "rvedp", "f_lv", "f_rv", "shift_lv", "shift_rv", "sept", "kappa")


def mount_flags():
    return ["--volumes-from", RUNNER_SELF] if RUNNER_SELF else ["-v", f"{REPO_HOST}:/work"]


def queue_hash(runs):
    payload = [{k: v for k, v in r.items() if k != "_viz"} for r in runs]
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def fatal(msg):
    print(msg); sys.exit(EXIT_FATAL_CONFIG)


def hms(sec):
    sec = int(max(0, sec)); h, m = divmod(sec, 3600); m, s = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


REQUIRED = ("measurements.json", "params.json", "pv.csv")


def run_complete(dirpath):
    return all(os.path.exists(os.path.join(dirpath, f)) for f in REQUIRED)


def already_done(run_id, retry_incomplete=False):
    d = os.path.join(REPO_LOCAL, "results", run_id)
    if not run_complete(d):
        return False
    try:
        m = json.load(open(os.path.join(d, "measurements.json")))
    except Exception:
        return False
    if retry_incomplete and not m.get("complete"):
        return False
    return True


_TRANSIENT = ("unexpected EOF", "Cannot connect to the Docker daemon", "connection refused",
              "i/o timeout", "context deadline exceeded", "500 Internal Server Error",
              "container not found", "No such container")


def run_one(r, args):
    name = r["run_id"]
    cmd = ["docker", "run", "--rm",
           "--cpus", str(args.per_run_cpus), "--memory", f"{args.per_run_ram_gb}g",
           "-e", "OMP_NUM_THREADS=1", "-e", "OPENBLAS_NUM_THREADS=1",
           "-e", "XDG_CACHE_HOME=/work/.jitcache",
           *mount_flags(), "-w", "/work", args.image,
           "python", "src/edpvr.py", "--name", name, "--no-index",
           "--geom", str(r["geom"]), "--formulation", r["formulation"],
           "--steps", str(r.get("steps", 20))]
    for k in MAT_KEYS:
        if k in r:
            cmd += [f"--{k}", str(r[k])]
    if not r.get("_viz"):
        cmd.append("--no-frames")
    outdir = os.path.join(REPO_LOCAL, "results", name)
    mj = os.path.join(outdir, "measurements.json")
    try:
        os.remove(mj)
    except OSError:
        pass
    t = time.time()
    attempts = 1 + max(0, getattr(args, "run_retries", 0))
    transient = False
    for a in range(attempts):
        p = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(mj):
            break
        transient = p.returncode == 125 or any(s in (p.stderr or "") + (p.stdout or "") for s in _TRANSIENT)
        if a < attempts - 1 and transient:
            time.sleep(8)
            continue
        break
    dt = time.time() - t
    ran = os.path.exists(mj); conv = req = None; ok = False
    if ran:
        try:
            m = json.load(open(mj)); conv = m.get("converged_steps"); req = m.get("requested_steps")
            ok = bool(m.get("complete"))
        except Exception:
            ran = False
    return {"r": r, "ok": ok, "ran": ran, "dt": dt, "conv": conv, "req": req,
            "infra": (not ran) and transient}


HDR = ("  #   run_id       arm       geom  form  ratio   af    bf   endo   steps  status    "
       "secs   avg    ETA")
def fmt_row(idx, total, res, avg, eta):
    r = res["r"]; form = "C" if r.get("_comp", r["formulation"]) == "compressible" else "N"
    ratio = r.get("_ratio", (r["rvedp"] / r["lvedp"] if r.get("lvedp") else 0))
    steps = f"{res['conv']}/{res['req']}" if res["conv"] is not None else "-"
    status = "ok " if res["ok"] else ("inc" if res["ran"] else "CRASH")
    return (f"{idx:5d}/{total} {r['run_id']:<12} {r['_arm']:<10} {str(r['geom']):>4}  {form}   "
            f"{ratio:5.2f}  {r.get('af',0):5.1f} {r.get('bf',0):4.1f} {r.get('endo',0):5.1f}  "
            f"{steps:>6}  {status:<5} {res['dt']:6.0f} {avg:5.0f}  {hms(eta)}")


def main():
    from config import CFG
    _m = CFG.machine

    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=os.path.join(HERE, "designs", "runqueue.json"))
    ap.add_argument("--image", default="rvedpvr-fenics")
    ap.add_argument("--max-cpus", type=float, default=_m.max_cpus or 16.0)
    ap.add_argument("--max-ram-gb", type=float, default=_m.max_ram_gb or 20.0)
    ap.add_argument("--per-run-cpus", type=float, default=_m.per_run_cpus)
    ap.add_argument("--per-run-ram-gb", type=float, default=_m.per_run_ram_gb)
    ap.add_argument("--limit", type=int, default=0,
                    help="run only the first N of THIS MACHINE'S SLICE (smoke test). Applied AFTER "
                         "--range/--shard, so it never reaches into another machine's slice.")
    ap.add_argument("--plan-scale", type=float, default=1.0,
                    help="passed to plan.py --scale when building the queue: 1 gives ~5000 runs, "
                         "2 gives ~9500")
    ap.add_argument("--range", default=_m.range or "", metavar="A-B",
                    help="solve only queue indices [A, B) on this machine, e.g. 0-4000. "
                         "Runs already finished here are skipped, so re-running resumes.")
    ap.add_argument("--shard", default=_m.shard or "", metavar="i/N",
                    help="alternative to --range: split the queue into N equal parts and take "
                         "part i (1-based), e.g. 2/4")
    ap.add_argument("--concurrency", type=int, default=_m.concurrency or 0,
                    help="force N solves in flight, overriding the cpu/ram auto-sizing")
    ap.add_argument("--viz-per-arm", type=int, default=1, help="keep solver frames for N runs/arm")
    ap.add_argument("--no-probe", action="store_true",
                    help="build the queue WITHOUT the Docker geometry probe. The probe runs ~3000 "
                         "throwaway containers and can take hours on a fresh machine, while plan.py's "
                         "geometric guards have already been applied.")
    ap.add_argument("--build-queue", action="store_true",
                    help="run plan.py then screen.py to (re)build the queue before solving")
    ap.add_argument("--plan-only", action="store_true",
                    help="build the queue (plan.py -> screen.py) and stop; solve nothing")
    ap.add_argument("--run-retries", type=int, default=_m.run_retries,
                    help="retry a run this many times if the CONTAINER errored transiently (docker EOF/"
                         "daemon hiccup) and produced no output; solve divergences are not retried")
    ap.add_argument("--retry-incomplete", action="store_true",
                    help="on resume, also re-run runs that produced measurements but did NOT complete")
    args = ap.parse_args()
    CFG.echo()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    DOCKER_WAIT_S = float(os.environ.get("DOCKER_WAIT_S", 7200))
    t_wait = time.time()
    while True:
        probe = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            break
        el = time.time() - t_wait
        if el > DOCKER_WAIT_S:
            print(f"[run] docker still unreachable after {el/60:.0f} min -- continuing anyway; "
                  f"per-run retries will absorb it.")
            break
        print(f"[run] waiting for the Docker engine ({el/60:.1f} min elapsed): "
              f"{(probe.stderr or probe.stdout).strip()[:160]}")
        time.sleep(15)

    if (args.range or args.shard) and not os.path.exists(args.queue):
        print("=" * 88)
        print(f"[run] WARNING: {args.queue} is missing but a slice ({args.range or args.shard}) was")
        print( "      requested. Building the queue locally. On a MULTI-MACHINE study this is wrong --")
        print( "      every box must read the SAME queue file. Check the printed queue_hash matches.")
        print("=" * 88)
    if args.build_queue or args.plan_only or not os.path.exists(args.queue):
        print("[run] building queue (plan.py -> screen.py) ...")
        plan_cmd = [sys.executable, os.path.join(HERE, "plan.py")]
        if args.plan_scale != 1.0:
            plan_cmd += ["--scale", str(args.plan_scale)]
        rc = subprocess.run(plan_cmd).returncode
        if rc != 0:
            print(f"[run] plan.py exited {rc}; retrying once ...")
            rc = subprocess.run(plan_cmd).returncode
        screen_cmd = [sys.executable, os.path.join(HERE, "screen.py"),
                      "--image", args.image, "--max-cpus", str(args.max_cpus)]
        if args.no_probe:
            screen_cmd.append("--no-probe")
        rc = subprocess.run(screen_cmd).returncode
        if rc != 0 and "--no-probe" not in screen_cmd:
            print(f"[run] screen.py exited {rc}; falling back to --no-probe ...")
            rc = subprocess.run(screen_cmd + ["--no-probe"]).returncode
        if rc != 0:
            print(f"[run] screen.py still failing (rc={rc}); continuing to see if a usable queue exists.")

    def _load_queue(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    try:
        doc = _load_queue(args.queue)
        runs_all = doc["runs"]
        if not runs_all:
            raise ValueError("queue holds 0 runs")
    except Exception as e:
        print(f"[run] queue at {args.queue} is unusable ({e}); moving it aside and rebuilding.")
        try:
            os.replace(args.queue, args.queue + ".bad")
        except OSError:
            pass
        subprocess.run([sys.executable, os.path.join(HERE, "plan.py")])
        subprocess.run([sys.executable, os.path.join(HERE, "screen.py"), "--image", args.image,
                        "--max-cpus", str(args.max_cpus), "--no-probe"])
        doc = _load_queue(args.queue)
        runs_all = doc["runs"]
    qhash = queue_hash(runs_all)
    stored = (doc.get("meta") or {}).get("queue_hash")
    if stored and stored != qhash:
        if args.range or args.shard:
            fatal(f"[run] FATAL: {args.queue} does not match its own recorded hash "
                  f"({stored[:12]} != {qhash[:12]}) AND a slice was requested. A queue index only "
                  f"means the same run if every machine reads the same file. Re-copy designs/.")
        print(f"[run] queue hash changed ({stored[:12]} -> {qhash[:12]}); it was edited since it was "
              f"built. No slice requested, so re-stamping and continuing.")
        doc.setdefault("meta", {})["queue_hash"] = qhash
        try:
            tmp = args.queue + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
            os.replace(tmp, args.queue)
        except OSError as e:
            print(f"[run] could not re-stamp the queue ({e}); continuing anyway.")
    if args.plan_only:
        print(f"[run] --plan-only: queue built -- {len(runs_all)} runs, queue_hash {qhash[:12]}.")
        print(f"[run] Copy designs/runqueue.json to every machine, then give each one a shard:")
        print(f"[run]     run_study.bat 1/3 20   (and 2/3, 3/3 on the other boxes)")
        return
    print(f"[run] queue_hash {qhash[:12]} -- every machine in this study must print the same value.")
    per_arm = defaultdict(int)
    for r in runs_all:
        if per_arm[r["_arm"]] < args.viz_per_arm:
            r["_viz"] = True; per_arm[r["_arm"]] += 1

    lo, hi = 0, len(runs_all)
    slice_label = "whole queue"
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.replace("/", " ").split())
        except Exception:
            fatal(f"[run] --shard must look like 'i/N' (got {args.shard!r})")
        if not (1 <= i <= n):
            fatal(f"[run] --shard i must be in 1..N (got {args.shard})")
        per = (len(runs_all) + n - 1) // n
        lo, hi = (i - 1) * per, min(i * per, len(runs_all))
        slice_label = f"shard {i}/{n}"
    elif args.range:
        if "-" not in args.range:
            fatal(f"[run] --range must contain a '-' (got {args.range!r}). Use A-B, A- or -B; a bare "
                  f"number is ambiguous. To cap the run count use --limit.")
        try:
            a, _, b = args.range.partition("-")
            lo = int(a) if a.strip() else 0
            hi = int(b) if b.strip() else len(runs_all)
        except Exception:
            fatal(f"[run] --range must look like 'A-B' (got {args.range!r})")
        lo, hi = max(0, lo), min(len(runs_all), hi)
        slice_label = f"index range {lo}-{hi}"
    if lo >= hi:
        print(f"[run] empty slice ({slice_label}); queue holds {len(runs_all)} runs. Falling back to "
              f"the whole queue so the machine does not sit idle.")
        lo, hi = 0, len(runs_all)
        slice_label = "whole queue (slice was empty)"
    runs = runs_all[lo:hi]
    if args.limit:
        runs = runs[:args.limit]
        hi = lo + len(runs)

    todo = [r for r in runs if not already_done(r["run_id"], args.retry_incomplete)]
    done0 = len(runs) - len(todo)
    auto = max(1, min(int(args.max_cpus // args.per_run_cpus),
                      int(args.max_ram_gb // args.per_run_ram_gb)))
    conc = auto
    if args.concurrency:
        want = max(1, args.concurrency)
        cap = max(auto, int(auto * 1.5))
        conc = min(want, cap)
        if conc != want:
            print("=" * 88)
            print(f"[run] requested concurrency {want} exceeds what this machine can sustain "
                  f"(auto-size {auto} from {args.max_cpus:g} cpus / {args.max_ram_gb:g} GB at "
                  f"{args.per_run_cpus:g} cpu + {args.per_run_ram_gb:g} GB per solve).")
            print(f"[run] clamping to {conc}. Raise machine.max_cpus / max_ram_gb in "
                  f"STUDY_CONFIG.yaml if the Docker VM really is bigger than detected.")
            print("=" * 88)
    print("=" * 88)
    print(f"runner | {slice_label} of {len(runs_all)} | this slice {len(runs)} | "
          f"already done here {done0} | to run {len(todo)} | concurrency {conc}")
    print(f"arms: {doc['meta'].get('arms_runs')} | image {args.image}")
    if lo or hi < len(runs_all):
        print(f"NOTE: this machine solves queue indices [{lo}, {hi}) ONLY -- runs {runs[0]['run_id']} "
              f"... {runs[-1]['run_id']}.")
        print(f"      Other machines must take the remaining indices; afterwards merge with:")
        print(f"          python merge_results.py --into results --from <other>/results "
              f"--queue designs/runqueue.json")
    print("=" * 88)
    os.makedirs(os.path.join(REPO_LOCAL, "results"), exist_ok=True)
    if not todo:
        print("[run] nothing to do (all queued runs complete)."); return

    def form_key(r):
        return (r.get("material", "ho"), r.get("qform"), r.get("formulation"))
    warm, seen = [], set()
    for r in todo:
        k = form_key(r)
        if k not in seen:
            seen.add(k); warm.append(r)
    print(f"[warmup] compiling {len(warm)} distinct form(s) serially into /work/.jitcache ...")
    tsum = 0.0; nfin = 0; ok_n = 0; fail_n = 0; warm_ids = set(); warm_failed = []
    for w in warm:
        res = run_one(w, args)
        tsum += res["dt"]; nfin += 1; ok_n += int(res["ok"]); fail_n += int(not res["ok"])
        warm_ids.add(w["run_id"])
        print(f"[warmup] {w['run_id']} {form_key(w)} "
              f"{'ok' if res['ok'] else ('inc' if res['ran'] else 'CRASH')} {res['dt']:.0f}s")
        if not res["ran"]:
            warm_failed.append(w["run_id"])
            print("=" * 88)
            print(f"[warmup] {w['run_id']} produced no output. QUARANTINING this form and continuing; "
                  f"the batch will proceed without pre-compiling it.")
            print("=" * 88)
    rest = [r for r in todo if r["run_id"] not in warm_ids]
    t0 = time.time()
    print(HDR)
    queue = list(enumerate(rest)); idx = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        inflight = {}
        def submit():
            nonlocal idx
            if idx >= len(queue):
                return False
            i, r = queue[idx]; idx += 1
            inflight[ex.submit(run_one, r, args)] = idx - 1
            return True
        for _ in range(conc):
            submit()
        infra_streak = 0
        while inflight:
            fin, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in fin:
                i_run = inflight.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:
                    print(f"[run] worker for queue index {i_run} raised {type(e).__name__}: {e} "
                          f"-- recording as failed and continuing.")
                    res = {"r": queue[i_run][1] if i_run < len(queue) else {"run_id": "?", "geom": "?",
                           "_arm": "?", "formulation": "compressible"},
                           "ok": False, "ran": False, "dt": 0.0, "conv": None, "req": None,
                           "infra": False}
                infra_streak = infra_streak + 1 if res.get("infra") else 0
                if infra_streak >= max(2 * conc, 12):
                    print("=" * 88 + f"\n[run] {infra_streak} consecutive container/daemon errors -- the "
                          "Docker engine looks DOWN. Aborting so the launcher relaunches and resumes once "
                          "Docker recovers.\n" + "=" * 88)
                    sys.exit(5)
                nfin += 1; ok_n += int(res["ok"]); fail_n += int(not res["ok"])
                tsum += res["dt"]; avg = tsum / nfin
                remaining = len(todo) - nfin
                eta = remaining * avg / max(conc, 1)
                print(fmt_row(done0 + nfin, len(runs), res, avg, eta))
                if nfin % 50 == 0:
                    print(f"    ---- {done0+nfin}/{len(runs)} done | ok {ok_n} fail {fail_n} "
                          f"| in-flight {len(inflight)} | elapsed {hms(time.time()-t0)} "
                          f"| ETA {hms(eta)} ----")
                submit()

    agg = subprocess.run([sys.executable, os.path.join(HERE, "aggregate.py")],
                         capture_output=True, text=True)
    if agg.returncode != 0:
        print("[run] warning: aggregate.py failed; results/summary.csv not refreshed.")
        print("     ", (agg.stderr or agg.stdout).strip()[:400])
    print("=" * 88)
    print(f"done: {ok_n} ok, {fail_n} failed of {len(todo)} run "
          f"({done0} pre-existing) in {hms(time.time()-t0)}")
    if lo or hi < len(runs_all):
        print(f"This machine covered queue indices [{lo}, {hi}) of {len(runs_all)}. "
              f"Collect every machine's results/ and merge before analysing.")


if __name__ == "__main__":
    main()
