import argparse, json, os, shutil, sys

REQUIRED = ("measurements.json", "params.json", "pv.csv")
IDENTITY = ("geom", "af", "bf", "rvedp", "lvedp", "kappa", "formulation", "material")


def complete(d):
    return all(os.path.isfile(os.path.join(d, f)) for f in REQUIRED)


def run_dirs(root):
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and complete(p):
            yield name, p


def same_place(a, b):
    ra, rb = os.path.realpath(a), os.path.realpath(b)
    if os.path.normcase(ra) == os.path.normcase(rb):
        return True
    try:
        common = os.path.normcase(os.path.commonpath([ra, rb]))
        return common in (os.path.normcase(ra), os.path.normcase(rb))
    except ValueError:
        return False


def place(src, dst, move):
    tmp = dst + ".partial"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    if move:
        shutil.move(src, tmp)
    else:
        shutil.copytree(src, tmp)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--into", default="results", help="destination results/ dir (created if absent)")
    ap.add_argument("--from", dest="sources", nargs="+", required=True,
                    help="one or more source results/ dirs from other machines")
    ap.add_argument("--queue", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    "designs", "runqueue.json"),
                    help="run queue to audit coverage against (pass '' to skip the audit)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a run that already exists in the destination")
    ap.add_argument("--move", action="store_true",
                    help="move instead of copy (faster, empties the source)")
    ap.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    args = ap.parse_args()

    dest = os.path.abspath(args.into)
    os.makedirs(dest, exist_ok=True)
    have = {rid for rid, _ in run_dirs(dest)}
    print(f"destination {dest}: {len(have)} complete runs already present")

    queue = {}
    if args.queue and os.path.isfile(args.queue):
        doc = json.load(open(args.queue))
        queue = {r["run_id"]: r for r in doc["runs"]}
        qh = (doc.get("meta") or {}).get("queue_hash")
        print(f"queue {args.queue}: {len(queue)} runs" + (f", queue_hash {qh[:12]}" if qh else ""))
    elif args.queue:
        print(f"! queue {args.queue} not found -- skipping the coverage audit")

    added = skipped = replaced = mism = 0
    mismatches = []
    for src in args.sources:
        src = os.path.abspath(src)
        if same_place(src, dest):
            print(f"  ! skipping {src} (same as, or nested with, the destination)"); continue
        if not os.path.isdir(src):
            print(f"  ! skipping {src} (not a directory)"); continue
        n_src = n_add = n_skip = 0
        for rid, path in run_dirs(src):
            n_src += 1
            if queue and rid in queue:
                q = queue[rid]
                try:
                    p = json.load(open(os.path.join(path, "params.json")))
                except Exception:
                    p = {}
                bad = [k for k in IDENTITY if k in q and k in p and str(q[k]) != str(p[k])]
                if bad:
                    mism += 1
                    if len(mismatches) < 10:
                        mismatches.append(f"{rid}: {', '.join(bad)}")
                    continue
            target = os.path.join(dest, rid)
            if rid in have and not args.overwrite:
                n_skip += 1; skipped += 1; continue
            if not args.dry_run:
                if os.path.exists(target):
                    replaced += 1
                place(path, target, args.move)
            have.add(rid); n_add += 1; added += 1
        print(f"  {src}: {n_src} complete runs -> {n_add} merged, {n_skip} already had")

    print("-" * 70)
    verb = "would merge" if args.dry_run else "merged"
    print(f"{verb} {added} runs ({replaced} replaced, {skipped} skipped as duplicates)")
    if mism:
        print(f"*** REFUSED {mism} runs whose params.json disagrees with the queue row of the same")
        print(f"    run_id. That source was built from a DIFFERENT queue -- do not merge it.")
        for m in mismatches:
            print(f"      {m}")
    print(f"destination now holds {len(have)} complete runs")

    if queue:
        missing = sorted(rid for rid in queue if rid not in have)
        extra = [rid for rid in have if rid not in queue]
        pct = 100.0 * (len(queue) - len(missing)) / max(len(queue), 1)
        print(f"coverage: {len(queue)-len(missing)}/{len(queue)} of the queue ({pct:.1f}%)")
        if missing:
            print(f"MISSING {len(missing)} runs. First 20: {', '.join(missing[:20])}")
            if not args.dry_run:
                mp = os.path.join(dest, "missing.txt")
                open(mp, "w").write("\n".join(missing) + "\n")
                print(f"full list written to {mp}")
            print("Re-run those on any machine, then merge again.")
        else:
            print("coverage: COMPLETE -- every queued run is present.")
        if extra:
            print(f"note: {len(extra)} runs present that are not in this queue "
                  f"(smoke tests / a different study?)")

    if not args.dry_run:
        with open(os.path.join(dest, "merge_log.json"), "w") as f:
            json.dump({"sources": [os.path.abspath(s) for s in args.sources],
                       "added": added, "replaced": replaced, "skipped": skipped,
                       "refused_mismatched": mism, "total_after": len(have),
                       "queue_runs": len(queue) or None,
                       "missing": len([r for r in queue if r not in have]) if queue else None},
                      f, indent=2)
    print("Next: python aggregate.py   (-> results/summary.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
