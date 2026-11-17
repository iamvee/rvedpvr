#!/usr/bin/env bash
#   ./run_study.sh          solve the queue
#   ./run_study.sh 50       first 50 queued runs
#   ./run_study.sh plan     build the queue only
set -euo pipefail
cd "$(dirname "$0")"

ARG="${1:-}"
EXTRA="--max-cpus 16 --per-run-cpus 1 --per-run-ram-gb 2 --max-ram-gb 20 --run-retries 1"
MAXRESTARTS=40
MODE=""
if [ "$ARG" = "plan" ]; then MODE="--plan-only"
elif [ -n "$ARG" ]; then MODE="--limit $ARG"; fi

if ! docker info >/dev/null 2>&1; then
  echo "*** Docker is not responding. Start Docker / Docker Desktop, then re-run. ***" >&2
  exit 1
fi

echo "[1/3] Building the FE solver image (rvedpvr-fenics) - first run only..."
docker build -t rvedpvr-fenics -f docker/solver.Dockerfile .

echo "[2/3] Building the runner image (rvedpvr-runner) - first run only..."
docker build -t rvedpvr-runner -f docker/runner.Dockerfile .

try=0
while : ; do
  try=$((try + 1))
  before=$(find results -name measurements.json 2>/dev/null | wc -l)
  echo "[3/3] Launching the study ${MODE}  (restart budget ${try}/${MAXRESTARTS}) ..."
  docker rm -f rvedpvr-runner-c >/dev/null 2>&1 || true
  set +e
  docker run --rm --name rvedpvr-runner-c \
    -e RUNNER_SELF=rvedpvr-runner-c \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "$PWD:/work" -w /work \
    rvedpvr-runner \
    python run.py --image rvedpvr-fenics $EXTRA $MODE
  rc=$?
  set -e
  [ "$rc" -eq 0 ] && break
  [ "$ARG" = "plan" ] && { echo "*** queue build failed (exit $rc). ***" >&2; exit 1; }
  after=$(find results -name measurements.json 2>/dev/null | wc -l)
  [ "$after" -gt "$before" ] && try=0   # progress resets the budget; only consecutive no-progress gives up
  [ "$try" -ge "$MAXRESTARTS" ] && { echo "*** gave up after $MAXRESTARTS restarts with NO progress (exit $rc) -- lower --max-ram-gb. ***" >&2; exit 1; }
  echo "*** Runner exited ($rc) -- Docker/VM hiccup. Waiting 30s, then RESUMING (finished runs skipped)."
  echo "    Ctrl+C to stop. ***" >&2
  sleep 30
done

echo
echo "Done.  See results/summary.csv and results/<run_id>/"
