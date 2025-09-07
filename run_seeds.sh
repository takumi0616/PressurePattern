#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Seed orchestrator for running paired GPU jobs via notify-run.

Default behavior:
- Start at seed 19 and run up to seed 50
- Launch two jobs per "pair":
  - GPU0 runs seed s
  - GPU1 runs seed s+1 (skipped if s+1 > end)
- Wait for both jobs to finish, then proceed to the next pair (s += 2)
- Output logs to seed{SEED}.out in this directory
- Uses notify-run channel "gpu02" by default

Assumptions:
- This script resides in src/PressurePattern and will cd to that directory on start
- main_v5.py accepts --gpu and --seed
- notify-run is available as: notify-run <channel> -- <command...>

Usage:
  bash run_seeds.sh [--start N] [--end M] [--gpu0 ID] [--gpu1 ID] \
                    [--channel NAME] [--script PATH] [--python BIN] [--] [EXTRA ARGS...]

Examples:
  # Default: seeds 19..50 on GPUs 0/1, channel gpu02
  bash run_seeds.sh

  # Explicit range
  bash run_seeds.sh --start 19 --end 50 --gpu0 0 --gpu1 1 --channel gpu02 --script main_v5.py

  # Pass extra args to main_v5.py (after --)
  bash run_seeds.sh --start 1 --end 10 -- --result-dir ./results

  # Keep orchestrator running after terminal closes
  nohup bash run_seeds.sh --start 19 --end 50 > orchestrator.log 2>&1 &

Options:
  --start N       First seed to run (default: 19)
  --end M         Last seed to run (default: 50)
  --gpu0 ID       GPU id for the first seed in each pair (default: 0)
  --gpu1 ID       GPU id for the second seed in each pair (default: 1)
  --channel NAME  notify-run channel (default: gpu02)
  --script PATH   Target script to launch (default: main_v5.py)
  --python BIN    Python binary to use (default: python)
  -h, --help      Show this help
  --              All following arguments are forwarded to the Python script
EOF
}

# Defaults
START=19
END=50
GPU0=0
GPU1=1
CHANNEL="gpu02"
SCRIPT="main_v5.py"
PYTHON_BIN="python"
EXTRA_ARGS=()

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)   START="${2:?}"; shift 2 ;;
    --end)     END="${2:?}"; shift 2 ;;
    --gpu0)    GPU0="${2:?}"; shift 2 ;;
    --gpu1)    GPU1="${2:?}"; shift 2 ;;
    --channel) CHANNEL="${2:?}"; shift 2 ;;
    --script)  SCRIPT="${2:?}"; shift 2 ;;
    --python)  PYTHON_BIN="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "Unknown option: $1" 1>&2; usage; exit 1 ;;
  esac
done

# Ensure we run from this script's directory (expected: src/PressurePattern)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PIDFILE="$SCRIPT_DIR/run_seeds.pid"
# Prevent multiple orchestrators
if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && ps -p "$old_pid" >/dev/null 2>&1; then
    echo "[ERROR] Another run_seeds.sh is running with PID $old_pid (from $PIDFILE)." 1>&2
    echo "Stop it first: kill $old_pid" 1>&2
    exit 1
  fi
fi
echo "$$" > "$PIDFILE"

# Basic checks
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python binary '$PYTHON_BIN' not found in PATH." 1>&2
  exit 1
fi
if ! command -v notify-run >/dev/null 2>&1; then
  echo "[ERROR] notify-run not found in PATH." 1>&2
  exit 1
fi
if [[ ! -f "$SCRIPT" ]]; then
  echo "[ERROR] Target script '$SCRIPT' not found in $(pwd)" 1>&2
  exit 1
fi
if [[ "$START" -gt "$END" ]]; then
  echo "[ERROR] --start ($START) must be <= --end ($END)" 1>&2
  exit 1
fi

echo "[INFO] Working directory: $(pwd)"
echo "[INFO] Orchestrating $SCRIPT from seed $START to $END on GPUs $GPU0/$GPU1 via notify-run channel '$CHANNEL'"
if ((${#EXTRA_ARGS[@]} > 0)); then
  printf '[INFO] Extra args:'
  printf ' %q' "${EXTRA_ARGS[@]}"
  echo
fi

# Track active job PIDs to allow clean termination on signals
ACTIVE_PIDS=()
cleanup() {
  echo "[INFO] Received signal, terminating active jobs..."
  if ((${#ACTIVE_PIDS[@]} > 0)); then
    # Try to terminate children (python) of notify-run wrappers first
    for p in "${ACTIVE_PIDS[@]}"; do
      [[ -n "$p" ]] || continue
      pkill -TERM -P "$p" 2>/dev/null || true
      kill "$p" 2>/dev/null || true
    done
  fi
  wait 2>/dev/null || true
  rm -f "$PIDFILE" 2>/dev/null || true
  exit 130
}
trap 'cleanup' INT TERM

s="$START"
while [[ "$s" -le "$END" ]]; do
  s0="$s"
  s1=$((s + 1))
  echo "============================================================"
  echo "[INFO] Launching pair: seed $s0 -> GPU $GPU0, seed $s1 -> GPU $GPU1 (if <= $END)"
  ts="$(date '+%F %T')"
  pid0=''
  pid1=''

  # GPU0 job
  echo "[CMD ] notify-run $CHANNEL -- $PYTHON_BIN $SCRIPT --gpu $GPU0 --seed $s0 ${EXTRA_ARGS[*]} > seed${s0}.out 2>&1 &"
  notify-run "$CHANNEL" -- "$PYTHON_BIN" "$SCRIPT" --gpu "$GPU0" --seed "$s0" "${EXTRA_ARGS[@]}" > "seed${s0}.out" 2>&1 &
  pid0=$!

  # GPU1 job (only if within range)
  if [[ "$s1" -le "$END" ]]; then
    echo "[CMD ] notify-run $CHANNEL -- $PYTHON_BIN $SCRIPT --gpu $GPU1 --seed $s1 ${EXTRA_ARGS[*]} > seed${s1}.out 2>&1 &"
    notify-run "$CHANNEL" -- "$PYTHON_BIN" "$SCRIPT" --gpu "$GPU1" --seed "$s1" "${EXTRA_ARGS[@]}" > "seed${s1}.out" 2>&1 &
    pid1=$!
  else
    echo "[INFO] Skipping GPU $GPU1 because seed $s1 exceeds end $END"
  fi

  # Track current pair's PIDs for clean shutdown
  ACTIVE_PIDS=()
  [[ -n "$pid0" ]] && ACTIVE_PIDS+=("$pid0")
  [[ -n "$pid1" ]] && ACTIVE_PIDS+=("$pid1")
  echo "[INFO] Launched at $ts. Waiting for completion..."
  status0=0
  status1=0
  if [[ -n "$pid0" ]]; then
    if ! wait "$pid0"; then status0=$?; fi
  fi
  if [[ -n "$pid1" ]]; then
    if ! wait "$pid1"; then status1=$?; fi
  fi

  echo "[INFO] Pair finished: seed $s0 exit=$status0, seed $s1 exit=$status1"
  if [[ $status0 -ne 0 || $status1 -ne 0 ]]; then
    echo "[WARN] One or more jobs exited non-zero. Continuing to next pair..."
  fi

  ACTIVE_PIDS=()
  s=$((s + 2))
done

echo "============================================================"
echo "[INFO] All seeds completed from $START to $END."
rm -f "$PIDFILE" 2>/dev/null || true
exit 0
