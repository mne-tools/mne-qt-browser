#!/usr/bin/env bash
# Loop a pytest invocation under lldb until it dies with a native signal, then dump
# the backtrace, registers, faulting instruction and the memory around the pen/state
# pointers. Written for the macOS CI SIGBUS in QCosmeticStroker::drawPath (see the
# "Known flake" paragraph in AGENTS.md). Run from the repo root on the runner, e.g.:
#
#   tools/lldb_loop.sh -x -m pgtest mne-python/mne/report mne-python/mne/viz
#   tools/lldb_loop.sh tests/test_pg_specific.py -k transparent
#
# MAX_ITER (default 50) bounds the loop; the lldb output of the crashing iteration is
# left in lldb_crash.log. MALLOC_HISTORY=1 enables MallocStackLogging so lldb's
# "memory history" can name who last freed an address, at the cost of a changed heap
# layout that may stop the bug from reproducing.
trap 'exit 130' INT
export PYTHONFAULTHANDLER=1 MNE_LOGGING_LEVEL=warning OMP_NUM_THREADS=1
if [[ "${MALLOC_HISTORY:-0}" == "1" ]]; then
  export MallocStackLogging=1
else
  unset MallocStackLogging
fi
PYTHON=$(which python)
for i in $(seq 1 "${MAX_ITER:-50}"); do
  echo "=== iteration $i ($(date +%H:%M:%S))"
  # stop-on-exec: the python launcher execs the real interpreter, which would end a batch session
  lldb --batch \
    -o "settings set target.process.stop-on-exec false" \
    -o run \
    -k "bt 40" \
    -k "register read" \
    -k "disassemble --pc --count 8" \
    -k "memory read --format x --size 8 --count 8 \$x0" \
    -k "memory history \$x0" \
    -k "memory read --format x --size 8 --count 4 \$x8" \
    -k "memory history \$x8" \
    -k "thread list" \
    -k "quit 1" \
    -- "$PYTHON" -m pytest -p no:cacheprovider "$@" > lldb_iter.log 2>&1
  if grep -q "stop reason" lldb_iter.log; then
    mv lldb_iter.log lldb_crash.log
    echo "=== crashed on iteration $i; details in lldb_crash.log"
    grep -A80 "stop reason" lldb_crash.log
    exit 0
  fi
  grep -q " passed" lldb_iter.log || { echo "=== non-crash failure"; tail -20 lldb_iter.log; exit 2; }
done
echo "=== no crash in ${MAX_ITER:-50} iterations"
