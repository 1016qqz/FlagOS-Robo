#!/usr/bin/env bash
# Acceptance groups 3-6 from the W1 plan's section 5, with two corrections that
# the first verbatim run forced (see docs/W1_ACCEPTANCE.md):
#   - the closure flag is --planned-from, not --scope (plan's §5 wrote no flags
#     at all; the SOP and CI commands were already correct)
#   - `grep ... || echo OK` conflates "nothing matched" (rc=1) with "target does
#     not exist" (rc=2). Every text scan below asserts its target exists first
#     and reports rc=2 as ERROR, never as OK.
#
# No `set -e`: a non-zero RC is a RESULT to record.
# Log lives OUTSIDE the rsync'd tree (sync.sh uses --delete).
cd /workspace/FlagOS-Robo || exit 1
LOG=/tmp/acc_verbatim.log
: >"$LOG"
exec >>"$LOG" 2>&1

run() {  # run <label> <expected-rc-or-ANY> <command...>
  local label="$1"; shift
  local expect="$1"; shift
  echo "---------- [$label] expect_rc=$expect"
  echo "\$ $*"
  "$@"
  local rc=$?
  echo "RC[$label]=$rc"
  if [ "$expect" != "ANY" ] && [ "$rc" != "$expect" ]; then
    echo "MISMATCH[$label]: expected $expect got $rc"
  fi
  echo
}

# grep gate that cannot pass vacuously: rc=1 is PASS, rc=0 is FAIL (found),
# rc>=2 is ERROR (did not actually scan).
scan() {  # scan <label> <pattern> <paths...>
  local label="$1"; shift
  local pat="$1"; shift
  echo "---------- [$label] pattern=$pat targets=$*"
  local missing=0 t
  for t in "$@"; do
    [ -e "$t" ] || { echo "TARGET_MISSING: $t"; missing=1; }
  done
  if [ "$missing" = 1 ]; then
    echo "RC[$label]=SKIP_NOT_IN_W1_SCOPE  (not a pass: nothing was scanned)"
    echo; return
  fi
  grep -rnE "$pat" "$@"
  local rc=$?
  case $rc in
    1) echo "RC[$label]=0  PASS: no match, and the targets really were scanned";;
    0) echo "RC[$label]=1  FAIL: matches found above"; echo "MISMATCH[$label]: stale references present";;
    *) echo "RC[$label]=$rc  ERROR: grep could not scan"; echo "MISMATCH[$label]: scan error rc=$rc";;
  esac
  echo
}

echo "=== acc_verbatim start $(date -u +%FT%TZ)"
echo "python: $(which python)  $(python -V 2>&1)"
echo "repo:   $(pwd)"

echo "##### GROUP 3 -- dependency tree has no Megatron"
run  g3_checker 0 python tools/check_no_megatron.py
run  g3_piplist ANY bash -c 'pip list 2>/dev/null | grep -i megatron; echo "pip_grep_rc=$?"'
scan g3_srcgrep 'megatron' flagos_robo/

echo "##### GROUP 4 -- import closure is self-contained (flags per docs/MIGRATION_SOP.md)"
run g4_closure 0 python tools/check_import_closure.py \
  --source-root /workspace/flagscale_ref/FlagScale --package flagscale \
  --planned-from docs/migration_scope.txt \
  --discover-seeds \
  --examples-only pi0 --examples-only pi0_5 --examples-only gr00t_n1_5
run g4_selftest 0 python tools/check_import_closure.py --self-test

echo "##### GROUP 5 -- dynamic import / stale text scan"
scan g5_modpaths '["'"'"']flagscale\.' flagos_robo/
scan g5_examples 'flagscale' examples/
echo "---------- [g5_docs] documentation mentions of flagscale are LEGITIMATE"
echo "(migration provenance). Counted, not forbidden -- reviewed in W1_ACCEPTANCE.md:"
grep -rc 'flagscale' docs/*.md docs/*.txt 2>/dev/null | grep -v ':0$'
echo

echo "##### GROUP 6 -- unit tests"
run g6_pytest 0 python -m pytest tests/ -q

echo "##### EXTRA -- the other two tool self-tests"
run x_nomega_selftest 0 python tools/check_no_megatron.py --self-test
run x_precision_selftest 0 python tools/compare_precision.py --self-test

echo "=== acc_verbatim done $(date -u +%FT%TZ)"
echo "=== MISMATCH count: $(grep -c MISMATCH "$LOG")"
