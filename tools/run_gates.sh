#!/usr/bin/env bash
# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Local mirror of .github/workflows/ci.yml: every checker CI trusts, in one
# command, from inside the repo. Replaces the out-of-tree robo_file/_ops/
# gates.sh, which could not be cloned with the project (review item N).
#
# Two rules this script exists to enforce:
#
#  1. RC is read from the gate itself, never through a pipe. `cmd | tail` makes
#     `$?` report tail's status and masks a real failure -- that is exactly how
#     the old out-of-tree script reported a bogus GATES_EXIT.
#  2. Every real scan is preceded by its --self-test negative control. A checker
#     that cannot fail passes forever, so its green verdict is worth nothing.
#
# Usage:
#   bash tools/run_gates.sh
#   FLAGSCALE_ROOT=../FlagScale bash tools/run_gates.sh   # + closure scans
#
# The closure scans need the FlagScale baseline tree, which lives outside this
# repo (CI clones it to _ci/FlagScale). Without FLAGSCALE_ROOT they are reported
# as SKIP with the reason, never silently dropped.
#
# Exit status: 0 if every gate returned 0, else 1.
set -u
cd "$(dirname "$0")/.." || exit 90

PY=${PYTHON:-python3}
OUT=${GATES_OUT:-/tmp/robo_gates}
mkdir -p "$OUT"
FAILED=0
RAN=0
SKIPPED=0

# gate <label> <logstem> <cmd...>   RC captured directly, never through a pipe.
gate() {
  local label=$1 stem=$2
  shift 2
  "$@" > "$OUT/$stem.log" 2>&1
  local rc=$?
  RAN=$((RAN + 1))
  printf '%-46s RC=%d\n' "$label" "$rc"
  if [ "$rc" -ne 0 ]; then
    FAILED=$((FAILED + 1))
    echo "  --- tail $OUT/$stem.log ---"
    tail -12 "$OUT/$stem.log" | sed 's/^/  /'
  fi
}

skip() {
  SKIPPED=$((SKIPPED + 1))
  printf '%-46s SKIP  (%s)\n' "$1" "$2"
}

echo "=== negative controls: each checker must prove it CAN fail ==="
gate "check_no_megatron.py --self-test"    nc_megatron  "$PY" tools/check_no_megatron.py --self-test
gate "check_all_dangling.py --self-test"   nc_dangling  "$PY" tools/check_all_dangling.py --self-test
gate "check_symbol_refs.py --self-test"    nc_symbols   "$PY" tools/check_symbol_refs.py --self-test
gate "check_dep_drift.py --self-test"      nc_deps      "$PY" tools/check_dep_drift.py --self-test
gate "check_import_closure.py --self-test" nc_closure   "$PY" tools/check_import_closure.py --self-test
gate "compare_precision.py --self-test"    nc_precision "$PY" tools/compare_precision.py --self-test
gate "check_test_coverage.py --self-test"  nc_coverage  "$PY" tools/check_test_coverage.py --self-test
gate "check_conf_paths.py --self-test"     nc_conf      "$PY" tools/check_conf_paths.py --self-test
gate "check_leaf_import.py --self-test"    nc_leaf      "$PY" tools/check_leaf_import.py --self-test

echo
echo "=== purity: no Megatron in source, declared deps, or env ==="
gate "check_no_megatron.py"                g_megatron   "$PY" tools/check_no_megatron.py
gate "check_no_megatron.py --check-env"    g_megenv     "$PY" tools/check_no_megatron.py --check-env

echo
echo "=== structure: dangling refs, symbols, declared-vs-imported deps ==="
gate "check_all_dangling.py ."             g_dangling   "$PY" tools/check_all_dangling.py .
gate "check_symbol_refs.py ."              g_symbols    "$PY" tools/check_symbol_refs.py .
gate "check_dep_drift.py ."                g_deps       "$PY" tools/check_dep_drift.py .

echo
echo "=== coverage, conf paths, leaf imports ==="
gate "check_test_coverage.py ."            g_coverage   "$PY" tools/check_test_coverage.py .
# The conf files pin absolute paths that exist only in the training image, so a
# bare scan is red by construction off-image. The three documented overrides
# keep the environment-INDEPENDENT half of the gate (defaults compose, every
# ${...} resolves, no input key is empty) -- same narrowing as ci.yml.
gate "check_conf_paths.py examples/*/conf" g_conf \
  env FLAGOS_CHECKPOINT_DIR="${FLAGOS_CHECKPOINT_DIR:-$PWD}" \
      FLAGOS_TOKENIZER_PATH="${FLAGOS_TOKENIZER_PATH:-$PWD}" \
      FLAGOS_DATA_PATH="${FLAGOS_DATA_PATH:-$PWD}" \
      "$PY" tools/check_conf_paths.py examples/*/conf/*.yaml
gate "check_leaf_import.py ."              g_leaf       "$PY" tools/check_leaf_import.py .

echo
echo "=== migration closure (needs the FlagScale baseline tree) ==="
if [ -n "${FLAGSCALE_ROOT:-}" ] && [ -d "${FLAGSCALE_ROOT:-/nonexistent}" ]; then
  gate "closure: training entry points" g_cl_entry \
    "$PY" tools/check_import_closure.py \
      --source-root "$FLAGSCALE_ROOT" --package flagscale \
      --planned-from docs/migration_scope.txt \
      --discover-seeds \
      --examples-only pi0 --examples-only pi0_5 --examples-only gr00t_n1_5 \
      --cut flagscale.models.vla.qwen_gr00t --cut flagscale.models.vla.vlm
  gate "closure: launcher" g_cl_launcher \
    "$PY" tools/check_import_closure.py \
      --source-root "$FLAGSCALE_ROOT" --package flagscale \
      --planned-from docs/migration_scope.txt \
      --seed flagscale.runner.runner_train \
      --seed flagscale.runner.backend.backend_native_train \
      --examples-only pi0 --examples-only pi0_5 --examples-only gr00t_n1_5
else
  skip "closure: training entry points" "set FLAGSCALE_ROOT=<FlagScale checkout>"
  skip "closure: launcher"              "set FLAGSCALE_ROOT=<FlagScale checkout>"
fi

echo
echo "=== unit tests ==="
gate "pytest tests/unit_tests" g_pytest "$PY" -m pytest tests/unit_tests -q

echo
echo "ran=$RAN failed=$FAILED skipped=$SKIPPED  logs=$OUT"
if [ "$FAILED" -ne 0 ]; then
  echo "GATES: FAIL ($FAILED gate(s) returned non-zero)"
  exit 1
fi
echo "GATES: PASS"
exit 0
