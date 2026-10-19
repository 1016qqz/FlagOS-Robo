#!/usr/bin/env python3
"""Per-leaf import probe: every module under flagos_robo/ must be importable.

Why this gate exists
--------------------
check_import_closure.py walks package ENTRY POINTS (``__init__`` chains from a
seed). A leaf module that nobody imports is never exercised by it, so a sibling
module that was never migrated goes unnoticed. That is not hypothetical: the
W2-W3 migration shipped ``flagos_robo/runner/heartbeat/monitor.py`` whose line
29 does ``from .health_reader import ...`` while ``health_reader.py`` had not
been migrated. The closure gate was green, the unit tests were green, and the
module was simply dead on arrival -- because config.py reaches both modules by
STRING path, so seed discovery never walked into them.

This probe imports EVERY module individually and classifies each failure:

  MISSING-INTERNAL  ModuleNotFoundError naming a flagos_robo.* module
                    -> real defect: a module that was never migrated.
  MISSING-EXTERNAL  ModuleNotFoundError naming a third-party module
                    -> environment, not a defect (torch/transformers absent).
  OTHER             anything else (SyntaxError, side-effect crash at import).

Exit code is 1 when MISSING-INTERNAL or OTHER is non-empty. OTHER counts: a
module that cannot be parsed or blows up at import time is broken regardless of
which exception class carries the news. MISSING-EXTERNAL never fails the gate,
which is what lets this run in a dependency-free job.

Usage:
    python tools/check_leaf_import.py [repo_root]
    python tools/check_leaf_import.py --self-test
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

PKG = "flagos_robo"


def discover_modules(repo: Path) -> list[str]:
    """Dotted names of every .py under <repo>/flagos_robo, __init__ collapsed."""
    pkg_dir = repo / PKG
    mods: list[str] = []
    for p in sorted(pkg_dir.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        parts = list(p.relative_to(repo).parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][:-3]
        mods.append(".".join(parts))
    return mods


def probe(repo: Path) -> tuple[dict[str, list], list[str]]:
    """Import every module; return (buckets, all_module_names)."""
    mods = discover_modules(repo)
    sys.path.insert(0, str(repo))

    buckets: dict[str, list] = {"ok": [], "internal": [], "external": [], "other": []}
    for m in mods:
        try:
            importlib.import_module(m)
            buckets["ok"].append(m)
        except ModuleNotFoundError as e:
            name = getattr(e, "name", "") or ""
            key = "internal" if name.startswith(PKG) else "external"
            buckets[key].append((m, name))
        except BaseException:  # noqa: BLE001 - probe must survive anything
            tb = traceback.format_exc().strip().splitlines()[-1]
            buckets["other"].append((m, tb))
    return buckets, mods


def report(repo: Path, buckets: dict[str, list], mods: list[str]) -> int:
    print("=" * 74)
    print(f"FlagOS-Robo per-leaf import probe   repo={repo}")
    print(f"python={sys.version.split()[0]}   modules={len(mods)}")
    print("=" * 74)
    print(f"\nOK                 {len(buckets['ok'])}")
    print(f"MISSING-INTERNAL   {len(buckets['internal'])}   <-- real defects")
    print(f"MISSING-EXTERNAL   {len(buckets['external'])}   (environment)")
    print(f"OTHER              {len(buckets['other'])}   <-- real defects")

    print("\n[MISSING-INTERNAL] a flagos_robo.* module that does not exist")
    if not buckets["internal"]:
        print("    (none)")
    for m, name in buckets["internal"]:
        print(f"    {m}\n        -> needs {name}")

    print("\n[OTHER] non-ModuleNotFoundError failures at import time")
    if not buckets["other"]:
        print("    (none)")
    for m, tb in buckets["other"]:
        print(f"    {m}\n        {tb}")

    if buckets["external"]:
        by_dist: dict[str, int] = {}
        for _m, name in buckets["external"]:
            by_dist[name] = by_dist.get(name, 0) + 1
        print("\n[MISSING-EXTERNAL] grouped by absent third-party module")
        for name, cnt in sorted(by_dist.items(), key=lambda kv: -kv[1]):
            print(f"    {name:24s} blocks {cnt} module(s)")

    if not mods:
        # A zero-module scan prints "no defects" and would pass forever. That is
        # exactly the false-pass this gate is supposed to prevent elsewhere, so
        # refuse it here.
        print("\nFAIL: scanned 0 modules -- wrong repo root?", file=sys.stderr)
        return 2

    bad = len(buckets["internal"]) + len(buckets["other"])
    print(f"\n{'FAIL' if bad else 'OK'}: {bad} real defect(s)")
    return 1 if bad else 0


# --------------------------------------------------------------- self-test
# A checker that cannot fail passes forever. CI runs this FIRST as a negative
# control (see .github/workflows/ci.yml and tests/unit_tests/tools/
# test_ci_workflow.py::test_negative_control_steps_are_present).
#
# Each case builds a synthetic flagos_robo package in a temp dir and re-invokes
# this script as a SUBPROCESS, because importing modules mutates sys.modules of
# the running interpreter and cannot be cleanly undone in-process.

_CASES = [
    # (name, files, expected_rc, must_appear_in_stdout)
    (
        "L1 missing internal sibling is caught",
        {
            "__init__.py": "",
            "mon.py": "from .health_reader import HardwareHealthReader\n",
        },
        1,
        "flagos_robo.health_reader",
    ),
    (
        "L2 syntax error is caught (OTHER must affect exit code)",
        {"__init__.py": "", "broken.py": "def f(:\n"},
        1,
        "flagos_robo.broken",
    ),
    (
        "FP control: absent third-party must NOT fail the gate",
        {
            "__init__.py": "",
            "needs_torch.py": "import torch_not_installed_xyz\n",
        },
        0,
        "MISSING-EXTERNAL",
    ),
    (
        "FP control: a clean package must NOT fail the gate",
        {"__init__.py": "", "fine.py": "VALUE = 1\n"},
        0,
        "0 real defect(s)",
    ),
]


def run_self_test() -> int:
    failures: list[str] = []
    for title, files, want_rc, want_text in _CASES:
        tmp = Path(tempfile.mkdtemp(prefix="leafprobe-"))
        try:
            pkg = tmp / PKG
            pkg.mkdir()
            for rel, body in files.items():
                (pkg / rel).write_text(body, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(tmp)],
                capture_output=True,
                text=True,
            )
            out = proc.stdout + proc.stderr
            if proc.returncode != want_rc:
                failures.append(f"{title}: rc={proc.returncode} want {want_rc}")
            elif want_text not in out:
                failures.append(f"{title}: {want_text!r} absent from output")
            else:
                print(f"  ok   {title}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # A wrong root must be loud, not silently clean.
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "/nonexistent-root-xyz"],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        failures.append("wrong repo root exited 0 (false pass)")
    else:
        print("  ok   wrong repo root is rejected")

    print()
    if failures:
        print("SELF-TEST FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"SELF-TEST OK: {len(_CASES) + 1} case(s) passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default=".", help="repository root")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="prove the probe can fail: plant known defects and assert they are caught",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test()

    repo = Path(args.root).resolve()
    if not (repo / PKG).is_dir():
        print(f"ERROR: {repo / PKG} not found", file=sys.stderr)
        return 2
    buckets, mods = probe(repo)
    return report(repo, buckets, mods)


if __name__ == "__main__":
    sys.exit(main())
