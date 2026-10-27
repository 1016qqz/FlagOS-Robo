#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ratchet the static test-reference map so coverage can only improve.

This is not line coverage. It answers a coarser question that matters on a
migration project: which source modules does the test suite look at *at all*?
A module no test mentions is where a break sits undetected -- that is how
heartbeat/monitor.py shipped with a missing health_reader import under a green
258-test run.

Why a ratchet and not a zero gate: 86 of 132 modules are unreferenced today,
24190 LOC. A zero gate would be red forever and get deleted within a week. So
the baseline is checked in next to this file and the gate fails only when the
numbers move the WRONG WAY -- blind LOC up, or referenced modules down. New
code may arrive unreferenced; it just cannot push the totals backwards, and
the only way to re-baseline is an explicit --update-baseline commit that shows
up in review.

Three reference dimensions, because this repo writes tests two different ways:

  1. import -- `from flagos_robo.x import y` / `import flagos_robo.x`
  2. dotted string -- "flagos_robo.x.y" in a monkeypatch target or importlib arg
  3. path parts -- tests that must NOT import the module (train_pi.py pulls in
     torch, transformers and flash_attn, none installable on a CPU box) read
     the source and ast.parse it instead:
         ENTRY_POINTS = ["train_gr00t_n1_5.py", "train_pi.py"]
         TRAIN_DIR = ... / "flagos_robo" / "train"
     Dimensions 1 and 2 both match dotted module paths, so they score those
     files as zero-reference even though 26 assertions bind them. Dimension 3
     requires EVERY path component of the module to appear as a string literal
     in the test file, which is strict enough that a bare "utils.py" cannot
     claim an arbitrary utils module.

Usage:
    python tools/check_test_coverage.py [repo_root]
    python tools/check_test_coverage.py --update-baseline
    python tools/check_test_coverage.py --self-test

Exit codes:
    0  coverage holds or improves against the baseline
    1  coverage regressed, or the baseline file is missing
    2  nothing was scanned, or the baseline is unreadable -- 0 would be a lie
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

PKG = "flagos_robo"

BASELINE_NAME = "test_coverage_baseline.json"

# A regression in either direction fails. Equality passes: new unreferenced
# code is allowed to land, it just may not push the totals backwards.
RATCHET_FIELDS = (
    ("blind_loc", "max", "LOC in modules no test mentions"),
    ("referenced_modules", "min", "modules at least one test mentions"),
)

DOTTED = re.compile(rf"\b{PKG}(?:\.[A-Za-z_][A-Za-z0-9_]*)+")

SKIP_DIRS = {"__pycache__", ".git", "build", "dist", "_ci"}


def _py_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return [p for p in sorted(root.rglob("*.py")) if not SKIP_DIRS.intersection(p.parts)]


def _loc(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
        return 0


def module_name(repo: Path, path: Path) -> str | None:
    try:
        rel = path.relative_to(repo)
    except ValueError:
        return None
    if not rel.parts or rel.parts[0] != PKG:
        return None
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts) if parts else None


def _string_literals(tree: ast.AST) -> set[str]:
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


# Calls that mean "this test actually ingests source text", as opposed to merely
# mentioning path components in passing. Without this filter dimension 3 is a
# pure name match: a test that lists ["flagos_robo", "models", "pi0",
# "modeling_pi0.py"] and asserts nothing would be credited with covering that
# module, inflating the ratchet baseline. Measured: such a probe moved
# referenced_modules 48 -> 49 and blind_loc 22514 -> 21247.
SOURCE_INGEST_CALLS = frozenset(
    {
        "read_text",
        "read_bytes",
        "open",
        "parse",  # ast.parse
        "getsource",
        "getsourcelines",
        "getsourcefile",
    }
)


def _ingests_source(tree: ast.AST) -> bool:
    """True if the test file reads or parses source text at all."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name in SOURCE_INGEST_CALLS:
            return True
    return False


def collect_references(repo: Path) -> tuple[set[str], set[frozenset[str]], int]:
    """Three reference dimensions over every test file.

    Returns (dotted module names, per-test sets of string literals, test count).
    The literal sets stay per-file: a module counts as path-referenced only if
    ONE test file names all of its path components, never a union across files.
    """
    dotted: set[str] = set()
    literal_sets: set[frozenset[str]] = set()
    files = _py_files(repo / "tests")

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        dotted.update(DOTTED.findall(text))
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == PKG or node.module.startswith(PKG + "."):
                    dotted.add(node.module)
                    for alias in node.names:
                        dotted.add(f"{node.module}.{alias.name}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == PKG or alias.name.startswith(PKG + "."):
                        dotted.add(alias.name)
        if _ingests_source(tree):
            literal_sets.add(frozenset(_string_literals(tree)))

    return dotted, literal_sets, len(files)


def is_referenced(
    repo: Path,
    path: Path,
    mod: str,
    dotted: set[str],
    literal_sets: set[frozenset[str]],
) -> bool:
    # dimensions 1 + 2: the module itself, or anything beneath it, named dotted
    if any(ref == mod or ref.startswith(mod + ".") for ref in dotted):
        return True

    # dimension 3: one test file names every path component of this module
    rel_parts = set(path.relative_to(repo).parts)
    return any(rel_parts <= literals for literals in literal_sets)


def measure(repo: Path) -> dict:
    """Current coverage numbers. Raises ValueError if there is nothing to scan."""
    sources = _py_files(repo / PKG)
    mods: dict[str, tuple[Path, int]] = {}
    for path in sources:
        mod = module_name(repo, path)
        if mod:
            mods[mod] = (path, _loc(path))

    dotted, literal_sets, n_tests = collect_references(repo)

    if not mods:
        raise ValueError(f"no source modules under {repo / PKG}")
    if n_tests == 0:
        raise ValueError(f"no test files under {repo / 'tests'}")

    blind: list[tuple[int, str]] = []
    referenced = 0
    for mod, (path, loc) in mods.items():
        if is_referenced(repo, path, mod, dotted, literal_sets):
            referenced += 1
        else:
            blind.append((loc, mod))

    return {
        "total_modules": len(mods),
        "total_loc": sum(loc for _p, loc in mods.values()),
        "referenced_modules": referenced,
        "blind_loc": sum(loc for loc, _m in blind),
        "test_files": n_tests,
        "_blind_detail": sorted(blind, reverse=True),
    }


def _baseline_path(repo: Path) -> Path:
    return repo / "tools" / BASELINE_NAME


def load_baseline(repo: Path) -> dict | None:
    path = _baseline_path(repo)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_baseline(repo: Path, current: dict) -> Path:
    payload = {k: v for k, v in current.items() if not k.startswith("_")}
    payload["_comment"] = (
        "High-water mark for tools/check_test_coverage.py. blind_loc may only "
        "fall and referenced_modules may only rise. Regenerate with "
        "--update-baseline, and say in the commit message why coverage moved."
    )
    path = _baseline_path(repo)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def report(repo: Path, quiet: bool = False) -> int:
    try:
        current = measure(repo)
    except ValueError as exc:
        print(f"ERROR: {exc} -- nothing to check, so RC=0 would be meaningless", file=sys.stderr)
        return 2

    baseline = None
    try:
        baseline = load_baseline(repo)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: {_baseline_path(repo)} is unreadable: {exc}", file=sys.stderr)
        return 2

    pct = 100 * (current["total_loc"] - current["blind_loc"]) // max(current["total_loc"], 1)

    if not quiet:
        print(
            f"scanned {current['total_modules']} source modules "
            f"({current['total_loc']} LOC) against {current['test_files']} test files"
        )
        print(
            f"referenced: {current['referenced_modules']}/{current['total_modules']} modules, "
            f"blind: {current['blind_loc']} LOC ({100 - pct}% of the tree)"
        )

    if baseline is None:
        print(
            f"\nFAIL: no baseline at {_baseline_path(repo)}.\n"
            f"      Run --update-baseline once to pin today's numbers, and commit it.",
            file=sys.stderr,
        )
        return 1

    regressions: list[str] = []
    if not quiet:
        print("\n[ratchet] current vs baseline")
    for field, direction, label in RATCHET_FIELDS:
        now = current[field]
        was = baseline.get(field)
        if was is None:
            regressions.append(f"baseline has no {field!r}; re-run --update-baseline")
            continue
        bad = now > was if direction == "max" else now < was
        delta = now - was
        verdict = "REGRESSED" if bad else ("improved" if delta else "held")
        arrow = f"{was} -> {now}" if delta else f"{now}"
        if not quiet:
            print(f"    {'FAIL' if bad else 'ok  '}  {label:<42} {arrow:>16}  {verdict}")
        if bad:
            worse = "grew" if direction == "max" else "shrank"
            regressions.append(f"{label} {worse} from {was} to {now}")

    if not quiet and current["_blind_detail"]:
        print("\n[largest blind modules] no test mentions these at all")
        for loc, mod in current["_blind_detail"][:10]:
            print(f"    {loc:6d} LOC  {mod}")
        print(f"    -- {len(current['_blind_detail'])} modules with zero references")

    if regressions:
        print("\nFAIL: test coverage moved backwards", file=sys.stderr)
        for item in regressions:
            print(f"  - {item}", file=sys.stderr)
        print(
            "\n  Either add a test that mentions the module, or -- if the drop is "
            "deliberate\n  (code deleted, module merged) -- re-pin with "
            "--update-baseline in the same commit.",
            file=sys.stderr,
        )
        return 1

    if not quiet:
        print("\nCLEAN: coverage holds or improves against the baseline.")
    return 0


# A test that reads the source instead of importing it. This is the shape that
# binds train_pi.py today, and the shape a dotted-path-only scanner scores as
# zero-reference. It must count, or the ratchet pins a number that is wrong.
_TEST_PATH_PARTS = """\
import ast
from pathlib import Path

ENTRY_POINTS = ["_selftest_cov_target.py"]
TARGET_DIR = Path(__file__).resolve().parents[1] / "flagos_robo"


def test_target_defines_something():
    for entry in ENTRY_POINTS:
        tree = ast.parse((TARGET_DIR / entry).read_text(encoding="utf-8"))
        assert any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
"""

_TEST_IMPORT = """\
from flagos_robo._selftest_cov_target import thing


def test_thing():
    assert thing() == 1
"""

_TEST_UNRELATED = """\
def test_nothing_about_the_target():
    assert True
"""

# Names every path component of the target but never reads it. Dimension 3 is a
# set-subset match on string literals, so without the SOURCE_INGEST_CALLS filter
# this scores as full coverage and the ratchet pins an inflated number. Keep this
# red: it is the measured false positive, not a hypothetical one.
_TEST_PARTS_WITHOUT_READING = """\
PARTS = ["flagos_robo", "_selftest_cov_target.py"]


def test_only_names_the_parts():
    assert PARTS
"""

_SOURCE_TARGET = """\
def thing():
    return 1
"""

# Zero LOC on purpose. Dropping its only reference moves referenced_modules
# without moving blind_loc, which is the one input that isolates the
# min-direction half of the ratchet from the max-direction half.
_SOURCE_EMPTY = ""


def _write(repo: Path, rel: str, text: str) -> Path:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _case(repo: Path, source: str | None, test: str | None) -> int:
    """Add a source module and/or a test, measure, then clean up."""
    made: list[Path] = []
    try:
        if source is not None:
            made.append(_write(repo, f"{PKG}/_selftest_cov_target.py", source))
        if test is not None:
            made.append(_write(repo, "tests/unit_tests/test_selftest_cov.py", test))
        return report(repo, quiet=True)
    finally:
        for path in made:
            path.unlink(missing_ok=True)


def run_self_test(repo: Path) -> int:
    print("self-test: ratchet behavior on synthetic cases")
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td) / "repo"
        shutil.copytree(
            repo,
            sandbox,
            ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.egg-info"),
        )
        write_baseline(sandbox, measure(sandbox))

        base = report(sandbox, quiet=True)
        print(f"  [base ] unmodified tree, baseline just pinned  RC={base} (expect 0)")
        if base != 0:
            failures.append("the tree is not clean against its own fresh baseline")

        greens = [
            ("new module with an importing test", _SOURCE_TARGET, _TEST_IMPORT),
            ("new module with a source-reading test", _SOURCE_TARGET, _TEST_PATH_PARTS),
            ("new test alone, no new source", None, _TEST_UNRELATED),
        ]
        for label, source, test in greens:
            rc = _case(sandbox, source, test)
            ok = rc == 0
            print(f"  [green] {label:<42} RC={rc} (expect 0) {'ok' if ok else 'MISFIRE'}")
            if not ok:
                failures.append(f"false positive: {label}")

        rc = _case(sandbox, _SOURCE_TARGET, None)
        ok = rc == 1
        print(
            f"  [red  ] {'new module no test mentions':<42} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}"
        )
        if not ok:
            failures.append("missed a real regression: blind LOC grew")

        rc = _case(sandbox, _SOURCE_TARGET, _TEST_UNRELATED)
        ok = rc == 1
        print(
            f"  [red  ] {'new module, test ignores it':<42} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}"
        )
        if not ok:
            failures.append("missed a real regression: a test that adds no reference")

        rc = _case(sandbox, _SOURCE_TARGET, _TEST_PARTS_WITHOUT_READING)
        ok = rc == 1
        print(
            f"  [red  ] {'test names path parts but never reads':<42} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}"
        )
        if not ok:
            failures.append(
                "dimension 3 credited a test that only names path components; "
                "the SOURCE_INGEST_CALLS filter is not working"
            )

        # Pin a baseline that already counts an empty module as referenced, then
        # delete only its test. blind_loc cannot move (the module is 0 LOC), so
        # this fires only if the min-direction half of the ratchet works. Note a
        # real test file is NOT usable here: the two train entry points are each
        # referenced by two test files, so removing one changes nothing.
        src = _write(sandbox, f"{PKG}/_selftest_cov_target.py", _SOURCE_EMPTY)
        test = _write(sandbox, "tests/unit_tests/test_selftest_cov.py", _TEST_IMPORT)
        saved_baseline = _baseline_path(sandbox).read_text(encoding="utf-8")
        write_baseline(sandbox, measure(sandbox))
        test.unlink()
        rc = report(sandbox, quiet=True)
        src.unlink(missing_ok=True)
        _baseline_path(sandbox).write_text(saved_baseline, encoding="utf-8")
        ok = rc == 1
        print(
            f"  [red  ] {'deleting a test drops a reference':<42} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}"
        )
        if not ok:
            failures.append("missed a real regression: referenced_modules fell")

        moved = _baseline_path(sandbox)
        saved = moved.read_text(encoding="utf-8")
        moved.unlink()
        rc = report(sandbox, quiet=True)
        moved.write_text(saved, encoding="utf-8")
        ok = rc == 1
        print(
            f"  [red  ] {'baseline file missing':<42} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}"
        )
        if not ok:
            failures.append("a missing baseline passed; the gate would be vacuous")

        after = report(sandbox, quiet=True)
        print(f"  [back ] tree restored after mutations         RC={after} (expect 0)")
        if after != 0:
            failures.append("a mutation leaked; the sandbox did not return to clean")

        empty = Path(td) / "empty"
        (empty / PKG).mkdir(parents=True)
        (empty / "tests").mkdir(parents=True)
        rc = report(empty, quiet=True)
        ok = rc == 2
        print(
            f"  [rc2  ] {'empty tree scans nothing':<42} RC={rc} (expect 2) {'ok' if ok else 'BAD'}"
        )
        if not ok:
            failures.append("an empty tree did not produce RC=2; RC=0 could mean no work")

    if failures:
        print("\nself-test FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("\nself-test OK: 3 correct patterns stay green, 5 regressions go red, empty tree RC=2.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ratchet the static test-reference map; fail only on regression.",
    )
    parser.add_argument("root", nargs="?", default=".", help="repo root (default: cwd)")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="re-pin the high-water mark to today's numbers",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify the gate itself: correct code green, regressions red",
    )
    parser.add_argument("--quiet", action="store_true", help="exit code only")
    args = parser.parse_args(argv)

    repo = Path(args.root).resolve()
    if not (repo / PKG).is_dir():
        print(f"ERROR: {repo / PKG} not found", file=sys.stderr)
        return 2

    if args.self_test:
        return run_self_test(repo)

    if args.update_baseline:
        try:
            current = measure(repo)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        path = write_baseline(repo, current)
        print(f"baseline pinned at {path}")
        print(f"  referenced_modules = {current['referenced_modules']}/{current['total_modules']}")
        print(f"  blind_loc          = {current['blind_loc']}/{current['total_loc']}")
        return 0

    return report(repo, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
