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
"""W1 D4 - decide the FlagOS-Robo migration scope with a three-channel scan.

The question this tool answers
------------------------------
"If we copy module set *S* out of FlagScale into FlagOS-Robo, does *S* still
reach anything that was left behind?"  If yes, the migration will break at
import time (or worse, at runtime), so the scope is not closed yet.

Why three channels and not just an import closure
-------------------------------------------------
A pure AST import closure is necessary but **not sufficient**.  During W1 we
hit all three of its structural blind spots in this very repository, so the
scan is deliberately triangulated:

  C1 static  - AST transitive closure over ``import`` / ``from ... import``.
               Catches the ordinary case.
               BLIND to: anything not spelled as an import statement.

  C2 dynamic - ``importlib.import_module`` / ``__import__`` / dotted class
               paths that live in *strings*.  These resolve at run time, so a
               rename keeps importing fine and only explodes when the code path
               is finally executed.  Real hit: ``train/processor/pipeline.py``
               resolves ``step_entry["class"]`` from YAML.
               BLIND to: strings that are assembled, not literal.

  C3 text    - ``.yaml`` / ``.md`` / ``.sh`` / ``.toml`` references such as
               ``entrypoint: flagscale/train/train_pi.py``.  grep over ``*.py``
               cannot see these at all, yet they are what the launcher reads.

Seeds must be DISCOVERED, not hardcoded
---------------------------------------
The second blind spot bit us specifically because we seeded the closure from
the three *training* entry points only.  ``examples/*/conf/*.yaml`` also
declares ``inference`` and ``serve`` entry points, which drag in a few hundred
further lines.  So ``--discover-seeds`` parses every ``entrypoint:`` in the
example configs and uses the union.  Hand-passed ``--seed`` is additive.

Self-validation is mandatory
----------------------------
``OUTSIDE = 0`` is only meaningful if the tool is *able* to report a non-zero
number.  ``--self-test`` builds a synthetic package tree in a temp dir and
asserts, for each of the three channels, that a planted leak is detected and
that removing a known-required module flips OUTSIDE from 0 to non-zero.  The
self-test needs no FlagScale checkout, so it runs anywhere - including the
plain-PyTorch acceptance container.

Usage
-----
  # real scan against a FlagScale checkout
  python tools/check_import_closure.py --source-root /path/to/FlagScale \\
      --package flagscale --planned-from docs/migration_scope.txt --discover-seeds

  # negative control / CI smoke
  python tools/check_import_closure.py --self-test

Exit status: 0 = scope closed, 1 = leaks found, 2 = usage/input error.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# channel 2/3 patterns
# ---------------------------------------------------------------------------

#: calls whose argument is a module path resolved at run time
DYNAMIC_CALLS = ("import_module", "__import__", "load_module", "find_spec")

#: text files that can carry a module/script reference the launcher obeys
TEXT_SUFFIXES = (".yaml", ".yml", ".md", ".sh", ".toml", ".cfg", ".json")

#: files skipped everywhere (vendored / generated / scratch)
SKIP_DIR_NAMES = {
    ".git",
    ".github",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "_scratch",
    ".venv",
    "venv",
    "node_modules",
    "third_party",
}


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


# ---------------------------------------------------------------------------
# module <-> file mapping
# ---------------------------------------------------------------------------


def module_name_for(path: Path, source_root: Path) -> str:
    """``<root>/flagscale/train/train_pi.py`` -> ``flagscale.train.train_pi``."""
    rel = path.relative_to(source_root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def resolve_candidates(module: str, source_root: Path) -> list[Path]:
    """Every on-disk file a dotted module name could denote."""
    rel = Path(*module.split("."))
    return [source_root / f"{rel}.py", source_root / rel / "__init__.py"]


def module_exists(module: str, source_root: Path) -> bool:
    return any(p.is_file() for p in resolve_candidates(module, source_root))


def owning_module(module: str, source_root: Path) -> str | None:
    """Walk up the dotted name until a real module file is found.

    ``flagscale.models.utils.constants.ACTION_DIM`` is an *attribute* import;
    the module that must be migrated is ``flagscale.models.utils.constants``.
    """
    parts = module.split(".")
    while parts:
        cand = ".".join(parts)
        if module_exists(cand, source_root):
            return cand
        parts.pop()
    return None


# ---------------------------------------------------------------------------
# channel 1 - AST transitive import closure
# ---------------------------------------------------------------------------


def imports_of(path: Path, self_module: str, package: str) -> set[str]:
    """In-package modules imported by ``path``, relative imports resolved."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    except SyntaxError:
        return set()

    pkg_parts = self_module.split(".")
    # for a package __init__ the module itself is the anchor, else its parent
    anchor = pkg_parts if (path.name == "__init__.py") else pkg_parts[:-1]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(package + "."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = anchor[: len(anchor) - node.level + 1]
                mod = ".".join([*base, node.module]) if node.module else ".".join(base)
            else:
                mod = node.module or ""
            if not (mod == package or mod.startswith(package + ".")):
                continue
            found.add(mod)
            # ``from pkg.sub import name`` - name may itself be a submodule
            for alias in node.names:
                if alias.name != "*":
                    found.add(f"{mod}.{alias.name}")
    return found


@dataclass
class ClosureResult:
    reached: set[str] = field(default_factory=set)
    unresolved: set[str] = field(default_factory=set)
    edges: dict[str, set[str]] = field(default_factory=dict)
    #: relative paths of every file the closure actually opened. This is what
    #: lets channels 2 and 3 ask "is this reference coming FROM code we carry
    #: over?" instead of reporting every ref in the monorepo.
    files: set[str] = field(default_factory=set)


def static_closure(seeds: list[str], source_root: Path, package: str) -> ClosureResult:
    """Breadth-first transitive import closure starting from ``seeds``."""
    res = ClosureResult()
    queue = list(seeds)
    while queue:
        module = queue.pop()
        owner = owning_module(module, source_root)
        if owner is None:
            res.unresolved.add(module)
            continue
        if owner in res.reached:
            continue
        res.reached.add(owner)
        for path in resolve_candidates(owner, source_root):
            if not path.is_file():
                continue
            res.files.add(path.relative_to(source_root).as_posix())
            for imported in imports_of(path, owner, package):
                res.edges.setdefault(owner, set()).add(imported)
                target = owning_module(imported, source_root)
                if target is None:
                    res.unresolved.add(imported)
                elif target not in res.reached:
                    queue.append(target)
    return res


# ---------------------------------------------------------------------------
# channel 2 - dynamic / string module references
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    channel: str
    file: str
    line: int
    text: str
    module: str | None = None

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "file": self.file,
            "line": self.line,
            "text": self.text,
            "module": self.module,
        }


def _dotted_to_module(value: str, package: str) -> str | None:
    """Pull an in-package module path out of a string literal.

    Accepts both dotted (``flagscale.train.x.Cls``) and slash
    (``flagscale/train/x.py``) spellings, since configs use the latter.
    """
    cand = value.strip()
    if cand.endswith(".py"):
        cand = cand[: -len(".py")].replace("/", ".").replace("\\", ".")
    cand = cand.strip(".")
    if cand == package or cand.startswith(package + "."):
        return cand
    return None


def scan_dynamic(source_root: Path, package: str) -> list[Reference]:
    """String literals naming an in-package module, plus importlib call sites."""
    refs: list[Reference] = []
    for path in sorted(source_root.rglob("*.py")):
        if should_skip(path):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError:
            continue
        rel = str(path.relative_to(source_root))

        for node in ast.walk(tree):
            # (a) literal strings that look like an in-package module path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                mod = _dotted_to_module(node.value, package)
                if mod:
                    refs.append(Reference("dynamic-string", rel, node.lineno, node.value, mod))
            # (b) importlib-style calls whose argument is NOT a literal
            elif isinstance(node, ast.Call):
                fn = node.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name not in DYNAMIC_CALLS:
                    continue
                literal = node.args and isinstance(node.args[0], ast.Constant)
                if not literal:
                    refs.append(
                        Reference(
                            "dynamic-call",
                            rel,
                            node.lineno,
                            f"{name}(<runtime value>)",
                            None,
                        )
                    )
    return refs


# ---------------------------------------------------------------------------
# channel 3 - text references (yaml / md / sh / toml)
# ---------------------------------------------------------------------------


def scan_text(source_root: Path, package: str, subdirs: list[str] | None = None) -> list[Reference]:
    """References to the package inside non-Python files.

    These are invisible to any ``*.py`` scan but are exactly what the launcher
    reads: ``entrypoint: flagscale/train/train_pi.py``.
    """
    roots = [source_root / s for s in subdirs] if subdirs else [source_root]
    pattern = re.compile(rf"{re.escape(package)}[./][\w./]*")
    refs: list[Reference] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or should_skip(path) or path.suffix not in TEXT_SUFFIXES:
                continue
            rel = str(path.relative_to(source_root))
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                for match in pattern.finditer(line):
                    refs.append(
                        Reference(
                            "text",
                            rel,
                            lineno,
                            line.strip()[:160],
                            _dotted_to_module(match.group(0), package),
                        )
                    )
    return refs


# ---------------------------------------------------------------------------
# seed discovery
# ---------------------------------------------------------------------------

ENTRYPOINT_RE = re.compile(r"^\s*entrypoint\s*:\s*(\S+)")


def discover_seeds(
    source_root: Path,
    package: str,
    examples: str = "examples",
    only_dirs: list[str] | None = None,
) -> list[str]:
    """Union of every ``entrypoint:`` declared under ``examples/*/conf``.

    Seeding from the training entry points alone was the W1 mistake: it hid the
    inference and serve closures entirely.

    ``only_dirs`` restricts discovery to specific example dirs. FlagScale ships
    28 entry points across all of ``examples/`` (LLM, VLM, compress, megatron
    GPT ...), of which FlagOS-Robo cares about three robotics suites. Without
    the filter the closure pulls in the entire monorepo and the scan is useless.
    """
    found: set[str] = set()
    root = source_root / examples
    if not root.is_dir():
        return []
    search_roots = [root / d for d in only_dirs] if only_dirs else [root]
    for base in search_roots:
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.y*ml")):
            if should_skip(path):
                continue
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                m = ENTRYPOINT_RE.match(line)
                if not m:
                    continue
                mod = _dotted_to_module(m.group(1).strip("\"'"), package)
                if mod:
                    found.add(mod)
    return sorted(found)


def vacuity_warnings(planned: list[str], package: str) -> list[str]:
    """Reject planned sets that make the scan tautological.

    A bare ``package`` subtree entry means "everything is in scope", so
    OUTSIDE is 0 by construction. That is not a result - it is the tool
    reporting success while deciding nothing. Same for a one-dot prefix like
    ``flagscale.train``, which swallows ``train.megatron`` wholesale.
    """
    problems = []
    for p in planned:
        if p.startswith("="):
            continue
        if p == package:
            problems.append(
                f"{p!r} is a subtree entry for the package root: it makes every "
                f"module 'planned' and OUTSIDE is 0 by construction. "
                f"Use '={package}' to migrate only the root __init__.py."
            )
    return problems


def load_planned(path: Path) -> list[str]:
    """Planned migration set: one module prefix per line, ``#`` comments ok."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def in_planned(module: str, planned: list[str]) -> bool:
    """Is ``module`` inside the planned migration set?

    Two entry forms, and the distinction is load-bearing:

    ``pkg.sub``   subtree - matches ``pkg.sub`` and anything below it.
                  Matching is on dotted boundaries, so ``a.b`` never matches
                  ``a.bc``.
    ``=pkg``      EXACT - matches only ``pkg`` itself, nothing below.

    The exact form exists because package roots must be migrated (their
    ``__init__.py`` is on the import chain) **without** dragging in every
    sibling subpackage. Listing a bare ``flagscale`` as a subtree silently
    makes the whole scan vacuous: every module is "planned", OUTSIDE is
    always 0, and the tool reports success while deciding nothing. We hit
    exactly that on the first real scan of FlagScale f8e9a62c.
    """
    for p in planned:
        if p.startswith("="):
            if module == p[1:]:
                return True
        elif module == p or module.startswith(p + "."):
            return True
    return False


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------


@dataclass
class ScanReport:
    seeds: list[str]
    planned: list[str]
    reached: list[str]
    outside: list[str]
    unresolved: list[str]
    dynamic: list[Reference]
    text: list[Reference]
    origins: set[str] = field(default_factory=set)
    cut: list[str] = field(default_factory=list)
    cut_reached: list[str] = field(default_factory=list)

    def _from_migrated_file(self, ref: Reference) -> bool:
        """Does this reference ORIGINATE inside the migration set?

        Load-bearing filter. A dynamic/text reference only threatens the
        migration if the file CONTAINING it is one we carry over. FlagScale's
        monorepo is full of refs between parts we never touch (runner_serve ->
        run_fs_serve_vllm.py, heartbeat, eval); reporting those buries the few
        real findings under ~117 irrelevant hits, which is how a checker stops
        being read. ``origins`` is the set of relative file paths reached by
        the static closure, plus the example configs that seed it.
        """
        return ref.file in self.origins

    @property
    def dynamic_outside(self) -> list[Reference]:
        """C2: a migrated file dynamically names a module we are NOT taking."""
        return [
            r
            for r in self.dynamic
            if r.module and self._from_migrated_file(r) and not in_planned(r.module, self.planned)
        ]

    @property
    def dynamic_opaque(self) -> list[Reference]:
        """C2b: runtime-resolved target inside a migrated file - manual review."""
        return [
            r for r in self.dynamic if r.channel == "dynamic-call" and self._from_migrated_file(r)
        ]

    @property
    def text_outside(self) -> list[Reference]:
        """C3: a migrated config/doc points at a module outside the set.

        Config files are not reached by the static closure, so they are
        always treated as in-scope origins (that is the whole point of C3 -
        ``entrypoint:`` paths are invisible to any import-level analysis).
        """
        return [
            r for r in self.text if r.module is not None and not in_planned(r.module, self.planned)
        ]

    @property
    def ok(self) -> bool:
        return not (self.outside or self.dynamic_outside or self.text_outside)

    def as_dict(self) -> dict:
        return {
            "seeds": self.seeds,
            "planned": self.planned,
            "reached_count": len(self.reached),
            "outside": self.outside,
            "unresolved": self.unresolved,
            "cut": self.cut,
            "cut_reached": self.cut_reached,
            "dynamic_outside": [r.as_dict() for r in self.dynamic_outside],
            "dynamic_opaque": [r.as_dict() for r in self.dynamic_opaque],
            "text_outside": [r.as_dict() for r in self.text_outside],
            "origins_count": len(self.origins),
            "ok": self.ok,
        }


def scan(
    source_root: Path,
    package: str,
    planned: list[str],
    seeds: list[str],
    examples: str = "examples",
    text_subdirs: list[str] | None = None,
    cut: list[str] | None = None,
) -> ScanReport:
    closure = static_closure(seeds, source_root, package)
    cut = list(cut or [])
    not_planned = [m for m in closure.reached if not in_planned(m, planned)]
    cut_reached = sorted(m for m in not_planned if in_planned(m, cut))
    outside = sorted(m for m in not_planned if not in_planned(m, cut))
    return ScanReport(
        seeds=sorted(seeds),
        planned=list(planned),
        reached=sorted(closure.reached),
        outside=outside,
        unresolved=sorted(closure.unresolved),
        dynamic=scan_dynamic(source_root, package),
        text=scan_text(source_root, package, text_subdirs or [examples]),
        origins=set(closure.files),
        cut=cut,
        cut_reached=cut_reached,
    )


def stale_cut_entries(cut: list[str], cut_reached: list[str]) -> list[str]:
    """``--cut`` prefixes that this scan never actually reached.

    A cut entry is a claim: "the source tree really does import this, and the
    rewrite step deletes that import". If the scan does not reach the prefix,
    the claim is already stale - the import was removed, renamed, or never
    existed - and keeping the entry would silently absorb a future leak under
    the same name. So a stale entry is an error, not a warning: it is the same
    failure as the vacuous planned set, one level down. Without this check
    ``--cut`` degrades into a suppression list that drives OUTSIDE to 0 by
    construction, which is exactly what the planned-set vacuity guard exists
    to prevent.
    """
    return [p for p in cut if not any(in_planned(m, [p]) for m in cut_reached)]


def dead_planned_entries(planned: list[str], source_root: Path, package: str) -> list[str]:
    """Planned entries that name a module which does not exist in the source tree.

    The closure scan only ever asks "is every module I REACHED inside the
    planned set?". It is structurally blind in the other direction: a planned
    entry naming a module that does not exist is never reached, so it is never
    compared against anything, and the scan still reports PASS. That blind spot
    is not hypothetical - ``flagscale.runner.backend_native_train`` sat in the
    scope file through all of W1 while the real module is
    ``flagscale.runner.backend.backend_native_train``. Three green gates never
    saw it, because nothing in the closure direction can see it.

    So check the planned set against the filesystem directly. Entries outside
    ``package`` are skipped (they are deliberate renames/merges, not modules of
    the source tree).
    """
    dead = []
    for entry in planned:
        module = entry.lstrip("=")
        if module != package and not module.startswith(package + "."):
            continue
        rel = module.split(".")
        path = source_root.joinpath(*rel)
        if path.is_dir() or path.with_suffix(".py").is_file():
            continue
        dead.append(entry)
    return dead


def render(report: ScanReport, verbose: bool = False) -> None:
    print(f"seeds   : {len(report.seeds)}")
    for s in report.seeds:
        print(f"    + {s}")
    print(f"planned : {len(report.planned)} prefix(es)")
    print(f"C1 static closure : reached {len(report.reached)} module(s) in package")
    if verbose:
        for m in report.reached:
            mark = " " if in_planned(m, report.planned) else "!"
            print(f"    {mark} {m}")

    if report.outside:
        print(f"  !! OUTSIDE planned set: {len(report.outside)}")
        for m in report.outside:
            print(f"     - {m}")
    else:
        print("  -> OUTSIDE planned set: 0")

    if report.cut:
        print(
            f"  xx declared CUT (reached in source, import deleted on rewrite): "
            f"{len(report.cut_reached)} module(s) via {len(report.cut)} prefix(es)"
        )
        for m in report.cut_reached:
            print(f"     x {m}")
        print(
            "     NOTE: a cut is only PROVEN by re-running this tool against the "
            "migrated tree, where these modules do not exist at all."
        )

    if report.unresolved:
        print(f"  .. unresolved (not on disk, likely attribute import): {len(report.unresolved)}")
        if verbose:
            for m in report.unresolved:
                print(f"     ? {m}")

    print(f"C2 dynamic refs   : {len(report.dynamic)} total")
    if report.dynamic_outside:
        print(f"  !! string module refs outside planned set: {len(report.dynamic_outside)}")
        for r in report.dynamic_outside:
            print(f"     - {r.file}:{r.line}  {r.text}")
    else:
        print("  -> string module refs outside planned set: 0")
    if report.dynamic_opaque:
        print(
            f"  .. runtime-resolved call sites needing manual review: {len(report.dynamic_opaque)}"
        )
        for r in report.dynamic_opaque:
            print(f"     ? {r.file}:{r.line}  {r.text}")

    print(f"C3 text refs      : {len(report.text)} total")
    if report.text_outside:
        print(f"  !! text refs to rewrite: {len(report.text_outside)}")
        for r in report.text_outside[:40]:
            print(f"     - {r.file}:{r.line}  {r.text}")
        if len(report.text_outside) > 40:
            print(f"     ... and {len(report.text_outside) - 40} more")
    else:
        print("  -> text refs to rewrite: 0")

    print()
    if report.ok:
        print("PASS: migration scope is closed (C1/C2/C3 all clean)")
    else:
        print("FAIL: migration scope is NOT closed - see leaks above")


# ---------------------------------------------------------------------------
# self-test: synthetic fixture, no FlagScale checkout needed
# ---------------------------------------------------------------------------

#: a miniature package reproducing all three channels.
#: ``demo.keep.*`` is the planned migration set; ``demo.left_behind.*`` is not.
FIXTURE: dict[str, str] = {
    "demo/__init__.py": "",
    "demo/keep/__init__.py": "",
    # entry point; imports a sibling and a leaf, both planned
    "demo/keep/entry.py": (
        "from demo.keep import helper\n"
        "from demo.keep.leaf import CONST\n"
        "def main():\n"
        "    return helper.go(CONST)\n"
    ),
    # relative import must resolve to demo.keep.leaf, not demo.leaf
    "demo/keep/helper.py": "from .leaf import CONST\ndef go(x):\n    return x\n",
    "demo/keep/leaf.py": "CONST = 1\n",
    # present on disk but outside the planned set - the leak targets
    "demo/left_behind/__init__.py": "",
    "demo/left_behind/strays.py": "VALUE = 2\n",
    "demo/left_behind/dyn_target.py": "class Thing:\n    pass\n",
}

FIXTURE_PLANNED = ["demo.keep"]
FIXTURE_SEEDS = ["demo.keep.entry"]

#: planted leaks, one per channel
LEAK_C1 = ("demo/keep/entry.py", "from demo.left_behind.strays import VALUE\n")
LEAK_C2 = ("demo/keep/entry.py", 'CLASS_PATH = "demo.left_behind.dyn_target.Thing"\n')
LEAK_C3 = ("examples/demo/conf/train.yaml", "  entrypoint: demo/left_behind/strays.py\n")
#: an in-set text ref that must NOT be reported - guards against a blanket
#: "any mention of the package" text match
TEXT_CLEAN = ("examples/demo/conf/ok.yaml", "  entrypoint: demo/keep/entry.py\n")


def _write_fixture(root: Path, extra: dict[str, str] | None = None) -> None:
    files = dict(FIXTURE)
    files[TEXT_CLEAN[0]] = TEXT_CLEAN[1]
    for rel, content in (extra or {}).items():
        files[rel] = files.get(rel, "") + content
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _scan_fixture(
    root: Path,
    planned: list[str],
    seeds: list[str] | None = None,
    cut: list[str] | None = None,
) -> ScanReport:
    return scan(
        root,
        "demo",
        planned,
        seeds if seeds is not None else FIXTURE_SEEDS,
        examples="examples",
        cut=cut,
    )


def run_self_test(verbose: bool = False) -> int:
    """Prove each channel detects a planted leak, and that clean stays clean."""
    checks: list[tuple[str, bool, str]] = []
    tmp = Path(tempfile.mkdtemp(prefix="closure_selftest_"))
    try:
        # --- control: pristine fixture must be fully closed -----------------
        base = tmp / "base"
        _write_fixture(base)
        rep = _scan_fixture(base, FIXTURE_PLANNED)
        checks.append(
            (
                "control: pristine fixture is closed",
                rep.ok and not rep.outside and not rep.text_outside,
                f"outside={rep.outside} dyn={[r.text for r in rep.dynamic_outside]} "
                f"text={[r.text for r in rep.text_outside]}",
            )
        )
        checks.append(
            (
                "control: relative import resolved inside subpackage",
                "demo.keep.leaf" in rep.reached,
                f"reached={rep.reached}",
            )
        )
        checks.append(
            (
                "control: in-set text ref (demo/keep/entry.py) not reported",
                not rep.text_outside,
                f"text_outside={[r.text for r in rep.text_outside]}",
            )
        )

        # --- C1: planted static import leak ---------------------------------
        c1 = tmp / "c1"
        _write_fixture(c1, {LEAK_C1[0]: LEAK_C1[1]})
        rep1 = _scan_fixture(c1, FIXTURE_PLANNED)
        checks.append(
            (
                "C1 detects planted static import leak",
                "demo.left_behind.strays" in rep1.outside and not rep1.ok,
                f"outside={rep1.outside}",
            )
        )

        # --- C2: planted string module path ---------------------------------
        c2 = tmp / "c2"
        _write_fixture(c2, {LEAK_C2[0]: LEAK_C2[1]})
        rep2 = _scan_fixture(c2, FIXTURE_PLANNED)
        hit2 = [r.module for r in rep2.dynamic_outside]
        checks.append(
            (
                "C2 detects planted dynamic string module path",
                "demo.left_behind.dyn_target.Thing" in hit2 and not rep2.ok,
                f"dynamic_outside={hit2}",
            )
        )
        checks.append(
            (
                "C2 leak is INVISIBLE to C1 (proves C1 alone is insufficient)",
                not rep2.outside,
                f"C1 outside={rep2.outside}",
            )
        )

        # --- C3: planted yaml entrypoint ------------------------------------
        c3 = tmp / "c3"
        _write_fixture(c3, {LEAK_C3[0]: LEAK_C3[1]})
        rep3 = _scan_fixture(c3, FIXTURE_PLANNED)
        hit3 = [r.text for r in rep3.text_outside]
        checks.append(
            (
                "C3 detects planted yaml entrypoint leak",
                any("left_behind" in t for t in hit3) and not rep3.ok,
                f"text_outside={hit3}",
            )
        )
        checks.append(
            (
                "C3 leak is INVISIBLE to C1+C2 (proves text channel is needed)",
                not rep3.outside and not rep3.dynamic_outside,
                f"C1={rep3.outside} C2={[r.module for r in rep3.dynamic_outside]}",
            )
        )

        # --- removal control: OUTSIDE must flip 0 -> non-zero ---------------
        # Drop demo.keep.helper from the planned set. It is genuinely reached,
        # so a working tool MUST now report it. If OUTSIDE stays 0 here, the
        # tool is spinning and every previous "0" was meaningless.
        rep4 = _scan_fixture(base, ["demo.keep.entry", "demo.keep.leaf"])
        checks.append(
            (
                "removal control: dropping a required module flips OUTSIDE 0 -> n",
                "demo.keep.helper" in rep4.outside,
                f"outside={rep4.outside}",
            )
        )

        # --- seed discovery -------------------------------------------------
        seeds = discover_seeds(base, "demo")
        checks.append(
            (
                "seed discovery reads examples/*/conf/*.yaml entrypoints",
                seeds == ["demo.keep.entry"],
                f"seeds={seeds}",
            )
        )
        seeds3 = discover_seeds(c3, "demo")
        checks.append(
            (
                "seed discovery picks up the EXTRA entrypoint (B2 blind spot)",
                "demo.left_behind.strays" in seeds3,
                f"seeds={seeds3}",
            )
        )
        # seeding from the discovered union makes the C3 leak visible to C1 too
        rep5 = _scan_fixture(c3, FIXTURE_PLANNED, seeds=seeds3)
        checks.append(
            (
                "discovered seeds turn the hidden entrypoint into a C1 leak",
                "demo.left_behind.strays" in rep5.outside,
                f"outside={rep5.outside}",
            )
        )

        # --- --cut: deliberate non-migration, and its anti-rot guards -------
        # Reuse the C1 fixture: demo.left_behind.strays IS reached and is NOT
        # planned, i.e. exactly the shape of a module we decide to drop.
        rep_cut = _scan_fixture(c1, FIXTURE_PLANNED, cut=["demo.left_behind"])
        checks.append(
            (
                "cut: a declared non-migrated module leaves OUTSIDE and lands in CUT",
                rep_cut.ok
                and not rep_cut.outside
                and "demo.left_behind.strays" in rep_cut.cut_reached,
                f"outside={rep_cut.outside} cut_reached={rep_cut.cut_reached}",
            )
        )
        checks.append(
            (
                "cut: dropping the cut entry flips the same scan back to FAIL",
                not rep1.ok and "demo.left_behind.strays" in rep1.outside,
                f"without cut: ok={rep1.ok} outside={rep1.outside}",
            )
        )
        checks.append(
            (
                "cut: a prefix this scan never reached is reported STALE",
                stale_cut_entries(["demo.never_imported"], rep_cut.cut_reached)
                == ["demo.never_imported"],
                f"stale={stale_cut_entries(['demo.never_imported'], rep_cut.cut_reached)}",
            )
        )
        checks.append(
            (
                "cut: cannot shadow a PLANNED module (would be stale, never silent)",
                stale_cut_entries(["demo.keep.helper"], rep_cut.cut_reached)
                == ["demo.keep.helper"],
                "a planned module is never in cut_reached, so cutting it errors out",
            )
        )
        checks.append(
            (
                "cut: does not touch the C2 channel",
                bool(_scan_fixture(c2, FIXTURE_PLANNED, cut=["demo.left_behind"]).dynamic_outside),
                "a dynamic string ref into a cut subtree must still be reported",
            )
        )

        # --- prefix boundary ------------------------------------------------
        checks.append(
            (
                "planned prefix matches on dotted boundary only",
                in_planned("demo.keep.leaf", ["demo.keep"])
                and not in_planned("demo.keeper", ["demo.keep"]),
                "in_planned('demo.keeper', ['demo.keep']) must be False",
            )
        )

        # --- dead planned entry (the W1 blind spot) -------------------------
        # A planned entry naming a nonexistent module is never reached, so it
        # is never compared, so every closure gate reports PASS. That is how
        # `flagscale.runner.backend_native_train` (real path has an extra
        # `.backend.`) survived all of W1. Check both directions here: the
        # typo must be caught, and a live entry must not be flagged.
        dead_typo = dead_planned_entries([*FIXTURE_PLANNED, "demo.keep.no_such_module"], c1, "demo")
        checks.append(
            (
                "planned entry naming a nonexistent module is reported DEAD",
                dead_typo == ["demo.keep.no_such_module"],
                f"dead={dead_typo}",
            )
        )
        checks.append(
            (
                "dead-entry check does not flag entries that do exist",
                dead_planned_entries(FIXTURE_PLANNED, c1, "demo") == [],
                f"dead={dead_planned_entries(FIXTURE_PLANNED, c1, 'demo')}",
            )
        )
        checks.append(
            (
                "dead entry is invisible to the closure channel (why it is needed)",
                not any(
                    "no_such_module" in m
                    for m in _scan_fixture(
                        c1, [*FIXTURE_PLANNED, "demo.keep.no_such_module"]
                    ).reached
                ),
                "unreached entry never enters reached/outside, so C1 cannot see it",
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(name) for name, _, _ in checks)
    failed = 0
    for name, passed, detail in checks:
        status = "ok  " if passed else "FAIL"
        print(f"[{status}] {name.ljust(width)}")
        if verbose or not passed:
            print(f"         {detail}")
        failed += 0 if passed else 1

    print()
    if failed:
        print(f"SELF-TEST FAILED: {failed}/{len(checks)} check(s) failed")
        return 1
    print(
        f"SELF-TEST PASSED: {len(checks)}/{len(checks)} checks.\n"
        "Each of C1/C2/C3 detects a planted leak, each leak is invisible to the\n"
        "other channels, dropping a required module flips OUTSIDE 0 -> n, and\n"
        "seed discovery finds the non-training entrypoint."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source-root", type=Path, help="FlagScale checkout root")
    ap.add_argument("--package", default="flagscale", help="top-level package name")
    ap.add_argument(
        "--planned-from",
        type=Path,
        help="file listing planned module prefixes, one per line",
    )
    ap.add_argument(
        "--planned",
        action="append",
        default=[],
        help="planned module prefix (repeatable); merged with --planned-from",
    )
    ap.add_argument("--seed", action="append", default=[], help="seed module (repeatable)")
    ap.add_argument(
        "--cut",
        action="append",
        default=[],
        help=(
            "module prefix deliberately NOT migrated (repeatable). Reported "
            "separately from OUTSIDE instead of failing the scan. Every prefix "
            "must actually be reached by this scan or the tool exits 2."
        ),
    )
    ap.add_argument(
        "--discover-seeds",
        action="store_true",
        help="union in every 'entrypoint:' under examples/*/conf/*.yaml",
    )
    ap.add_argument("--examples", default="examples", help="examples dir name")
    ap.add_argument(
        "--examples-only",
        action="append",
        default=[],
        help="restrict --discover-seeds to these examples subdirs (repeatable)",
    )
    ap.add_argument(
        "--text-subdir",
        action="append",
        default=[],
        help="subdir to scan for text refs (default: the examples dir)",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="negative control on a synthetic fixture; needs no checkout",
    )
    ap.add_argument("--json", action="store_true", help="emit the report as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test(verbose=args.verbose)

    if not args.source_root:
        ap.error("--source-root is required unless --self-test is given")
    source_root: Path = args.source_root.expanduser().resolve()
    if not source_root.is_dir():
        print(f"error: source root does not exist: {source_root}", file=sys.stderr)
        return 2
    if not (source_root / args.package).is_dir():
        print(
            f"error: package {args.package!r} not found under {source_root}",
            file=sys.stderr,
        )
        return 2

    planned = list(args.planned)
    if args.planned_from:
        if not args.planned_from.is_file():
            print(f"error: no such planned file: {args.planned_from}", file=sys.stderr)
            return 2
        planned += load_planned(args.planned_from)
    if not planned:
        print("error: no planned module prefixes given", file=sys.stderr)
        return 2

    vacuous = vacuity_warnings(planned, args.package)
    if vacuous:
        print("error: planned set would make this scan vacuous:", file=sys.stderr)
        for msg in vacuous:
            print(f"  - {msg}", file=sys.stderr)
        return 2

    dead = dead_planned_entries(planned, source_root, args.package)
    if dead:
        print(
            "error: planned entry(ies) name a module that does not exist in "
            f"{source_root} (typo or stale path):",
            file=sys.stderr,
        )
        for entry in dead:
            print(f"  - {entry}", file=sys.stderr)
        print(
            "  The closure direction cannot catch this: an entry that is never "
            "reached is never compared, so the scan would still report PASS.",
            file=sys.stderr,
        )
        return 2

    seeds = list(args.seed)
    if args.discover_seeds:
        discovered = discover_seeds(
            source_root, args.package, args.examples, args.examples_only or None
        )
        if not discovered:
            print(
                f"error: --discover-seeds found no 'entrypoint:' under "
                f"{source_root / args.examples}",
                file=sys.stderr,
            )
            return 2
        seeds += discovered
    if not seeds:
        print("error: no seeds given (use --seed or --discover-seeds)", file=sys.stderr)
        return 2

    text_subdirs = list(args.text_subdir)
    if not text_subdirs and args.examples_only:
        # Keep channels aligned: if seed discovery was narrowed to specific
        # example suites, the text channel must scan the SAME suites. Letting
        # C3 roam the whole examples/ tree while C1 is scoped to three suites
        # produced 65 irrelevant "refs to rewrite" on the first real scan.
        text_subdirs = [f"{args.examples}/{d}" for d in args.examples_only]

    if not args.json:
        # --json must emit ONLY json on stdout, or every downstream consumer
        # (CI step, jq, pytest) has to strip a banner line first.
        print(f"check_import_closure: root={source_root} package={args.package}")
    report = scan(
        source_root,
        args.package,
        planned,
        sorted(set(seeds)),
        examples=args.examples,
        text_subdirs=text_subdirs or None,
        cut=args.cut,
    )
    stale = stale_cut_entries(report.cut, report.cut_reached)
    if stale:
        print(
            "error: --cut prefixes that this scan never reached (stale claims):",
            file=sys.stderr,
        )
        for p in stale:
            print(f"  - {p}", file=sys.stderr)
        print(
            "  A cut entry asserts the source tree imports it. If nothing "
            "reaches it, drop the entry - keeping it would absorb a future leak.",
            file=sys.stderr,
        )
        return 2
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, ensure_ascii=False))
    else:
        render(report, verbose=args.verbose)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
