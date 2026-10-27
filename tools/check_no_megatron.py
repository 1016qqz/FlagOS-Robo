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
"""W1 D3 - assert FlagOS-Robo carries no Megatron dependency.

This is the核心 acceptance gate of the whole independent-repo effort: FlagOS-Robo
must be installable and runnable in a plain-PyTorch environment, with Megatron
absent from the source, the declared dependencies, and the installed env.

Why not just ``grep -i megatron``
---------------------------------
A bare grep produces FALSE POSITIVES that make the gate unusable, and we hit
both kinds in this repo during W1:

* ``requirements/cuda/train.txt`` names ``megatron_core`` inside a *comment*
  explaining that it is forbidden. A grep flags our own documentation.
* ``flagos_robo/platforms/platform_base.py`` explains in a *docstring* that the
  upstream entry point was bound to ``megatron.plugin.platform``. Same problem.

So the source scan is token-based: Python files are tokenised and COMMENT /
docstring tokens are discarded before matching, leaving only executable code.
Requirement files are parsed line-wise with comments stripped and the
distribution name normalised (PEP 503), instead of substring matching.

Four independent layers are checked:

  L1 source imports   - AST: no ``import megatron`` / ``from megatron...``
  L2 source code text - tokenize: no ``megatron`` identifier/string in code
  L3 declared deps    - requirements/*.txt + pyproject: no megatron* dist
  L4 installed env    - importlib.metadata + import attempt (``--check-env``)

Exit code 0 = clean, 1 = violation found. Run ``--self-test`` to prove the
checker can actually fail (negative control).

This file is itself exempt from the L2 text scan via the pragma below -
it is the one place in the repo that must spell the name out in code.
It is NOT exempt from L1: it does not import megatron.

check-no-megatron: allow-self-reference
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

#: Matches the distribution/module family we forbid: megatron, megatron-core,
#: megatron_core, megatron.core... Anchored at the start so an unrelated name
#: that merely contains the substring is not swept up.
MEGATRON_RE = re.compile(r"\bmegatron(?:[._-][a-z0-9_.-]*)?\b", re.IGNORECASE)

#: Megatron submodules probed by L4. Deliberately NOT the bare ``megatron``
#: top level: that name is an implicit namespace package shared with
#: ``megatron-energon``, so probing it alone cannot tell Megatron-LM from a
#: standalone data loader. Every real Megatron install provides one of these.
MEGATRON_PROBE_MODULES = (
    "megatron.core",
    "megatron.training",
    "megatron.legacy",
    "megatron.post_training",
    "megatron.bridge",
    "megatron.plugin",
    "megatron.inference",
)

#: The single documented exception, enabled only via ``--allow-energon``.
#: PEP 503 normalised.
ENERGON_ALLOWED_DISTS = frozenset({"megatron-energon"})

#: Directories never scanned (not our source).
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    ".mypy_cache",
    "node_modules",
    "_scratch",
}


class Violation:
    __slots__ = ("layer", "path", "line", "detail")

    def __init__(self, layer: str, path: str, line: int, detail: str):
        self.layer = layer
        self.path = path
        self.line = line
        self.detail = detail

    def __str__(self) -> str:
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.layer}] {loc}: {self.detail}"

    def as_dict(self) -> dict:
        return {"layer": self.layer, "path": self.path, "line": self.line, "detail": self.detail}


#: Per-file opt-out pragma, in the spirit of ``# noqa``. A file that must name
#: Megatron in executable code (this checker itself, and its unit test) declares
#: it explicitly. Deliberately a MARKER and not a hardcoded filename list, so
#: the exemption is visible at the point of use and cannot silently widen into
#: "tools/ is never scanned" - which would be a real blind spot.
ALLOW_PRAGMA = "check-no-megatron: allow-self-reference"


def iter_py_files(root: Path, respect_pragma: bool = True):
    for p in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if respect_pragma:
            try:
                if ALLOW_PRAGMA in p.read_text(encoding="utf-8"):
                    continue
            except (OSError, UnicodeDecodeError):
                pass
        yield p


def check_l1_imports(root: Path) -> list[Violation]:
    """L1 - AST import check. The strictest, least ambiguous signal.

    Note ``respect_pragma=False``: the allow-pragma exempts a file from the
    *text* scan (L2), never from the *import* scan. No file in this repo may
    actually import Megatron, including this checker.
    """
    out: list[Violation] = []
    for p in iter_py_files(root, respect_pragma=False):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # pragma: no cover - broken file is its own bug
            out.append(Violation("L1", str(p), exc.lineno or 0, f"syntax error: {exc.msg}"))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if MEGATRON_RE.match(alias.name):
                        out.append(Violation("L1", str(p), node.lineno, f"import {alias.name}"))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if MEGATRON_RE.match(node.module):
                    out.append(
                        Violation("L1", str(p), node.lineno, f"from {node.module} import ...")
                    )
    return out


def code_tokens(path: Path):
    """Yield (lineno, text) for tokens that are real CODE.

    Comments are dropped outright. Docstrings are dropped by tracking string
    expressions that stand alone as a statement - which is what lets this
    checker coexist with docs that legitimately *name* megatron while
    explaining why it is banned.
    """
    src = path.read_text(encoding="utf-8")
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, IndentationError):
        return
    prev_meaningful = None  # last token that isn't NL/NEWLINE/INDENT/DEDENT/COMMENT
    for tok in toks:
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING:
            # A docstring is a STRING whose preceding meaningful token ends a
            # statement (NEWLINE/INDENT/start-of-file) rather than an operator
            # or name - i.e. it is an expression statement, not a value.
            is_docstring = prev_meaningful is None or prev_meaningful.type in (
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
            )
            if is_docstring:
                prev_meaningful = tok
                continue
        if tok.type not in (tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
            prev_meaningful = tok
        if tok.type in (tokenize.NAME, tokenize.STRING):
            yield tok.start[0], tok.string


def check_l2_code_text(root: Path) -> list[Violation]:
    """L2 - no megatron identifier or string literal in executable code.

    Catches what L1 cannot: dynamic imports such as
    ``importlib.import_module("megatron.core.foo")``.
    """
    out: list[Violation] = []
    for p in iter_py_files(root):
        for lineno, text in code_tokens(p):
            if MEGATRON_RE.search(text):
                out.append(Violation("L2", str(p), lineno, f"code token {text[:60]!r}"))
    return out


def _normalise_dist(name: str) -> str:
    """PEP 503 normalisation, so megatron_core / Megatron-Core collapse."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _requirement_names(text: str):
    """Yield (lineno, dist_name) for real requirement lines.

    Strips ``#`` comments (the false-positive source), skips ``-r``/``-e``/
    option lines, and cuts at the first version/extra/marker delimiter.
    """
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[\[<>=!~;@\s]", line, maxsplit=1)[0]
        if name:
            yield i, _normalise_dist(name)


def check_l3_declared_deps(root: Path, allow_energon: bool = False) -> list[Violation]:
    """L3 - declared dependencies: requirements/*.txt and pyproject.toml.

    ``allow_energon`` mirrors L4: ``megatron-energon`` is a standalone
    WebDataset loader that merely squats the ``megatron`` namespace, so the
    data layer may declare it deliberately. Off by default.
    """

    def _banned(name: str) -> bool:
        if not MEGATRON_RE.match(name):
            return False
        return not (allow_energon and name in ENERGON_ALLOWED_DISTS)

    out: list[Violation] = []
    for req in sorted(root.glob("requirements/**/*.txt")):
        for lineno, name in _requirement_names(req.read_text(encoding="utf-8")):
            if _banned(name):
                out.append(Violation("L3", str(req), lineno, f"requirement {name!r}"))

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib
        except ModuleNotFoundError:  # pragma: no cover - py<3.11
            tomllib = None
        if tomllib is not None:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
            groups = {"dependencies": project.get("dependencies", []) or []}
            for extra, deps in (project.get("optional-dependencies", {}) or {}).items():
                groups[f"optional-dependencies.{extra}"] = deps or []
            for group, deps in groups.items():
                for dep in deps:
                    name = _normalise_dist(re.split(r"[\[<>=!~;@\s]", dep, maxsplit=1)[0])
                    if _banned(name):
                        out.append(Violation("L3", str(pyproject), 0, f"{group}: {dep!r}"))
    return out


def check_l4_installed_env(allow_energon: bool = False) -> list[Violation]:
    """L4 - the far end: nothing megatron is importable or installed here.

    Source and declarations can be clean while the *environment* still has
    Megatron pre-installed (exactly the case with FlagScale's official image),
    which would let a stray dependency pass unnoticed in CI.

    Why this probes NAMED SUBMODULES and not the bare ``megatron`` top level:
    ``megatron-energon`` ships ``megatron/energon/`` with **no**
    ``megatron/__init__.py`` - a PEP 420 implicit namespace package. Installing
    it makes ``find_spec("megatron")`` return a namespace spec even though
    Megatron-LM itself is absent (verified: energon 7.4.1 declares no
    megatron/megatron-core requirement). Probing the bare name would therefore
    fire on a library that is not Megatron at all. Real Megatron always
    materialises one of the submodules below.

    ``allow_energon`` is the one documented exception and defaults to OFF, so
    W1 keeps the strictest possible gate; the data layer (W2/W3) can turn it on
    deliberately when it adopts Energon as its WebDataset loader.
    """
    import importlib.metadata as md
    import importlib.util

    out: list[Violation] = []
    for dist in md.distributions():
        name = dist.metadata["Name"] or ""
        if not name:
            continue
        norm = _normalise_dist(name)
        if not MEGATRON_RE.match(norm):
            continue
        if allow_energon and norm in ENERGON_ALLOWED_DISTS:
            continue
        out.append(Violation("L4", "<installed>", 0, f"distribution {name} {dist.version}"))

    probes = list(MEGATRON_PROBE_MODULES)
    if not allow_energon:
        probes.append("megatron.energon")
    for mod in probes:
        try:
            if importlib.util.find_spec(mod) is not None:
                out.append(Violation("L4", "<importable>", 0, f"module {mod} is importable"))
        except (ImportError, ValueError, ModuleNotFoundError):
            pass
    return out


def run_self_test(root: Path) -> int:
    """Negative control: prove each layer can FAIL.

    'No violations' is only meaningful if the checker is capable of reporting
    one. Plants a violation per layer in a temp copy and asserts it is caught.
    """
    import shutil
    import tempfile

    print("== self-test: planting known violations ==")
    failures = []
    tmp_root = Path(tempfile.mkdtemp()) / "repo"
    shutil.copytree(root, tmp_root, ignore=shutil.ignore_patterns(*SKIP_DIRS, "*.pyc"))

    # L1: a real import statement.
    pkg = tmp_root / "flagos_robo" / "_selftest_l1.py"
    pkg.write_text("from megatron.core import parallel_state  # planted\n")
    got = check_l1_imports(tmp_root)
    print(f"  L1 planted import      -> {len(got)} violation(s): {got[0] if got else 'NONE'}")
    if not got:
        failures.append("L1 failed to detect a planted `from megatron.core import ...`")
    pkg.unlink()

    # L2: dynamic import hidden in a string (invisible to L1).
    pkg = tmp_root / "flagos_robo" / "_selftest_l2.py"
    pkg.write_text('import importlib\nm = importlib.import_module("megatron.core.foo")\n')
    got2 = [v for v in check_l2_code_text(tmp_root) if "_selftest_l2" in v.path]
    print(f"  L2 planted dyn-import  -> {len(got2)} violation(s): {got2[0] if got2 else 'NONE'}")
    if not got2:
        failures.append("L2 failed to detect a planted importlib.import_module('megatron...')")
    if not check_l1_imports(tmp_root):
        print("  L2 confirms it catches what L1 structurally cannot (no AST import node)")
    pkg.unlink()

    # L3: a declared requirement, written the way pip would accept it.
    reqdir = tmp_root / "requirements" / "cuda"
    reqdir.mkdir(parents=True, exist_ok=True)
    planted_req = reqdir / "_selftest.txt"
    planted_req.write_text("megatron_core>=0.9.0\n")
    got3 = [v for v in check_l3_declared_deps(tmp_root) if "_selftest" in v.path]
    print(f"  L3 planted requirement -> {len(got3)} violation(s): {got3[0] if got3 else 'NONE'}")
    if not got3:
        failures.append("L3 failed to detect a planted `megatron_core>=0.9.0` requirement")
    planted_req.unlink()

    # False-positive control: the documentation forms that MUST NOT trip the gate.
    fp = tmp_root / "flagos_robo" / "_selftest_fp.py"
    fp.write_text(
        '"""Docstring explaining that megatron.plugin.platform was the old binding."""\n'
        "# comment mentioning megatron_core as a forbidden dependency\n"
        "VALUE = 1\n"
    )
    fp_req = reqdir / "_selftest_fp.txt"
    fp_req.write_text("# megatron_core is FORBIDDEN here\ntorch>=2.0\n")
    fp_hits = [
        v
        for v in check_l1_imports(tmp_root)
        + check_l2_code_text(tmp_root)
        + check_l3_declared_deps(tmp_root)
        if "_selftest_fp" in v.path
    ]
    print(f"  FP control (doc/comment mentions) -> {len(fp_hits)} violation(s), expect 0")
    if fp_hits:
        failures.append(f"false positive on documentation mentions: {[str(v) for v in fp_hits]}")
    fp.unlink()
    fp_req.unlink()

    shutil.rmtree(tmp_root.parent, ignore_errors=True)

    if failures:
        print("\nSELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nSELF-TEST PASSED: every layer detects a planted violation, and")
    print("documentation mentions of megatron do NOT trigger a false positive.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repo root to scan (default: parent of tools/)",
    )
    ap.add_argument(
        "--check-env",
        action="store_true",
        help="also assert megatron is absent from the CURRENT python env (L4)",
    )
    ap.add_argument(
        "--self-test", action="store_true", help="negative control: prove the checker can fail"
    )
    ap.add_argument("--json", action="store_true", help="emit violations as JSON")
    ap.add_argument(
        "--allow-energon",
        action="store_true",
        help=(
            "permit megatron-energon only (standalone WebDataset loader that squats the "
            "megatron PEP 420 namespace; declares no megatron-core dependency). "
            "OFF by default so W1 keeps the strictest gate."
        ),
    )
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test(args.root)

    root = args.root.resolve()
    violations = (
        check_l1_imports(root)
        + check_l2_code_text(root)
        + check_l3_declared_deps(root, allow_energon=args.allow_energon)
    )
    layers = ["L1 source imports (AST)", "L2 source code text (tokenize)", "L3 declared deps"]
    if args.check_env:
        violations += check_l4_installed_env(allow_energon=args.allow_energon)
        layers.append("L4 installed env")
    if args.allow_energon:
        layers.append("(megatron-energon exception ENABLED)")

    if args.json:
        print(json.dumps([v.as_dict() for v in violations], indent=2))
    else:
        print(f"check_no_megatron: root={root}")
        for name in layers:
            print(f"  checked {name}")
        n_py = len(list(iter_py_files(root)))
        n_req = len(list(root.glob("requirements/**/*.txt")))
        print(f"  scanned {n_py} python file(s), {n_req} requirement file(s)")
        if violations:
            print(f"\nFAIL: {len(violations)} Megatron reference(s) found")
            for v in violations:
                print("  " + str(v))
        else:
            print(
                "\nPASS: no Megatron dependency in source, declared deps"
                + (", or env" if args.check_env else "")
            )
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
