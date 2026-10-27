#!/usr/bin/env python3
"""Gate: dependency declarations must agree with the code that imports them.

Three declaration sources can drift apart independently -- pyproject's
[project.dependencies], the requirements/*.txt tree, and the actual imports in
flagos_robo/. This gate pins the three relationships that are genuinely
*wrong* when broken, and reports the rest without failing.

Verdict classes, chosen so the gate can stay at RC=0 on a correct tree:

  D1  DRIFT / CONFLICT          hard, must be zero.
      A pyproject dependency that no requirements file declares, or a pinned
      version below pyproject's own floor. Both are unambiguous contradictions
      between two files that are supposed to say the same thing.

  D2  UNDECLARED IMPORT         allowlisted.
      A third-party module imported by the package but declared nowhere. Vendor
      accelerator runtimes legitimately live here: they ship inside the vendor
      image and must NOT be pip-installed. Anything not in the allowlist is a
      real missing declaration.

  D3  OPTIONAL BUT UNGUARDED    hard, must be zero.
      A module declared only in optional.txt yet imported with no availability
      check -- an optional dependency that is actually required, so the package
      breaks on a minimal install.

  REPORT  declared-but-unimported. Transitive and runtime-only dependencies
      make this permanently non-empty, so it informs and never gates.

Guardedness is deliberately broader than try/except ImportError. This repo also
probes with importlib.util.find_spec() before importing (see
flagos_robo/train/datasets/video_utils.py), which protects a minimal install
exactly as well. A gate that only recognized try/except would flag that file as
a defect -- a false positive on correct code, which gets gates deleted.

Exit codes: 0 clean, 1 defects found, 2 nothing scanned (an empty scan must
never look like success).
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
import tempfile
from pathlib import Path

# Modules imported by the package but intentionally absent from every
# requirements file, with the reason. Adding an entry is a deliberate act.
ALLOWED_UNDECLARED_IMPORTS: dict[str, str] = {
    "torch_npu": "Ascend runtime, ships inside the vendor image; pip install would break it",
    "torch_musa": "Moore Threads runtime, ships inside the vendor image",
    "torch_txda": "TXDA runtime, ships inside the vendor image",
    "flag_gems": "optional kernel library, installed from source per platform",
}

# import name -> distribution name, where the two differ.
IMPORT_TO_DIST: dict[str, str] = {
    "PIL": "pillow",
    "av": "av",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "huggingface_hub": "huggingface-hub",
    "typing_extensions": "typing-extensions",
    "tree": "dm-tree",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "msgpack_numpy": "msgpack-numpy",
    "torch_npu": "torch-npu",
    "torch_musa": "torch-musa",
    "torch_txda": "torch-txda",
    "flag_gems": "flag-gems",
    "pytorch_lightning": "pytorch-lightning",
    "dateutil": "python-dateutil",
}

REQUIREMENT_FILES = [
    "requirements/common.txt",
    "requirements/cuda/base.txt",
    "requirements/cuda/train.txt",
    "requirements/cuda/serve.txt",
    "requirements/optional.txt",
    "requirements/dev.txt",
]

# Everything a default install pulls in; optional/dev are opt-in by definition.
RUNTIME_FILES = REQUIREMENT_FILES[:4]

_IMPORT_ERROR_NAMES = {"ImportError", "ModuleNotFoundError", "Exception"}


def norm(name: str) -> str:
    """PEP 503 normalization: the only correct way to compare distribution names."""
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_for(module: str) -> str:
    return norm(IMPORT_TO_DIST.get(module, module))


def stdlib_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()))
    names |= {
        "__future__",
        "dataclasses",
        "typing",
        "pathlib",
        "collections",
        "concurrent",
        "importlib",
        "multiprocessing",
        "unittest",
        "logging",
        "asyncio",
        "email",
        "http",
        "json",
        "sqlite3",
        "xml",
        "ctypes",
    }
    return names


def _handles_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        exc = handler.type
        if exc is None:
            return True
        if isinstance(exc, ast.Name) and exc.id in _IMPORT_ERROR_NAMES:
            return True
        if isinstance(exc, ast.Tuple):
            if any(isinstance(e, ast.Name) and e.id in _IMPORT_ERROR_NAMES for e in exc.elts):
                return True
    return False


def _mentions_find_spec(node: ast.AST) -> bool:
    """True if this expression probes module availability.

    Recognizes find_spec(...) however it is spelled -- importlib.util.find_spec,
    a bare find_spec after a from-import, or an `is not None` comparison around
    either. This is the second guard idiom in this repo, and treating it as
    unguarded would flag correct code.
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Attribute) and fn.attr == "find_spec":
                return True
            if isinstance(fn, ast.Name) and fn.id == "find_spec":
                return True
    return False


def guarded_statement_ids(tree: ast.AST) -> set[int]:
    """ids of nodes whose import is protected against the module being absent.

    Two idioms, both of which keep a minimal install working:
      * inside `try:` whose handler catches ImportError/ModuleNotFoundError
      * inside `if find_spec("x"):` (either branch -- the else branch raises or
        falls back, which is the whole point of probing)
    """
    guarded: set[int] = set()

    def mark(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            for sub in ast.walk(stmt):
                guarded.add(id(sub))

    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and _handles_import_error(node):
            mark(node.body)
        elif isinstance(node, ast.If) and _mentions_find_spec(node.test):
            mark(node.body)
            mark(node.orelse)
    return guarded


def top_level_imports(path: Path) -> set[tuple[str, bool]]:
    """{(top_level_module, is_guarded)} for one file; relative imports skipped."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set()

    guarded = guarded_statement_ids(tree)
    out: set[tuple[str, bool]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add((alias.name.split(".")[0], id(node) in guarded))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # relative -> internal
            if node.module:
                out.add((node.module.split(".")[0], id(node) in guarded))
    return out


def parse_requirements(path: Path, seen: set[Path] | None = None) -> dict[str, str]:
    """{normalized_dist: raw_line}, following `-r` includes."""
    seen = seen if seen is not None else set()
    path = path.resolve()
    if path in seen or not path.is_file():
        return {}
    seen.add(path)
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r ", "--requirement ")):
            out.update(parse_requirements(path.parent / line.split(None, 1)[1].strip(), seen))
            continue
        if line.startswith("-"):
            continue  # --extra-index-url and friends
        m = re.match(r"^([A-Za-z0-9._-]+)", line)
        if m:
            out[norm(m.group(1))] = line
    return out


def parse_pyproject_deps(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    m = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
    if m:
        for item in re.findall(r'"([^"]+)"', m.group(1)):
            name = re.match(r"^([A-Za-z0-9._-]+)", item)
            if name:
                out[norm(name.group(1))] = item
    return out


def _version_tuple(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in v.split("."):
        digits = re.match(r"^(\d+)", chunk)
        parts.append(int(digits.group(1)) if digits else 0)
    return tuple(parts)


def scan(repo: Path) -> dict:
    """Collect every fact the verdicts need, with no judgement applied yet."""
    pkg = repo / "flagos_robo"
    std = stdlib_names()
    imports: dict[str, dict] = {}
    nfiles = 0

    if pkg.is_dir():
        for path in sorted(pkg.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            nfiles += 1
            for mod, is_guarded in top_level_imports(path):
                if not mod or mod in std or mod == "flagos_robo":
                    continue
                entry = imports.setdefault(mod, {"guarded_only": True, "sites": []})
                entry["sites"].append(str(path.relative_to(repo)))
                if not is_guarded:
                    entry["guarded_only"] = False

    req = {rel: parse_requirements(repo / rel) for rel in REQUIREMENT_FILES}
    declared_all: dict[str, str] = {}
    for rel in REQUIREMENT_FILES:
        for dist, line in req[rel].items():
            declared_all.setdefault(dist, f"{rel}: {line}")
    runtime: dict[str, str] = {}
    for rel in RUNTIME_FILES:
        for dist, line in req[rel].items():
            runtime.setdefault(dist, f"{rel}: {line}")

    pyproject = parse_pyproject_deps(repo / "pyproject.toml")

    # D1: a pyproject dependency missing everywhere, or pinned below its floor.
    drift: list[str] = []
    conflict: list[str] = []
    for dist, spec in sorted(pyproject.items()):
        where = declared_all.get(dist)
        if where is None:
            drift.append(f"{dist}: pyproject={spec!r} but no requirements file declares it")
            continue
        floor = re.search(r">=\s*([0-9][0-9A-Za-z.\-]*)", spec)
        exact = re.search(r"==\s*([0-9][0-9A-Za-z.\-]*)", where.split(": ", 1)[1])
        if floor and exact and _version_tuple(exact.group(1)) < _version_tuple(floor.group(1)):
            conflict.append(
                f"{dist}: pinned {exact.group(1)} < pyproject floor {floor.group(1)}  ({where})"
            )

    # D2: imported but declared nowhere, minus the vendor allowlist.
    undeclared: list[tuple[str, dict]] = []
    allowed_hits: list[str] = []
    for mod in sorted(imports):
        if dist_for(mod) in declared_all:
            continue
        if mod in ALLOWED_UNDECLARED_IMPORTS:
            allowed_hits.append(mod)
        else:
            undeclared.append((mod, imports[mod]))

    # D3: optional-only, yet imported with neither guard idiom.
    optional_only = set(req["requirements/optional.txt"]) - set(runtime)
    unguarded_optional: list[tuple[str, dict]] = [
        (mod, info)
        for mod, info in sorted(imports.items())
        if dist_for(mod) in optional_only and not info["guarded_only"]
    ]

    imported_dists = {dist_for(m) for m in imports}
    unused = sorted(set(runtime) - imported_dists)

    return {
        "nfiles": nfiles,
        "nmodules": len(imports),
        "pyproject": pyproject,
        "drift": drift,
        "conflict": conflict,
        "undeclared": undeclared,
        "allowed_hits": allowed_hits,
        "unguarded_optional": unguarded_optional,
        "unused": unused,
        "runtime": runtime,
    }


def report(res: dict, repo: Path, quiet: bool = False) -> int:
    def emit(msg: str = "") -> None:
        if not quiet:
            print(msg)

    if res["nfiles"] == 0 or res["nmodules"] == 0:
        print(
            f"ERROR: scanned {res['nfiles']} files / {res['nmodules']} third-party "
            f"imports under {repo}/flagos_robo -- nothing to check, so RC=0 would "
            f"be meaningless",
            file=sys.stderr,
        )
        return 2

    emit(
        f"scanned {res['nfiles']} python files, {res['nmodules']} third-party import roots, "
        f"{len(res['pyproject'])} pyproject dependencies"
    )

    emit(
        f"\n[D1] DRIFT / CONFLICT between pyproject and requirements "
        f"({len(res['drift'])} drift, {len(res['conflict'])} conflict)"
    )
    for line in res["drift"]:
        emit(f"    DRIFT     {line}")
    for line in res["conflict"]:
        emit(f"    CONFLICT  {line}")

    emit(
        f"\n[D2] imports declared in no requirements file "
        f"({len(res['allowed_hits'])} allowlisted, {len(res['undeclared'])} NOT allowlisted)"
    )
    for mod in res["allowed_hits"]:
        emit(f"    allowed   {mod:18s} {ALLOWED_UNDECLARED_IMPORTS[mod]}")
    for mod, info in res["undeclared"]:
        tag = "guarded" if info["guarded_only"] else "HARD"
        emit(
            f"    UNDECLARED {mod:17s} dist={dist_for(mod):22s} {tag}, {len(info['sites'])} site(s)"
        )
        for site in info["sites"][:3]:
            emit(f"               {site}")

    emit(
        f"\n[D3] optional-only dependencies imported with no guard "
        f"({len(res['unguarded_optional'])})"
    )
    for mod, info in res["unguarded_optional"]:
        emit(f"    UNGUARDED {mod:18s} dist={dist_for(mod):18s} {len(info['sites'])} site(s)")
        for site in info["sites"][:5]:
            emit(f"              {site}")

    emit(
        f"\n[report] declared for runtime but never imported ({len(res['unused'])}) "
        f"-- transitive deps live here, not gated"
    )
    for dist in res["unused"]:
        emit(f"    unused?   {dist:22s} {res['runtime'][dist]}")

    hard = (
        len(res["drift"])
        + len(res["conflict"])
        + len(res["undeclared"])
        + len(res["unguarded_optional"])
    )
    if hard:
        emit(
            f"\nFAIL(1): {len(res['drift'])} drift, {len(res['conflict'])} conflict, "
            f"{len(res['undeclared'])} undeclared, {len(res['unguarded_optional'])} unguarded-optional"
        )
        if res["undeclared"]:
            emit(
                "    Declare it in the right requirements file, or -- if it ships inside a\n"
                "    vendor image -- add it to ALLOWED_UNDECLARED_IMPORTS with the reason."
            )
        if res["unguarded_optional"]:
            emit(
                "    Either promote it out of optional.txt, or wrap the import in\n"
                "    try/except ImportError or an importlib.util.find_spec() probe."
            )
        return 1
    emit("\nCLEAN: declarations agree with imports; no undeclared or mislabeled dependencies.")
    return 0


# --------------------------------------------------------------------------- #
# self-test: prove the gate can go red, and prove it leaves correct code alone
# --------------------------------------------------------------------------- #

# D3 cases use modules that optional.txt really declares (torchcodec,
# flash-attn), because D3 is only about *optional-only* dependencies. An
# undeclared module is a D2 matter and stays red whether it is guarded or not --
# a guard keeps the process alive, it does not declare anything.
#
# The two guard idioms below are both in use here, so both must stay green: a
# gate that only understood try/except would flag video_utils.py, which is
# correct code -- and a gate that flags correct code gets deleted.
_NEG_GUARD_TRY_EXCEPT = """\
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None
"""

_NEG_GUARD_FIND_SPEC = """\
import importlib.util

if importlib.util.find_spec("torchcodec"):
    from torchcodec.decoders import VideoDecoder
else:
    raise ImportError("torchcodec is required but not available.")
"""

_NEG_VENDOR_ALLOWLISTED = "import torch_npu\n"

# Same optional-only modules as the two green cases, imported with no probe at
# all: that is the real defect D3 exists to catch.
_POS_OPTIONAL_UNGUARDED_TRY = "from flash_attn import flash_attn_func\n"

_POS_OPTIONAL_UNGUARDED_SPEC = "from torchcodec.decoders import VideoDecoder\n"

_POS_UNDECLARED_HARD = "import some_absent_optional_pkg_xyz\n"

_POS_UNDECLARED_GUARDED = """\
try:
    import another_absent_pkg_xyz
except ImportError:
    another_absent_pkg_xyz = None
"""


def _run_case(root: Path, rel: str, text: str) -> int:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    try:
        return report(scan(root), root, quiet=True)
    finally:
        target.unlink()


def _run_mutated_requirements(root: Path, mutate) -> int:
    """Apply a reversible edit to a declaration file and re-run the gate."""
    saved: dict[Path, str] = {}
    try:
        for path, text in mutate(root).items():
            saved[path] = path.read_text(encoding="utf-8") if path.is_file() else ""
            path.write_text(text, encoding="utf-8")
        return report(scan(root), root, quiet=True)
    finally:
        for path, text in saved.items():
            path.write_text(text, encoding="utf-8")


def _drop_pyproject_dep_from_requirements(root: Path) -> dict[Path, str]:
    """Remove hydra-core from requirements so pyproject's entry dangles -> D1 DRIFT."""
    out: dict[Path, str] = {}
    for rel in REQUIREMENT_FILES:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        kept = [ln for ln in text.splitlines() if not re.match(r"(?i)^hydra[-_]core\b", ln.strip())]
        if len(kept) != len(text.splitlines()):
            out[path] = "\n".join(kept) + "\n"
    return out


def _pin_below_floor(root: Path) -> dict[Path, str]:
    """Pin hydra-core below pyproject's >= floor -> D1 CONFLICT."""
    out: dict[Path, str] = {}
    for rel in REQUIREMENT_FILES:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        new = re.sub(r"(?im)^(hydra[-_]core)\s*==.*$", r"\1==0.1.0", text)
        if new != text:
            out[path] = new
    return out


def _move_hard_dep_to_optional(root: Path) -> dict[Path, str]:
    """Move an unguarded hard import (av) into optional.txt only -> D3."""
    out: dict[Path, str] = {}
    optional = root / "requirements/optional.txt"
    moved: str | None = None
    for rel in RUNTIME_FILES:
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if re.match(r"(?i)^av\s*==", line.strip()):
                moved = line.strip()
        new = (
            "\n".join(ln for ln in text.splitlines() if not re.match(r"(?i)^av\s*==", ln.strip()))
            + "\n"
        )
        if new != text:
            out[path] = new
    if moved and optional.is_file():
        out[optional] = optional.read_text(encoding="utf-8").rstrip("\n") + f"\n{moved}\n"
    return out


def run_self_test(repo: Path) -> int:
    print("self-test: gate behavior on synthetic cases")
    failures: list[str] = []

    with tempfile.TemporaryDirectory() as td:
        sandbox = Path(td) / "repo"
        shutil.copytree(
            repo,
            sandbox,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "_ci", "build", "dist", "*.egg-info"
            ),
        )

        base = report(scan(sandbox), sandbox, quiet=True)
        print(f"  [base ] unmodified tree                       RC={base} (expect 0)")
        if base != 0:
            failures.append(
                "the unmodified tree is not clean; fix the tree (or the allowlist) "
                "before trusting any case below"
            )

        negatives = [
            ("optional dep behind try/except ImportError", _NEG_GUARD_TRY_EXCEPT),
            ("optional dep behind importlib find_spec", _NEG_GUARD_FIND_SPEC),
            ("vendor runtime on the allowlist", _NEG_VENDOR_ALLOWLISTED),
        ]
        for label, text in negatives:
            rc = _run_case(sandbox, "flagos_robo/_selftest_dep_case.py", text)
            ok = rc == 0
            print(f"  [green] {label:<42} RC={rc} (expect 0) {'ok' if ok else 'MISFIRE'}")
            if not ok:
                failures.append(f"false positive on a correct pattern: {label}")

        positives_code = [
            ("undeclared hard import", _POS_UNDECLARED_HARD),
            ("undeclared guarded import, still undeclared", _POS_UNDECLARED_GUARDED),
            ("optional-only dep imported with no guard", _POS_OPTIONAL_UNGUARDED_TRY),
            ("optional-only codec imported with no probe", _POS_OPTIONAL_UNGUARDED_SPEC),
        ]
        for label, text in positives_code:
            rc = _run_case(sandbox, "flagos_robo/_selftest_dep_case.py", text)
            ok = rc == 1
            print(f"  [red  ] {label:<42} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}")
            if not ok:
                failures.append(f"missed a real defect: {label}")

        positives_decl = [
            ("pyproject dep in no requirements file", _drop_pyproject_dep_from_requirements),
            ("requirements pin below pyproject floor", _pin_below_floor),
            ("hard dep demoted to optional-only", _move_hard_dep_to_optional),
        ]
        for label, mutate in positives_decl:
            rc = _run_mutated_requirements(sandbox, mutate)
            ok = rc == 1
            print(f"  [red  ] {label:<42} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}")
            if not ok:
                failures.append(f"missed a real defect: {label}")

        after = report(scan(sandbox), sandbox, quiet=True)
        print(f"  [back ] tree restored after mutations         RC={after} (expect 0)")
        if after != 0:
            failures.append("a mutation leaked; the sandbox did not return to clean")

        empty = Path(td) / "empty"
        (empty / "flagos_robo").mkdir(parents=True)
        rc = report(scan(empty), empty, quiet=True)
        ok = rc == 2
        print(
            f"  [rc2  ] {'empty tree scans nothing':<42} RC={rc} (expect 2) {'ok' if ok else 'BAD'}"
        )
        if not ok:
            failures.append("an empty tree did not produce RC=2; RC=0 could mean no work")

    if failures:
        print("\nself-test FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nself-test OK: 3 correct patterns stay green, 7 defects go red, empty tree RC=2.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate dependency declarations against the imports in flagos_robo/.",
    )
    parser.add_argument("root", nargs="?", default=".", help="repo root (default: cwd)")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="verify the gate itself: correct code green, defects red",
    )
    parser.add_argument("--quiet", action="store_true", help="exit code only")
    args = parser.parse_args(argv)

    repo = Path(args.root).resolve()
    if not (repo / "flagos_robo").is_dir():
        print(f"ERROR: {repo}/flagos_robo not found", file=sys.stderr)
        return 2
    if args.self_test:
        return run_self_test(repo)
    return report(scan(repo), repo, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
