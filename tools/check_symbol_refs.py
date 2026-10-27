#!/usr/bin/env python3
"""Internal *symbol*-level reference gate for FlagOS-Robo (pure stdlib, no imports of the code).

Why this gate exists -- it covers a blind spot the import-closure gate cannot see.
``tools/check_import_closure.py`` resolves ``from flagos_robo.a.b import Y`` down to
the *module* ``flagos_robo.a.b`` and deliberately discards ``Y``; its own
``owning_module()`` docstring says so ("ACTION_DIM is an *attribute* import; the
module that must be migrated is ...constants"), and its ``unresolved`` bucket is
printed but never counted toward the verdict. That is correct for *migration*
questions and wrong for *delivery* questions: this repo was built by SUBSETTING
upstream FlagScale -- whole classes were dropped from ``backend/__init__.py`` and
``cli.py``/``run.py`` were rewritten as subsets -- so a reference to a dropped
*name* inside a module that still resolves is a latent ImportError/AttributeError
that passes every existing gate.

Three verdict classes, deliberately NOT merged (see ``--help`` epilog):

  D1 DANGLING  (hard, must be zero)
      ``from <internal module> import Y`` where the module resolves to a file but
      Y is neither defined, re-exported, nor a submodule -- and the import is NOT
      inside a ``try/except ImportError``. An unguarded dangling name is an import
      crash at startup.

  D2 NAME-GAP  (hard, must be zero)
      A ``try: import ...`` / ``except ImportError:`` fallback where the handler
      does not rebind a name the try block bound, the handler does not bail out
      (return/raise/continue/break), AND the unbound name is actually *read* after
      the try statement. All three conditions are required: a handler that returns,
      or a name used only inside the try body, is a correct graceful degradation,
      not a defect. Checking only "try binds a name the except does not" reports
      those correct patterns as failures.

  D3 GUARDED-UNRESOLVED  (reported, allowlisted)
      ``from <internal module> import Y`` inside a try/except ImportError where the
      module does not exist in this subset at all. That is the documented way this
      repo drops upstream-only features (see ``flagos_robo/transformations/__init__.py``
      and ``state_scope_transformation.py``). Known sites live in ``ALLOWED_GUARDED_MODULES``
      with a reason; a *new* unlisted site fails the gate, so the exception set
      cannot grow silently.

Exit codes: 0 clean, 1 defects found, 2 nothing scanned / bad arguments.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
import tempfile
from pathlib import Path

PKG = "flagos_robo"
SCAN_DIRS = ("flagos_robo", "tests", "tools", "examples")

# Vendored upstream clones must not be scanned: they are not delivered code and
# they reference modules this subset never migrated. Same pruning rule as
# tools/check_all_dangling.py -- any new vendored tree must be added here too.
PRUNE_DIRS = ("__pycache__", "_ci", ".git", "build", "dist", ".mypy_cache")

# D3 allowlist: internal modules that are intentionally absent from this subset
# and are therefore imported behind try/except ImportError. Value = why.
ALLOWED_GUARDED_MODULES = {
    "flagos_robo.inference.core.diffusion.timestep_tracker_transformation": "FlagScale diffusion-inference extra; registry stays constructible without it "
    "(flagos_robo/transformations/__init__.py registers it only if importable)",
    "flagos_robo.inference.core.diffusion.taylorseer_transformation": "FlagScale diffusion-inference extra; same conditional registration",
    "flagos_robo.inference.runtime_context": "FlagScale diffusion engine's context provider is out of robotics scope; "
    "state_scope_transformation falls back to a current_ctx() that returns None",
}

BAIL_NODES = (ast.Return, ast.Raise, ast.Continue, ast.Break)


# --------------------------------------------------------------------------- #
# tree walking / module <-> path mapping
# --------------------------------------------------------------------------- #


def iter_py(root: Path):
    """Yield delivered .py files under the scanned subtrees, pruning vendored dirs."""
    for d in SCAN_DIRS:
        sub = root / d
        if not sub.is_dir():
            continue
        for p in sorted(sub.rglob("*.py")):
            if any(part in PRUNE_DIRS for part in p.parts):
                continue
            yield p


def mod_name_of(root: Path, path: Path) -> str | None:
    """flagos_robo/a/b.py -> flagos_robo.a.b ; .../__init__.py -> flagos_robo.a"""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    parts = list(rel.parts)
    if parts[0] != PKG:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def module_to_path(root: Path, mod: str) -> Path | None:
    if not mod.startswith(PKG):
        return None
    rel = Path(*mod.split("."))
    for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
        if cand.is_file():
            return cand
    if (root / rel).is_dir():
        return root / rel  # PEP 420 namespace dir
    return None


def resolve_relative(cur_mod: str, is_pkg: bool, level: int, module: str | None) -> str:
    """PEP 328 relative import -> absolute module name.

    level=1 means "the package containing this file". For a package's __init__.py
    cur_mod IS that package; for a plain module the containing package is cur_mod
    minus its last component. Each extra level strips one more. Getting this wrong
    makes every ``from .x import Y`` resolve to <self>.x and report false defects.
    """
    parts = cur_mod.split(".")
    if not is_pkg:
        parts = parts[:-1]
    if level > 1:
        parts = parts[: len(parts) - (level - 1)]
    if module:
        parts = parts + module.split(".")
    return ".".join(parts)


def symbols_of(path: Path) -> tuple[set[str], bool]:
    """-> (names bound at module scope, module_uses_star_import).

    A star import makes the symbol set unknowable by AST, so such modules are
    excluded from D1 rather than guessed about.
    """
    if path.is_dir():
        return set(), False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return set(), False

    syms: set[str] = set()
    star = False

    def bind(node: ast.AST) -> None:
        nonlocal star
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            syms.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    syms.add(tgt.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            syms.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                if a.name == "*":
                    star = True
                    continue
                syms.add(a.asname or a.name.split(".")[0])

    for node in tree.body:
        bind(node)
        # Conditionally/guard-defined names still bind at module scope.
        if isinstance(node, (ast.Try, ast.If, ast.With)):
            for sub in ast.walk(node):
                bind(sub)
    return syms, star


# --------------------------------------------------------------------------- #
# D1 helpers: guardedness of an import site
# --------------------------------------------------------------------------- #


def _handles_import_error(node: ast.Try) -> bool:
    """True if any handler would catch a failing import."""
    for h in node.handlers:
        if h.type is None:  # bare except
            return True
        names: list[str] = []
        if isinstance(h.type, ast.Name):
            names = [h.type.id]
        elif isinstance(h.type, ast.Tuple):
            names = [e.id for e in h.type.elts if isinstance(e, ast.Name)]
        if any(n in ("ImportError", "ModuleNotFoundError", "Exception") for n in names):
            return True
    return False


def _guarded_node_ids(tree: ast.AST) -> set[int]:
    """ids of nodes inside a ``try:`` body whose handlers catch import failure."""
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and _handles_import_error(node):
            for stmt in node.body:
                for sub in ast.walk(stmt):
                    guarded.add(id(sub))
    return guarded


def import_sites(path: Path, cur_mod: str | None, is_pkg: bool):
    """Yield (lineno, abs_module, names, is_guarded) for internal from-imports."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return
    guarded = _guarded_node_ids(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        names = [a.name for a in node.names if a.name != "*"]
        if not names:
            continue
        if node.level:
            if cur_mod is None:
                continue  # relative import outside the package: not resolvable
            abs_mod = resolve_relative(cur_mod, is_pkg, node.level, node.module)
        else:
            abs_mod = node.module or ""
        if abs_mod.startswith(PKG):
            yield node.lineno, abs_mod, names, id(node) in guarded


# --------------------------------------------------------------------------- #
# D2: try/except ImportError fallbacks that leave a *used* name unbound
# --------------------------------------------------------------------------- #


def _bound_names(stmts) -> set[str]:
    out: set[str] = set()
    for st in stmts:
        for sub in ast.walk(st):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                for a in sub.names:
                    if a.name != "*":
                        out.add(a.asname or a.name.split(".")[0])
            elif isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Name):
                        out.add(tgt.id)
            elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                out.add(sub.target.id)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                out.add(sub.name)
    return out


def _is_noreturn_call(node: ast.AST) -> bool:
    """True for calls that terminate the process: sys.exit / os._exit / exit / quit.

    These are plain Call nodes, not Return/Raise, so a handler ending in
    ``sys.exit(1)`` (examples/pi0/download.py) looks like a fall-through to a
    purely syntactic check while actually never reaching the code below.
    """
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return f.id in ("exit", "quit")
    if isinstance(f, ast.Attribute):
        base = f.value
        if isinstance(base, ast.Name):
            return (base.id, f.attr) in (
                ("sys", "exit"),
                ("os", "_exit"),
                ("os", "abort"),
            )
    return False


def _handler_bails(h: ast.ExceptHandler) -> bool:
    """True if the handler cannot fall through to code after the try statement.

    ``except ImportError: return`` (utils/determinism.py) and
    ``except ImportError: ... sys.exit(1)`` (examples/pi0/download.py) are both
    correct graceful degradations: the names the try block failed to bind are
    never reached.
    """
    for st in h.body:
        for sub in ast.walk(st):
            if isinstance(sub, BAIL_NODES) or _is_noreturn_call(sub):
                return True
    return False


def _enclosing_scope(tree: ast.AST, try_node: ast.Try) -> ast.AST:
    """Innermost function (or the module) whose body contains ``try_node``.

    Scope matters because Python resolves names per scope, not per line. Walking
    the whole module and comparing line numbers reports a *different* function's
    same-named local as "read after the try" -- e.g. ``current_time`` in
    runner/elastic/monitor_service.py, which is only read inside its own try body
    while a later method happens to reuse the name.
    """
    best: ast.AST = tree
    best_span = 1 << 30
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        lo = node.lineno
        hi = getattr(node, "end_lineno", lo) or lo
        if (
            lo <= try_node.lineno
            and (getattr(try_node, "end_lineno", try_node.lineno) or try_node.lineno) <= hi
        ):
            span = hi - lo
            if span < best_span:
                best, best_span = node, span
    return best


def _names_read_after(tree: ast.AST, try_node: ast.Try) -> set[str]:
    """Names read (ast.Load) after the try statement, within its own scope.

    Two exclusions, both required to avoid reporting correct code:
      * reads *inside* the try body -- if the import raised, that body already
        stopped at the failing line, so a name used only there (``dist`` in
        train/perf_monitor/perf_logger.py) is never observed unbound;
      * reads in any *other* scope -- a same-named local in a sibling function
        is a different binding entirely.
    """
    end = getattr(try_node, "end_lineno", try_node.lineno) or try_node.lineno
    scope = _enclosing_scope(tree, try_node)
    scope_end = getattr(scope, "end_lineno", None) or (1 << 30)

    body_lines: set[int] = set()
    for st in try_node.body:
        lo = st.lineno
        hi = getattr(st, "end_lineno", lo) or lo
        body_lines.update(range(lo, hi + 1))

    # Nested functions inside the scope are deferred bodies: a name read there
    # runs later (and may close over a rebinding), so it is not a straight-line
    # use of the possibly-unbound name.
    nested_lines: set[int] = set()
    for node in ast.walk(scope):
        if node is scope:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            lo = node.lineno
            hi = getattr(node, "end_lineno", lo) or lo
            nested_lines.update(range(lo, hi + 1))

    # Reaching-definitions kill set: the FIRST line at or after the try on which
    # a name is re-bound. A read below that line sees the new binding, so it
    # cannot observe the try's failure -- e.g. tools/acceptance/d7_instantiate.py
    # assigns `cfg` again in a later try before reading it.
    rebind_line: dict[str, int] = {}

    def note_rebind(name: str, ln: int) -> None:
        if ln > end and (name not in rebind_line or ln < rebind_line[name]):
            rebind_line[name] = ln

    for node in ast.walk(scope):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.lineno not in nested_lines:
                note_rebind(node.id, node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if node.lineno not in nested_lines:
                for a in node.names:
                    if a.name != "*":
                        note_rebind(a.asname or a.name.split(".")[0], node.lineno)

    out: set[str] = set()
    for node in ast.walk(scope):
        if not isinstance(node, ast.Name) or not isinstance(node.ctx, ast.Load):
            continue
        ln = node.lineno
        if ln <= end or ln > scope_end:
            continue
        if ln in body_lines or ln in nested_lines:
            continue
        killed = rebind_line.get(node.id)
        if killed is not None and killed < ln:
            continue
        out.add(node.id)
    return out


def _names_bound_before(tree: ast.AST, try_node: ast.Try) -> set[str]:
    """Names already bound *above* the try, in its own scope (plus parameters).

    A pre-set default makes the handler harmless: the read after the try sees the
    earlier binding, not an unbound name. runner/runner_train.py does exactly
    this -- ``result = ""`` sits above two import-guarded trys, so the later
    ``result = result.stdout... if result else ""`` is safe. Upstream FlagScale
    carries the identical shape. Definite-assignment must consider *all* paths
    reaching the read, so scanning only below the try is half an analysis.

    Only *unconditional* bindings count. A binding nested in an if/for/while/try
    above the try is not on every path, so it does not make the read safe --
    ``if cond: result = ""`` leaves ``result`` unbound when cond is false, which
    is a real NameError and must stay reported. Hence the straight-line walk over
    the scope's own body rather than ast.walk over everything above the try.
    """
    scope = _enclosing_scope(tree, try_node)
    start = try_node.lineno

    out: set[str] = set()
    args = getattr(scope, "args", None)
    if args is not None:
        for a in (
            list(getattr(args, "posonlyargs", []))
            + list(args.args)
            + list(getattr(args, "kwonlyargs", []))
        ):
            out.add(a.arg)
        for extra in (args.vararg, args.kwarg):
            if extra is not None:
                out.add(extra.arg)

    for stmt in getattr(scope, "body", []):
        if stmt.lineno >= start:
            break
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in targets:
                for sub in ast.walk(t):
                    if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                        out.add(sub.id)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                if a.name != "*":
                    out.add(a.asname or a.name.split(".")[0])
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(stmt.name)
        elif isinstance(stmt, ast.With):
            for item in stmt.items:
                if item.optional_vars is not None:
                    for sub in ast.walk(item.optional_vars):
                        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                            out.add(sub.id)
    return out


def fallback_gaps(path: Path):
    """Yield (lineno, missing_names, handler_kinds) for REAL name gaps only.

    Reported only when all four hold: the handler does not rebind the name, no
    handler bails out, the name is actually read after the try statement, and the
    name is not already bound above the try.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try) or not _handles_import_error(node):
            continue
        if any(_handler_bails(h) for h in node.handlers):
            continue
        try_names = _bound_names(node.body)
        exc_names: set[str] = set()
        for h in node.handlers:
            exc_names |= _bound_names(h.body)
        missing = try_names - exc_names - _names_bound_before(tree, node)
        if not missing:
            continue
        real = sorted(missing & _names_read_after(tree, node))
        if not real:
            continue
        kinds: list[str] = []
        for h in node.handlers:
            if h.type is None:
                kinds.append("bare")
            elif isinstance(h.type, ast.Name):
                kinds.append(h.type.id)
            elif isinstance(h.type, ast.Tuple):
                kinds.extend(e.id for e in h.type.elts if isinstance(e, ast.Name))
        yield node.lineno, real, kinds


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #


def scan(root: Path) -> dict:
    """Return the three defect buckets plus work-volume counters.

    The counters exist so that RC=0 cannot be produced by scanning nothing: the
    caller fails with RC=2 when files_scanned or sites_checked is zero.
    """
    files = list(iter_py(root))
    sym_cache: dict[str, tuple[set[str], bool]] = {}

    def syms_for(mod: str):
        if mod not in sym_cache:
            p = module_to_path(root, mod)
            sym_cache[mod] = symbols_of(p) if p is not None else (set(), False)
        return sym_cache[mod]

    dangling: list[str] = []  # D1 hard
    name_gaps: list[str] = []  # D2 hard
    guarded_unresolved: list[str] = []  # D3 allowlisted
    unlisted_guarded: list[str] = []  # D3 violations
    sites = 0

    for path in files:
        rel = path.relative_to(root)
        cur_mod = mod_name_of(root, path)
        is_pkg = path.name == "__init__.py"

        for lineno, mod, names, is_guarded in import_sites(path, cur_mod, is_pkg):
            sites += 1
            target = module_to_path(root, mod)
            if target is None:
                # module itself absent from this subset
                if is_guarded:
                    reason = ALLOWED_GUARDED_MODULES.get(mod)
                    line = f"{rel}:{lineno}  {mod} ({', '.join(names)})"
                    if reason:
                        guarded_unresolved.append(f"{line}\n        reason: {reason}")
                    else:
                        unlisted_guarded.append(line)
                else:
                    dangling.append(f"{rel}:{lineno}  module missing: {mod} ({', '.join(names)})")
                continue
            if is_guarded:
                continue  # module present; a guarded symbol import degrades by design
            have, star = syms_for(mod)
            if star:
                continue  # star import makes the symbol set unknowable
            for n in names:
                if n in have:
                    continue
                if module_to_path(root, f"{mod}.{n}") is not None:
                    continue  # submodule import, not an attribute
                dangling.append(f"{rel}:{lineno}  {mod} has no symbol {n!r}")

        for lineno, missing, kinds in fallback_gaps(path):
            name_gaps.append(
                f"{rel}:{lineno}  except({'/'.join(kinds)}) leaves unbound and later read: "
                + ", ".join(missing)
            )

    return {
        "files_scanned": len(files),
        "sites_checked": sites,
        "dangling": dangling,
        "name_gaps": name_gaps,
        "guarded_unresolved": guarded_unresolved,
        "unlisted_guarded": unlisted_guarded,
    }


def report(res: dict, root: Path, quiet: bool = False) -> int:
    def emit(*a):
        if not quiet:
            print(*a)

    bar = "=" * 74
    emit(bar)
    emit(f"FlagOS-Robo internal symbol-reference gate   repo={root}")
    emit(
        f"scanned {res['files_scanned']} python files, "
        f"{res['sites_checked']} internal from-import sites"
    )
    emit(bar)

    if res["files_scanned"] == 0 or res["sites_checked"] == 0:
        emit("\nFAIL(2): nothing was scanned -- wrong --repo, or the tree is empty.")
        return 2

    emit(f"\n[D1] DANGLING unguarded symbol/module references ({len(res['dangling'])})")
    for line in res["dangling"] or ["    (none)"]:
        emit(f"    {line}" if line.strip() != "(none)" else line)

    emit(f"\n[D2] NAME-GAP fallbacks, name unbound AND read later ({len(res['name_gaps'])})")
    for line in res["name_gaps"] or ["    (none)"]:
        emit(f"    {line}" if line.strip() != "(none)" else line)

    emit(
        f"\n[D3] GUARDED-UNRESOLVED, intentional subset drops "
        f"({len(res['guarded_unresolved'])} allowlisted, "
        f"{len(res['unlisted_guarded'])} NOT allowlisted)"
    )
    for line in res["guarded_unresolved"]:
        emit(f"    ok       {line}")
    for line in res["unlisted_guarded"]:
        emit(f"    UNLISTED {line}")
    if not res["guarded_unresolved"] and not res["unlisted_guarded"]:
        emit("    (none)")

    bad = len(res["dangling"]) + len(res["name_gaps"]) + len(res["unlisted_guarded"])
    emit("\n" + bar)
    if bad:
        emit(
            f"FAIL(1): {len(res['dangling'])} dangling, {len(res['name_gaps'])} name-gap, "
            f"{len(res['unlisted_guarded'])} unlisted-guarded"
        )
        if res["unlisted_guarded"]:
            emit(
                "    A new guarded import of a module absent from this subset must be\n"
                "    added to ALLOWED_GUARDED_MODULES in this file, with the reason it\n"
                "    is intentionally not migrated."
            )
        return 1
    emit("CLEAN: no dangling symbols, no real name gaps, no unlisted guarded drops.")
    return 0


# --------------------------------------------------------------------------- #
# self-test: prove the gate can go red, not just that it says CLEAN
# --------------------------------------------------------------------------- #

_NEG_GUARDED_DROP = """\
try:
    from flagos_robo.inference.runtime_context import current_ctx
except ImportError:
    def current_ctx():
        return None
VALUE = current_ctx()
"""

_NEG_HANDLER_BAILS = """\
def seed():
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(0)
"""

_NEG_USED_ONLY_IN_TRY = """\
RANK = 0
try:
    import torch.distributed as dist
    if dist.is_initialized():
        RANK = dist.get_rank()
except ImportError:
    pass
print(RANK)
"""

_POS_DANGLING_SYMBOL = "from flagos_robo.utils.determinism import no_such_symbol_xyz\n"

_POS_DANGLING_MODULE = "from flagos_robo.no_such_module_xyz import Thing\n"

_POS_REAL_NAME_GAP = """\
try:
    from flagos_robo.utils.determinism import seed_worker
except ImportError:
    pass
HANDLER = seed_worker
"""

_POS_UNLISTED_GUARDED = """\
try:
    from flagos_robo.not_migrated_feature_xyz import Feature
except ImportError:
    Feature = None
USED = Feature
"""

# The pre-set default that makes a handler harmless must be *unconditional*. A
# binding inside an if/for/while is not on every path, so the read after the try
# can still raise NameError. This case exists because a path-insensitive
# ast.walk over everything above the try silently MISSES it -- a false negative,
# the direction a gate must never fail in.
_POS_CONDITIONAL_PRESET = """\
def f(cond):
    if cond:
        handler = None
    try:
        from flagos_robo.utils.determinism import seed_worker
        handler = seed_worker
    except ImportError:
        pass
    return handler
"""

# Counterpart to the case above: the same shape with the default hoisted to the
# function's straight-line body is genuinely safe and must stay green. Together
# the pair pins the definite-assignment rule from both sides.
_NEG_UNCONDITIONAL_PRESET = """\
def f(cond):
    handler = None
    try:
        from flagos_robo.utils.determinism import seed_worker
        handler = seed_worker
    except ImportError:
        pass
    return handler
"""


def _run_case(root: Path, rel: str, text: str) -> int:
    """Write one case file into a scratch copy of the repo and return the RC."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    try:
        return report(scan(root), root, quiet=True)
    finally:
        target.unlink()


def run_self_test(repo: Path) -> int:
    """Bidirectional: correct patterns must stay green, real defects must go red.

    Negative cases are the *documented* graceful-degradation shapes this repo
    actually uses -- a gate that flags them would be reverted within a week, so
    keeping them green is as much a requirement as catching the positives.
    """
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
        print(f"  [base ] unmodified tree                     RC={base} (expect 0)")
        if base != 0:
            failures.append(
                "the unmodified tree is not clean; fix the tree (or the allowlist) "
                "before trusting any case below"
            )

        negatives = [
            ("guarded drop of an allowlisted module", _NEG_GUARDED_DROP),
            ("handler returns, name never reached", _NEG_HANDLER_BAILS),
            ("name used only inside the try body", _NEG_USED_ONLY_IN_TRY),
            ("default pre-set unconditionally above try", _NEG_UNCONDITIONAL_PRESET),
        ]
        for label, text in negatives:
            rc = _run_case(sandbox, "flagos_robo/_selftest_case.py", text)
            ok = rc == 0
            print(f"  [green] {label:<40} RC={rc} (expect 0) {'ok' if ok else 'MISFIRE'}")
            if not ok:
                failures.append(f"false positive on a correct pattern: {label}")

        positives = [
            ("unguarded import of a missing symbol", _POS_DANGLING_SYMBOL),
            ("unguarded import of a missing module", _POS_DANGLING_MODULE),
            ("real name gap, unbound name read later", _POS_REAL_NAME_GAP),
            ("guarded drop not in the allowlist", _POS_UNLISTED_GUARDED),
            ("default pre-set only on one branch", _POS_CONDITIONAL_PRESET),
        ]
        for label, text in positives:
            rc = _run_case(sandbox, "flagos_robo/_selftest_case.py", text)
            ok = rc == 1
            print(f"  [red  ] {label:<40} RC={rc} (expect 1) {'ok' if ok else 'BLIND'}")
            if not ok:
                failures.append(f"missed a real defect: {label}")

        empty = Path(td) / "empty"
        (empty / "flagos_robo").mkdir(parents=True)
        rc = report(scan(empty), empty, quiet=True)
        ok = rc == 2
        print(
            f"  [rc2  ] {'empty tree scans nothing':<40} RC={rc} (expect 2) {'ok' if ok else 'BAD'}"
        )
        if not ok:
            failures.append("an empty tree did not produce RC=2; RC=0 could mean no work")

    if failures:
        print("\nself-test FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nself-test OK: 4 correct patterns stay green, 5 defects go red, empty tree RC=2.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gate internal symbol-level references for FlagOS-Robo.",
        epilog=(
            "Exit codes: 0 clean, 1 defects, 2 nothing scanned. "
            "This gate is the symbol-level complement to tools/check_import_closure.py, "
            "which resolves imports to the module and discards the imported name."
        ),
    )
    ap.add_argument(
        "repo",
        nargs="?",
        default=None,
        help="repo root (default: the parent of this tools/ directory)",
    )
    ap.add_argument("--repo", dest="repo_opt", default=None, help="same as the positional form")
    ap.add_argument("--self-test", action="store_true", help="prove the gate can go red")
    ap.add_argument("--quiet", action="store_true", help="verdict lines only")
    args = ap.parse_args(argv)

    raw = args.repo_opt or args.repo
    root = Path(raw).resolve() if raw else Path(__file__).resolve().parent.parent
    if not (root / PKG).is_dir():
        print(f"FAIL(2): {root} does not contain a {PKG}/ package", file=sys.stderr)
        return 2

    if args.self_test:
        return run_self_test(root)
    return report(scan(root), root, quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
