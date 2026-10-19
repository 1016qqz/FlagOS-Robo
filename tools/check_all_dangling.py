#!/usr/bin/env python3
"""全仓扫描 __all__ 悬挂名：__all__ 里声明、但只在 try/except ImportError 内绑定的名字。

动机：这类名字在依赖齐备时无害，在依赖缺失的环境（CPU-only 容器、文档构建、
      裁剪安装、CI lint）里会让 `from pkg import *` 抛 AttributeError —— 一个
      只在部分环境复现的隐性缺陷，常规 pytest 跑不出来。

方法：纯 AST，不 import 任何重依赖（本机无 torch 也能跑）。
      对每个模块算两个集合：
        guarded   = 只在 try 体内绑定的名字
        unguarded = try 之外（含 except 体内）必然绑定的名字
      悬挂 = __all__ ∩ (guarded - unguarded)

用法：python3 tools/check_all_dangling.py [repo_root]
      python3 tools/check_all_dangling.py --self-test
退出码：发现悬挂 -> 1；干净 -> 0；扫到 0 个文件 -> 2（防「零扫描假通过」）
"""
import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def collect_binds(nodes, sink):
    """把一批语句里会产生名字绑定的部分收进 sink。"""
    for node in nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                sink.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sink.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    sink.add(tgt.id)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            if isinstance(node.target, ast.Name):
                sink.add(node.target.id)
        elif isinstance(node, (ast.If, ast.For, ast.While, ast.With)):
            # 条件/循环体内的绑定不保证执行，但对 __all__ 判定按“可能绑定”处理，
            # 归入 unguarded 以避免假阳性。
            collect_binds(node.body, sink)
            collect_binds(getattr(node, "orelse", []), sink)


def literal_all(node):
    """从 __all__ 赋值节点取出字符串字面量列表，取不到返回 None。"""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return None


def scan_module(path):
    """返回 (declared, dangling)；文件无 __all__ 时 declared 为 None。"""
    with open(path, encoding="utf-8") as fh:
        try:
            tree = ast.parse(fh.read())
        except SyntaxError as exc:
            return None, [f"<SyntaxError: {exc}>"]

    declared = None
    guarded, unguarded = set(), set()

    for stmt in tree.body:
        if isinstance(stmt, ast.Try):
            collect_binds(stmt.body, guarded)
            for handler in stmt.handlers:
                collect_binds(handler.body, unguarded)
            collect_binds(stmt.orelse, guarded)
            collect_binds(stmt.finalbody, unguarded)
        else:
            collect_binds([stmt], unguarded)

        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                    got = literal_all(stmt.value)
                    if got is not None:
                        declared = got

    if declared is None:
        return None, []

    only_guarded = guarded - unguarded
    return declared, [n for n in declared if n in only_guarded]


# os.walk 要剪掉的目录名。
#
# `_ci` 不是可选项，是实测出来的缺陷修复：docs/MIGRATION_SOP.md 要求把上游
# FlagScale 克隆到 `_ci/FlagScale` 供 check_import_closure.py 的 --source-root
# 使用。那棵树里全是上游代码，其中就有本仓已经修掉的 guarded-import + __all__
# 形态。不剪掉的话，本工具会把**上游的**悬挂名报成本仓的缺陷：
#   实测 _ci 存在时 → 扫 170 个文件（本仓 169）、RC=1，点名
#   `_ci/FlagScale/flagscale/models/vla/__init__.py`，而本仓代码是干净的。
# 即「门禁在干净代码上报红」，比漏报更糟 —— 会训练使用者忽略这道门禁。
#
# 为什么只有本工具需要剪：check_leaf_import.py 扫的是 `repo/flagos_robo`
# （rglob 自包名起算），天然进不到 `_ci`；本工具扫的是整棵 root。
_PRUNE = {
    ".git",
    "__pycache__",
    "build",
    "dist",
    ".pytest_cache",
    "_ci",  # 上游克隆，见上
}


def scan_tree(root):
    """扫一棵树，返回 (scanned, with_all, findings)。"""
    scanned = with_all = 0
    findings = []

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in _PRUNE
        ]
        for name in sorted(filenames):
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            scanned += 1
            declared, dangling = scan_module(path)
            if declared is not None:
                with_all += 1
            if dangling:
                findings.append((os.path.relpath(path, root), declared, dangling))

    return scanned, with_all, findings


def report(root):
    scanned, with_all, findings = scan_tree(root)

    print("=" * 72)
    print("__all__ DANGLING-NAME AUDIT")
    print("=" * 72)
    print(f"扫描 .py 文件      : {scanned}")
    print(f"其中定义 __all__   : {with_all}")
    print(f"发现悬挂的文件数   : {len(findings)}")
    print()

    if scanned == 0:
        # 零扫描必须报错，不能打印 CLEAN 返回 0 —— 否则「路径写错」与「仓库干净」
        # 输出完全无法区分，这正是本工具早期版本把 --self-test 当成路径参数吃掉后
        # 产生的假通过。
        print("FAIL: 扫描到 0 个 .py 文件 —— repo_root 写错了？", file=sys.stderr)
        return 2

    if not findings:
        print("CLEAN: 没有 __all__ 名字只在 try/except 内绑定。")
        return 0

    for rel, declared, dangling in findings:
        print(f"[DANGLING] {rel}")
        print(f"           __all__ 共 {len(declared) if declared else 0} 项")
        print(f"           只在 try 内绑定 -> {dangling}")
        print("           依赖缺失时 `from ... import *` 会抛 AttributeError")
        print()
    return 1


# --------------------------------------------------------------- self-test
# 不能 fail 的检查器永远 pass。CI 先把本节当负对照跑一遍，见
# .github/workflows/ci.yml 与 tests/unit_tests/tools/test_ci_workflow.py::
# test_negative_control_steps_are_present。
#
# 每个 case 在 temp 目录造一个合成模块，再以子进程重入本脚本，断言：
# 已知阳性必须被抓到（RC=1 且点名那个符号），已知阴性必须不报（RC=0）。

_POSITIVE = '''
try:
    from .pi0 import PI0Policy
except ImportError:
    pass

__all__ = ["PI0Policy"]
'''

_NEG_FALLBACK = '''
try:
    from .pi0 import PI0Policy
except ImportError:
    PI0Policy = None

__all__ = ["PI0Policy"]
'''

_NEG_PLAIN = '''
class RealClass:
    pass

__all__ = ["RealClass"]
'''

_NEG_NO_ALL = '''
try:
    from .pi0 import PI0Policy
except ImportError:
    pass
'''

_CASES = [
    ("阳性：try 内绑定 + 列进 __all__ 必须被抓到", _POSITIVE, 1, "PI0Policy"),
    ("阴性：except 里有 fallback 赋值，不算悬挂", _NEG_FALLBACK, 0, "CLEAN"),
    ("阴性：普通 class 定义，不算悬挂", _NEG_PLAIN, 0, "CLEAN"),
    ("阴性：无 __all__ 的 guarded import，不算悬挂", _NEG_NO_ALL, 0, "CLEAN"),
]


def run_self_test():
    failures = []
    me = str(Path(__file__).resolve())
    for title, body, want_rc, want_text in _CASES:
        tmp = Path(tempfile.mkdtemp(prefix="dangling-"))
        try:
            (tmp / "mod.py").write_text(body, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, me, str(tmp)], capture_output=True, text=True
            )
            out = proc.stdout + proc.stderr
            if proc.returncode != want_rc:
                failures.append(f"{title}: rc={proc.returncode} 期望 {want_rc}")
            elif want_text not in out:
                failures.append(f"{title}: 输出里找不到 {want_text!r}")
            else:
                print(f"  ok   {title}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # 零扫描必须报错而不是 CLEAN。
    proc = subprocess.run(
        [sys.executable, me, "/nonexistent-root-xyz"], capture_output=True, text=True
    )
    if proc.returncode == 0:
        failures.append("路径不存在却退出 0（假通过）")
    else:
        print("  ok   路径写错时拒绝打印 CLEAN")

    # _ci/ 剪枝：上游克隆树里的悬挂名不能报成本仓缺陷。
    #
    # 这一对是**双向**的，缺任何一边都不算证明：
    #   剪枝侧  _ci/ 下的阳性样本 → 必须 CLEAN RC=0
    #   对照侧  同一样本换成未剪枝目录名 → 必须 RC=1 被抓到
    # 只测剪枝侧的话，「剪枝生效」和「扫描器整体失灵」输出一样。
    # 两侧都种了一个干净的顶层 .py，确保 scanned>0，不会退化成零扫描 RC=2。
    for subdir, want_rc, label in (("_ci", 0, "剪枝"), ("vendor_not_pruned", 1, "对照")):
        tmp = Path(tempfile.mkdtemp(prefix="dangling-prune-"))
        try:
            (tmp / "clean_top.py").write_text(_NEG_PLAIN, encoding="utf-8")
            nested = tmp / subdir / "FlagScale" / "pkg"
            nested.mkdir(parents=True)
            (nested / "__init__.py").write_text(_POSITIVE, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, me, str(tmp)], capture_output=True, text=True
            )
            if proc.returncode != want_rc:
                failures.append(
                    f"_ci 剪枝（{label} {subdir}/）: rc={proc.returncode} 期望 {want_rc}"
                )
            else:
                print(f"  ok   _ci {label}：{subdir}/ 下的悬挂名 rc={want_rc}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print("SELF-TEST FAILED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"SELF-TEST OK: {len(_CASES) + 3} case(s) passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="__all__ 悬挂名审计")
    ap.add_argument("root", nargs="?", default=".", help="仓库根目录")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="负对照：种入已知悬挂名，断言必须被抓到",
    )
    args = ap.parse_args(argv)
    if args.self_test:
        return run_self_test()
    return report(os.path.abspath(args.root))


if __name__ == "__main__":
    sys.exit(main())
