# Copyright (c) 2025, BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for tools/check_import_closure.py.

The tool already ships a ``--self-test`` that builds a synthetic package and
plants one leak per channel. These tests cover what the self-test cannot:

* the self-test itself actually passes (regression gate in CI);
* the ``=exact`` vs subtree planned-set semantics;
* the vacuity guard, which exists because a bare package-root entry made the
  first real scan report ``OUTSIDE: 0`` tautologically;
* ``--json`` emits parseable json with no banner line.

This file names Megatron modules on purpose: it asserts that FlagScale's
megatron training tree stays OUT of the migration scope. That is the opposite
of a Megatron dependency, so it declares the checker's opt-out pragma.

check-no-megatron: allow-self-reference
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "check_import_closure.py"


def load_tool():
    """Import the tool as a module named ``check_import_closure``.

    The name matters: the module defines dataclasses, and dataclasses resolve
    annotations via ``sys.modules[cls.__module__]``. Loading it under a name
    that is not registered in ``sys.modules`` raises AttributeError inside
    dataclasses.py, which is exactly how the first attempt failed.
    """
    name = "check_import_closure"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return load_tool()


def run_tool(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


class TestToolExists:
    def test_tool_file_present(self):
        assert TOOL.is_file(), f"missing deliverable: {TOOL}"

    def test_tool_is_pure_stdlib(self, tool):
        """It must run on a bare python with no FlagScale checkout present."""
        src = TOOL.read_text(encoding="utf-8")
        for banned in ("import yaml", "import hydra", "import omegaconf", "import torch"):
            assert banned not in src, f"tool must stay stdlib-only, found: {banned}"


class TestSelfTest:
    def test_self_test_passes(self):
        res = run_tool("--self-test")
        assert res.returncode == 0, res.stdout + res.stderr
        assert "SELF-TEST PASSED" in res.stdout

    def test_self_test_covers_all_three_channels(self):
        res = run_tool("--self-test")
        out = res.stdout
        for marker in (
            "C1 detects planted static import leak",
            "C2 detects planted dynamic string module path",
            "C3 detects planted yaml entrypoint leak",
        ):
            assert marker in out, f"self-test no longer checks: {marker}"

    def test_self_test_proves_channels_are_not_redundant(self):
        """The whole justification for 3 channels is mutual invisibility."""
        out = run_tool("--self-test").stdout
        assert "INVISIBLE to C1" in out
        assert "INVISIBLE to C1+C2" in out


class TestPlannedSetSemantics:
    def test_subtree_entry_matches_descendants(self, tool):
        planned = ["flagscale.train"]
        assert tool.in_planned("flagscale.train", planned)
        assert tool.in_planned("flagscale.train.utils", planned)
        assert tool.in_planned("flagscale.train.a.b.c", planned)

    def test_subtree_entry_respects_dotted_boundary(self, tool):
        """``a.b`` must never match ``a.bc`` - the classic prefix bug."""
        planned = ["flagscale.train"]
        assert not tool.in_planned("flagscale.training", planned)
        assert not tool.in_planned("flagscale.train_extra", planned)

    def test_exact_entry_matches_only_itself(self, tool):
        planned = ["=flagscale.train"]
        assert tool.in_planned("flagscale.train", planned)
        assert not tool.in_planned("flagscale.train.utils", planned)
        assert not tool.in_planned("flagscale.train.megatron.train_gpt", planned)

    def test_exact_and_subtree_entries_compose(self, tool):
        """Real scope shape: exact package roots + specific subtrees."""
        planned = ["=flagscale", "=flagscale.train", "flagscale.train.datasets"]
        assert tool.in_planned("flagscale", planned)
        assert tool.in_planned("flagscale.train", planned)
        assert tool.in_planned("flagscale.train.datasets.foo", planned)
        # the point of the exact form: siblings stay OUT
        assert not tool.in_planned("flagscale.train.megatron", planned)
        assert not tool.in_planned("flagscale.compress", planned)


class TestVacuityGuard:
    """Regression gate for the bug that made the first real scan meaningless."""

    def test_bare_package_root_subtree_is_rejected(self, tool):
        problems = tool.vacuity_warnings(["flagscale", "flagscale.train"], "flagscale")
        assert len(problems) == 1
        assert "vacuous" in problems[0] or "by construction" in problems[0]

    def test_exact_package_root_is_accepted(self, tool):
        assert tool.vacuity_warnings(["=flagscale", "flagscale.train"], "flagscale") == []

    def test_cli_exits_2_on_vacuous_scope(self, tmp_path):
        """Exit 2 must come from the VACUITY guard, not from 'package missing'.

        The first version of this test passed --source-root REPO_ROOT with
        --package flagscale, which does not exist here: the tool exited 2 for
        an unrelated reason and the test was green for the wrong cause. So
        build a real minimal package first.
        """
        pkg = tmp_path / "flagos_robo"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "mod.py").write_text("x = 1\n", encoding="utf-8")
        scope = tmp_path / "scope.txt"
        scope.write_text("flagos_robo\n", encoding="utf-8")
        res = run_tool(
            "--source-root",
            str(tmp_path),
            "--package",
            "flagos_robo",
            "--planned-from",
            str(scope),
            "--seed",
            "flagos_robo.mod",
        )
        assert res.returncode == 2, f"expected exit 2, got {res.returncode}: {res.stderr}"
        assert "vacuous" in res.stderr, f"exit 2 for the wrong reason: {res.stderr}"

    def test_exact_root_scope_is_not_rejected_by_cli(self, tmp_path):
        """Positive control: the '=' form must get PAST the vacuity guard."""
        pkg = tmp_path / "flagos_robo"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("", encoding="utf-8")
        (pkg / "mod.py").write_text("x = 1\n", encoding="utf-8")
        scope = tmp_path / "scope.txt"
        scope.write_text("=flagos_robo\nflagos_robo.mod\n", encoding="utf-8")
        res = run_tool(
            "--source-root",
            str(tmp_path),
            "--package",
            "flagos_robo",
            "--planned-from",
            str(scope),
            "--seed",
            "flagos_robo.mod",
        )
        assert res.returncode == 0, f"expected clean pass, got {res.returncode}: {res.stderr}"
        assert "vacuous" not in res.stderr


class TestPlannedFileParsing:
    def test_comments_and_blank_lines_ignored(self, tool, tmp_path):
        f = tmp_path / "scope.txt"
        f.write_text(
            "# header\n\nflagscale.train  # trailing comment\n   \n=flagscale\n",
            encoding="utf-8",
        )
        assert tool.load_planned(f) == ["flagscale.train", "=flagscale"]


class TestOriginFilter:
    """C2/C3 must only report refs coming FROM files we actually migrate.

    Without this, scanning FlagScale produced 117 dynamic + 65 text findings
    from parts of the monorepo we never touch, burying the 3 real ones.
    """

    def test_report_tracks_origin_files(self, tool):
        rep = tool.ScanReport(
            seeds=["demo.a"],
            planned=["demo"],
            reached=["demo.a"],
            outside=[],
            unresolved=[],
            dynamic=[
                tool.Reference("dynamic", "demo/a.py", 1, "x", "other.thing"),
                tool.Reference("dynamic", "untouched/b.py", 1, "x", "other.thing"),
            ],
            text=[],
            origins={"demo/a.py"},
        )
        outside = rep.dynamic_outside
        assert len(outside) == 1, "ref from a non-migrated file must be filtered out"
        assert outside[0].file == "demo/a.py"

    def test_opaque_calls_also_origin_filtered(self, tool):
        rep = tool.ScanReport(
            seeds=[],
            planned=["demo"],
            reached=[],
            outside=[],
            unresolved=[],
            dynamic=[
                tool.Reference("dynamic-call", "demo/a.py", 9, "import_module(?)", None),
                tool.Reference("dynamic-call", "untouched/b.py", 9, "import_module(?)", None),
            ],
            text=[],
            origins={"demo/a.py"},
        )
        assert [r.file for r in rep.dynamic_opaque] == ["demo/a.py"]


class TestJsonOutput:
    def test_json_is_parseable_with_no_banner(self, tmp_path):
        """--json must emit ONLY json, so CI/jq need no stdout stripping."""
        scope = tmp_path / "scope.txt"
        scope.write_text("=flagscale\nflagscale.train\n", encoding="utf-8")
        res = run_tool(
            "--source-root",
            str(REPO_ROOT),
            "--package",
            "flagos_robo",
            "--planned-from",
            str(scope),
            "--seed",
            "flagos_robo.platforms",
            "--json",
        )
        payload = json.loads(res.stdout)  # raises if a banner leaked in
        for key in ("seeds", "planned", "reached_count", "outside", "ok"):
            assert key in payload, f"json missing key: {key}"


class TestRealScopeFileIsUsable:
    """docs/migration_scope.txt is a deliverable; keep it parseable and sane."""

    SCOPE = REPO_ROOT / "docs" / "migration_scope.txt"

    def test_scope_file_exists(self):
        assert self.SCOPE.is_file()

    def test_scope_file_parses_and_is_not_vacuous(self, tool):
        planned = tool.load_planned(self.SCOPE)
        assert len(planned) > 20, f"scope looks truncated: {len(planned)} entries"
        assert tool.vacuity_warnings(planned, "flagscale") == [], (
            "docs/migration_scope.txt would make the scan tautological"
        )

    def test_scope_pins_package_roots_as_exact(self, tool):
        planned = tool.load_planned(self.SCOPE)
        assert "=flagscale" in planned, "package root must be an exact (=) entry"

    def test_scope_includes_all_four_entrypoints(self, tool):
        """The 4 entrypoints discovered from examples/*/conf must be in scope."""
        planned = tool.load_planned(self.SCOPE)
        for mod in (
            "flagscale.train.train_pi",
            "flagscale.train.train_gr00t_n1_5",
            "flagscale.inference.inference_pi",
            "flagscale.serve.run_serve_vla",
        ):
            assert tool.in_planned(mod, planned), f"entrypoint out of scope: {mod}"

    def test_scope_excludes_megatron_training_tree(self, tool):
        """FlagOS-Robo must never pull FlagScale's megatron training path."""
        planned = tool.load_planned(self.SCOPE)
        for mod in (
            "flagscale.train.megatron",
            "flagscale.train.megatron.train_gpt",
            "flagscale.compress.compressor",
        ):
            assert not tool.in_planned(mod, planned), f"must stay out of scope: {mod}"
