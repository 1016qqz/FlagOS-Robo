# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Consistency tests for the CI workflow draft.

GitHub Actions cannot be executed locally, so the realistic failure mode for a
workflow file is not "the YAML is invalid" but "it passes a flag the tool does
not have" or "it pins a version that drifted from requirements/". Both are
silent until a push burns a CI run. These tests are the cheap proxy.

Written after the draft shipped with two such defects: a non-existent
``--flagscale-root`` flag, and ``pre-commit==4.0.1`` against
``requirements/dev.txt``'s 4.2.0.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WF = REPO_ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def wf_text() -> str:
    assert WF.exists(), f"CI workflow missing at {WF}"
    return WF.read_text()


@pytest.fixture(scope="module")
def wf_flat(wf_text: str) -> str:
    """Line continuations joined, comment lines dropped."""
    flat = wf_text.replace("\\\n", " ")
    return "\n".join(
        ln for ln in flat.splitlines() if not ln.lstrip().startswith("#")
    )


def _tool_invocations(flat: str):
    for line in flat.splitlines():
        # [a-z0-9_] not [a-z_]: a checker named e.g. check_fp8_cast.py must not
        # be invisible to this scan, or wiring it to CI reads as "unwired".
        m = re.search(r"(?:^|\s)(tools/[a-z0-9_]+\.py)(.*)", line)
        if m:
            yield m.group(1), m.group(2)


class TestWorkflowExists:
    def test_workflow_file_present(self, wf_text):
        assert "name: CI" in wf_text

    def test_declares_the_four_w1_gates(self, wf_flat):
        """purity and closure are the automated form of the W1 contract."""
        for job in ("lint:", "purity:", "closure:", "tests-cpu:"):
            assert job in wf_flat, f"job {job} missing from workflow"

    def test_every_job_has_a_checkout(self, wf_text):
        n_jobs = len(re.findall(r"^  [a-z][a-z0-9-]*:\n", wf_text, re.M))
        n_checkouts = wf_text.count("actions/checkout@")
        assert n_checkouts >= n_jobs - 1, (
            f"{n_jobs} jobs but only {n_checkouts} checkouts -- a job without a "
            "checkout silently operates on an empty workspace"
        )


class TestFlagsExist:
    """The defect that shipped: --flagscale-root was never a real flag."""

    def test_workflow_invokes_at_least_the_three_tools(self, wf_flat):
        invoked = {t for t, _ in _tool_invocations(wf_flat)}
        for t in (
            "tools/check_no_megatron.py",
            "tools/check_import_closure.py",
            "tools/compare_precision.py",
        ):
            assert t in invoked, f"{t} is never exercised by CI"

    def test_every_flag_passed_exists_in_the_tool(self, wf_flat):
        problems = []
        checked = 0
        for tool_rel, rest in _tool_invocations(wf_flat):
            tool = REPO_ROOT / tool_rel
            if not tool.exists():
                problems.append(f"missing tool {tool_rel}")
                continue
            src = tool.read_text()
            for flag in re.findall(r"--[a-z0-9][a-z0-9-]+", rest):
                checked += 1
                if f'"{flag}"' not in src:
                    problems.append(f"{tool_rel} has no flag {flag}")
        assert checked > 0, "no flags were checked -- the parser matched nothing"
        assert not problems, problems

    def test_negative_control_steps_are_present(self, wf_flat):
        """A checker that cannot fail passes forever; CI must run --self-test.

        The tool list is DERIVED from the workflow, not hardcoded. A hardcoded
        list only constrains the tools that existed when it was written: when
        check_leaf_import.py and check_all_dangling.py were added as gates G5
        and G6, deleting their --self-test steps outright kept this test green,
        which was measured, not assumed. Any tool CI trusts for a real scan is
        therefore required to carry its own negative control.
        """
        scanned, self_tested = set(), set()
        for tool, rest in _tool_invocations(wf_flat):
            if "--self-test" in rest:
                self_tested.add(tool)
            else:
                scanned.add(tool)
        assert scanned, "no real scan invocation found -- the parser matched nothing"
        missing = sorted(scanned - self_tested)
        assert not missing, (
            f"{missing} are trusted for a real scan but never run with "
            "--self-test; a checker that cannot fail passes forever"
        )
        # Floor: these three predate the derivation and must stay wired even if
        # someone turns their real-scan step into something the parser misses.
        for tool in (
            "tools/check_no_megatron.py",
            "tools/check_import_closure.py",
            "tools/compare_precision.py",
        ):
            assert tool in self_tested, f"{tool} --self-test not wired into CI"


# Gaps that are known, deliberate and scheduled -- not accidents. Each entry is
# a decision, so it carries its reason. Anything NOT listed here is a drift and
# goes red; and an entry that stops being a real gap also goes red, so closing
# the gap forces the allowlist to shrink instead of silently rotting.
KNOWN_UNWIRED_CHECKERS = {
    "tools/check_conf_paths.py": (
        "deferred to the dedicated CI workstream together with promoting "
        "gates.sh into the repo as tools/run_gates.sh (review items B/B')"
    ),
}
KNOWN_GATES_SCRIPT_GAPS = {
    "tools/compare_precision.py": (
        "gates.sh predates compare_precision.py's --self-test; wiring it is "
        "part of the same deferred CI workstream (review item N)"
    ),
}


class TestEveryGateIsWired:
    """A checker that exists on disk but is never invoked protects nothing.

    The derived --self-test check only constrains tools CI already calls; it is
    silent about a checker that was written, committed, and then forgotten.
    That is the default state here, not an exotic one: check_symbol_refs.py,
    check_dep_drift.py and check_test_coverage.py all began as one-off review
    scripts living outside the repo.
    """

    def test_every_checker_in_tools_is_invoked_by_ci(self, wf_flat):
        # Discover gates by what they DECLARE, not by how they are spelled. A
        # check_*.py glob would miss a gate named audit_*.py / verify_*.py and
        # report it as wired-and-fine; every gate here offers --self-test, so
        # that string is the self-declaration we key on.
        on_disk = {
            f"tools/{p.name}"
            for p in (REPO_ROOT / "tools").glob("*.py")
            if "--self-test" in p.read_text()
        }
        assert on_disk, "no tools/*.py declares --self-test -- zero-work check"
        invoked = {t for t, _ in _tool_invocations(wf_flat)}
        unwired = (on_disk - invoked) - set(KNOWN_UNWIRED_CHECKERS)
        assert not sorted(unwired), (
            f"{sorted(unwired)} exist in tools/ but CI never runs them -- a gate "
            "nobody invokes protects nothing. If this is deliberate, add it to "
            "KNOWN_UNWIRED_CHECKERS with the reason."
        )
        healed = sorted(t for t in KNOWN_UNWIRED_CHECKERS if t in invoked)
        assert not healed, (
            f"{healed} are now wired into CI -- drop them from "
            "KNOWN_UNWIRED_CHECKERS so the allowlist does not rot"
        )

    def test_local_gates_script_covers_the_checkers_ci_trusts(self, wf_flat):
        """gates.sh is the local mirror of CI; a one-directional containment.

        gates.sh may run extra ad-hoc probes, but it must not MISS a checker CI
        trusts -- otherwise a local green run does not predict the CI verdict.
        """
        gates = REPO_ROOT.parent / "robo_file" / "_ops" / "gates.sh"
        if not gates.exists():
            pytest.skip(f"local gates script not present at {gates}")
        text = gates.read_text()
        ci_tools = {t for t, _ in _tool_invocations(wf_flat) if (REPO_ROOT / t).exists()}
        assert ci_tools, "no existing tools parsed out of CI -- zero-work check"
        missing = sorted(
            t
            for t in ci_tools
            if t not in text and t not in KNOWN_GATES_SCRIPT_GAPS
        )
        assert not missing, (
            f"{missing} run in CI but are absent from gates.sh -- a local green "
            "run would not predict the CI verdict. If deliberate, add it to "
            "KNOWN_GATES_SCRIPT_GAPS with the reason."
        )
        healed = sorted(t for t in KNOWN_GATES_SCRIPT_GAPS if t in text)
        assert not healed, (
            f"{healed} are now in gates.sh -- drop them from "
            "KNOWN_GATES_SCRIPT_GAPS so the allowlist does not rot"
        )


class TestNoVersionDrift:
    """The second defect that shipped: pre-commit pinned twice, differently."""

    def test_pre_commit_pin_matches_dev_requirements(self, wf_text):
        ci = re.search(r"pre-commit==([\d.]+)", wf_text)
        dev = re.search(
            r"pre-commit==([\d.]+)",
            (REPO_ROOT / "requirements" / "dev.txt").read_text(),
        )
        assert ci and dev, "pre-commit must be pinned in both places"
        assert ci.group(1) == dev.group(1), (
            f"pin drift: ci.yml={ci.group(1)} dev.txt={dev.group(1)}"
        )

    def test_baseline_sha_matches_migration_map(self, wf_text):
        wf_shas = set(re.findall(r"\b[0-9a-f]{40}\b", wf_text))
        mm_shas = set(
            re.findall(
                r"\b[0-9a-f]{40}\b",
                (REPO_ROOT / "docs" / "MIGRATION_MAP.md").read_text(),
            )
        )
        assert wf_shas, "CI must pin the FlagScale baseline by full SHA"
        assert wf_shas <= mm_shas, (
            f"CI pins {wf_shas} which MIGRATION_MAP does not record ({mm_shas}); "
            "a drifting baseline silently invalidates the migration scope"
        )

    def test_baseline_is_a_full_sha_not_a_branch(self, wf_text):
        m = re.search(r"ref:\s*(\S+)", wf_text)
        assert m, "the FlagScale checkout must pin a ref"
        assert re.fullmatch(r"[0-9a-f]{40}", m.group(1)), (
            f"ref={m.group(1)!r} is not a 40-char SHA; branches move"
        )


class TestCommandsMatchTheDocs:
    def test_ci_closure_commands_match_the_sop(self, wf_flat):
        """If CI and the documented manual procedure diverge, one of them is
        wrong and nobody finds out.

        There are two closure scans, not one: the training entry points are
        found by --discover-seeds, while the launcher has to be named with
        two explicit --seed flags because nothing under examples/ imports it.
        Comparing only the first command in each file would let the second
        drift unnoticed, so both are collected and compared as a set.
        """
        sop = (REPO_ROOT / "docs" / "MIGRATION_SOP.md").read_text().replace("\\\n", " ")

        def invocations(blob):
            found = []
            for line in blob.splitlines():
                if "check_import_closure.py" in line and "--source-root" in line:
                    # frozenset of flags; values differ (paths, _ci/ prefix)
                    found.append(frozenset(re.findall(r"--[a-z-]+", line)))
            return found

        sop_cmds, ci_cmds = invocations(sop), invocations(wf_flat)
        assert sop_cmds, "SOP has no closure command with --source-root"
        assert ci_cmds, "CI has no closure command with --source-root"
        # The SOP shows the entry-point scan twice -- once in the gate section
        # and once inside the per-batch procedure -- so compare distinct flag
        # shapes, not occurrence counts. Two shapes is the invariant: the
        # --discover-seeds scan and the two-explicit---seed launcher scan.
        assert len(set(sop_cmds)) == 2, (
            f"expected 2 distinct documented closure scans (entry points + "
            f"launcher), found {len(set(sop_cmds))}: "
            f"{[sorted(c) for c in set(sop_cmds)]}"
        )
        assert set(sop_cmds) == set(ci_cmds), (
            f"flag drift: sop-only={[sorted(c) for c in set(sop_cmds) - set(ci_cmds)]} "
            f"ci-only={[sorted(c) for c in set(ci_cmds) - set(sop_cmds)]}"
        )

    def test_ci_runs_both_closure_scans(self, wf_flat):
        """The launcher scan is the one that caught the W1 dead scope entry;
        it must not be dropped from CI."""
        assert "--seed flagscale.runner.runner_train" in wf_flat
        assert "--seed flagscale.runner.backend.backend_native_train" in wf_flat

    def test_referenced_repo_paths_exist(self, wf_flat):
        missing = []
        seen = 0
        for rel in re.findall(r"(?:requirements/[\w./-]+|docs/[\w./-]+)", wf_flat):
            rel = rel.rstrip(".,;:")
            seen += 1
            if not (REPO_ROOT / rel).exists():
                missing.append(rel)
        assert seen > 0, "no repo paths found in the workflow"
        assert not missing, f"workflow references non-existent paths: {missing}"


class TestGpuJobIsSkippableNotBreakable:
    def test_gpu_job_is_conditional(self, wf_text):
        """A fork without self-hosted runners must skip, not fail."""
        block = wf_text.split("tests-gpu:", 1)
        assert len(block) == 2, "tests-gpu job missing"
        assert re.search(r"^\s+if:", block[1], re.M), (
            "tests-gpu has no `if:` guard, so forks will fail instead of skip"
        )

    def test_gpu_job_pins_the_container_image(self, wf_text):
        assert "harbor.baai.ac.cn/flagscale/flagscale:" in wf_text
        assert re.search(r"image:\s*\S+:\S+", wf_text), "image must carry a tag"
