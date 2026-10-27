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
"""Unit tests for tools/compare_precision.py.

These are the regression layer on top of the tool's own ``--self-test``. The
self-test proves each level can detect a planted defect; these tests pin the
behaviours that would otherwise be easy to "fix" into uselessness later:

* vacuous input must never be reported as PASS,
* strict mode must mean bit-exact (1 ULP is a failure),
* the L2 three-way triage must keep its three distinct classes,
* every divergence shape must stay distinguishable,
* exit codes must stay 0 / 1 / 2.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "compare_precision.py"

sys.path.insert(0, str(TOOL.parent))

import compare_precision as cp


def run_tool(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


class TestToolExists:
    def test_tool_file_present(self):
        assert TOOL.is_file(), f"missing deliverable: {TOOL}"

    def test_selftest_passes(self):
        """The tool's own planted-defect suite must be green."""
        res = run_tool("--self-test")
        assert res.returncode == 0, res.stdout + res.stderr
        assert "SELF-TEST PASSED" in res.stdout

    def test_no_args_is_usage_error(self):
        assert run_tool().returncode == 2

    def test_determinism_snippet_has_the_ordering_trap(self):
        """CUBLAS_WORKSPACE_CONFIG must be set before torch is imported."""
        res = run_tool("--emit-determinism-snippet")
        assert res.returncode == 0
        out = res.stdout
        assert "CUBLAS_WORKSPACE_CONFIG" in out
        assert "use_deterministic_algorithms" in out
        assert out.index("CUBLAS_WORKSPACE_CONFIG") < out.index("import torch"), (
            "snippet must set CUBLAS_WORKSPACE_CONFIG before importing torch, "
            "otherwise it silently has no effect"
        )


class TestVacuityIsNotPass:
    """A diffing tool that returns PASS on nothing is worse than no tool."""

    def test_empty_param_tables(self):
        assert not cp.compare_params("", "").ok

    def test_empty_configs(self):
        assert not cp.compare_config({}, {}).ok

    def test_empty_loss_series(self):
        assert not cp.compare_loss({}, {}).ok

    def test_empty_tensor_dirs_via_cli(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        res = run_tool("tensors", "--ref", str(a), "--tgt", str(b))
        assert res.returncode == 1, res.stdout
        assert "VACUOUS" in res.stdout

    def test_vacuous_flag_is_set_not_just_a_finding(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        assert cp.compare_tensor_dir(a, b).vacuous is True


class TestStrictMeansBitExact:
    def _dirs(self, tmp_path, ref_sum, tgt_sum):
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        for d, s in ((ref, ref_sum), (tgt, tgt_sum)):
            d.mkdir()
            (d / "step0.json").write_text(
                json.dumps(
                    {
                        "x": {
                            "shape": [2],
                            "dtype": "torch.float32",
                            "sum": s,
                            "mean": 1.0,
                        }
                    }
                ),
                encoding="utf-8",
            )
        return ref, tgt

    def test_one_ulp_fails_strict(self, tmp_path):
        import math

        ref, tgt = self._dirs(tmp_path, 1.0, math.nextafter(1.0, math.inf))
        assert not cp.compare_tensor_dir(ref, tgt, strict=True).ok

    def test_one_ulp_passes_relaxed(self, tmp_path):
        import math

        ref, tgt = self._dirs(tmp_path, 1.0, math.nextafter(1.0, math.inf))
        assert cp.compare_tensor_dir(ref, tgt, strict=False, rtol=1e-6).ok

    def test_identical_passes_strict(self, tmp_path):
        ref, tgt = self._dirs(tmp_path, 1.0, 1.0)
        assert cp.compare_tensor_dir(ref, tgt, strict=True).ok

    def test_cli_defaults_to_strict(self, tmp_path):
        """--strict is the DEFAULT: a verbatim migration has no excuse for drift."""
        import math

        ref, tgt = self._dirs(tmp_path, 1.0, math.nextafter(1.0, math.inf))
        res = run_tool("tensors", "--ref", str(ref), "--tgt", str(tgt))
        assert res.returncode == 1, f"default mode was not strict: {res.stdout}"
        res = run_tool("tensors", "--ref", str(ref), "--tgt", str(tgt), "--relaxed")
        assert res.returncode == 0, res.stdout


class TestL1Structure:
    def test_transposed_shape_caught_despite_equal_numel(self):
        """Same element count, different shape: the total-count check alone
        would wave this through."""
        res = cp.compare_params("w torch.Size([4, 8])", "w torch.Size([8, 4])")
        assert not res.ok
        assert any("shape mismatch" in f for f in res.findings)

    def test_fusion_mapping_applies(self):
        """q/k/v -> qkv is expressible without editing either dump."""
        ref = "l.qkv.weight (12, 4)"
        tgt = "l.in_proj.weight (12, 4)"
        assert not cp.compare_params(ref, tgt).ok
        mapped = cp.compare_params(ref, tgt, {"l.qkv.weight": "l.in_proj.weight"})
        assert mapped.ok, mapped.render()

    @pytest.mark.parametrize(
        "line, expected",
        [
            ("w torch.Size([4, 8])", (4, 8)),
            ("w (4, 8)", (4, 8)),
            ("w 4x8", (4, 8)),
            ("w [4, 8]", (4, 8)),
        ],
    )
    def test_shape_formats_all_parse(self, line, expected):
        assert cp.parse_param_table(line)["w"] == expected

    def test_comments_and_blanks_ignored(self):
        table = "# header\n\nw (2, 2)\n  \n"
        assert cp.parse_param_table(table) == {"w": (2, 2)}


class TestL2Triage:
    """The three classes must stay three. Collapsing them is how a real bug
    gets waved through as 'just numerics'."""

    def test_epsilon_is_semantic(self):
        assert cp.classify_difference("optimizer.eps") == cp.SEMANTIC_DIFFERENT
        assert cp.classify_difference("adam_epsilon") == cp.SEMANTIC_DIFFERENT
        assert cp.classify_difference("layernorm_epsilon") == cp.SEMANTIC_DIFFERENT

    def test_distributed_optimizer_is_math_equivalent(self):
        assert cp.classify_difference("use_distributed_optimizer") == cp.MATH_EQUIVALENT

    def test_flash_attn_is_numeric_different(self):
        assert cp.classify_difference("use_flash_attn") == cp.NUMERIC_DIFFERENT

    def test_unknown_key_defaults_to_semantic(self):
        """Fail closed: an unrecognised key is treated as must-fix, never
        silently tolerated."""
        assert cp.classify_difference("some.novel_flag") == cp.SEMANTIC_DIFFERENT

    def test_math_equivalent_is_a_note_not_a_finding(self):
        a = {
            "use_distributed_optimizer": False,
            "global_batch_size": 8,
            "micro_batch_size": 2,
            "data_parallel_size": 4,
            "gradient_accumulation_steps": 1,
        }
        b = dict(a, use_distributed_optimizer=True)
        res = cp.compare_config(a, b)
        assert res.ok, res.render()
        assert any(cp.MATH_EQUIVALENT in n for n in res.notes)

    def test_numeric_different_is_a_finding_in_this_scenario(self):
        """Identical code on identical hardware must not pick a different
        kernel, so a numeric-different delta is a defect here."""
        a = {
            "use_flash_attn": False,
            "global_batch_size": 8,
            "micro_batch_size": 2,
            "data_parallel_size": 4,
            "gradient_accumulation_steps": 1,
        }
        b = dict(a, use_flash_attn=True)
        assert not cp.compare_config(a, b).ok


class TestBatchIdentity:
    BASE = {
        "global_batch_size": 64,
        "micro_batch_size": 2,
        "world_size": 8,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "gradient_accumulation_steps": 8,
    }

    def test_consistent_config_has_no_problem(self):
        assert cp.check_batch_identity(self.BASE, "ref") == []

    def test_violated_identity_is_reported(self):
        bad = dict(self.BASE, gradient_accumulation_steps=7)
        problems = cp.check_batch_identity(bad, "ref")
        assert problems and "batch identity" in problems[0]

    def test_dp_derived_from_world_tp_pp(self):
        """DP is inferred as world/(TPxPP) -- the step most often miscomputed
        when parallelism changes."""
        cfg = dict(self.BASE)
        cfg.pop("gradient_accumulation_steps")
        assert cp.check_batch_identity(cfg, "ref") == []

    def test_indivisible_world_size_is_reported(self):
        cfg = dict(self.BASE, world_size=7)
        cfg.pop("gradient_accumulation_steps")
        problems = cp.check_batch_identity(cfg, "ref")
        assert problems and "not divisible" in problems[0]

    def test_violation_caught_even_when_ref_equals_tgt(self):
        """Both sides can be self-inconsistent in the same way: a pure diff
        would report 'identical' and miss it."""
        bad = dict(self.BASE, gradient_accumulation_steps=7)
        assert not cp.compare_config(bad, bad).ok


class TestL5DivergenceShapes:
    """Every shape must stay distinguishable from every other. A classifier
    that collapses two shapes sends you to the wrong level and you re-debug
    an already-correct layer."""

    STEPS = list(range(8))
    REF = [10.0 - 0.5 * i for i in STEPS]

    def test_step_zero_mismatch(self):
        tgt = [v + 2.0 for v in self.REF]
        shape, _ = cp.classify_divergence(self.STEPS, self.REF, tgt)
        assert shape == cp.SHAPE_STEP_ZERO

    def test_linear_growth_from_zero_delta(self):
        """Regression: this is THE realistic shape here -- bit-exact init
        (delta=0 at step 0) then accumulating drift. An earlier version
        anchored on deltas[0] and misfiled it as noise."""
        tgt = [v + 0.01 * i for i, v in enumerate(self.REF)]
        shape, _ = cp.classify_divergence(self.STEPS, self.REF, tgt)
        assert shape == cp.SHAPE_LINEAR

    def test_exponential_growth(self):
        tgt = [v + 1e-4 * (3.0**i) for i, v in enumerate(self.REF)]
        shape, _ = cp.classify_divergence(self.STEPS, self.REF, tgt)
        assert shape == cp.SHAPE_EXPONENTIAL

    def test_one_flat_one_descending(self):
        shape, _ = cp.classify_divergence(self.STEPS, self.REF, [10.0] * 8)
        assert shape == cp.SHAPE_ONE_FLAT

    def test_linear_and_exponential_are_not_conflated(self):
        lin = [v + 0.01 * i for i, v in enumerate(self.REF)]
        exp = [v + 1e-4 * (3.0**i) for i, v in enumerate(self.REF)]
        s_lin, _ = cp.classify_divergence(self.STEPS, self.REF, lin)
        s_exp, _ = cp.classify_divergence(self.STEPS, self.REF, exp)
        assert s_lin != s_exp

    def test_identical_curves_are_noise_and_pass(self):
        shape, stats = cp.classify_divergence(self.STEPS, self.REF, list(self.REF))
        assert shape == cp.SHAPE_NOISE
        assert stats["max_abs_delta"] == 0.0

    def test_every_shape_has_an_action(self):
        """A diagnosis with no prescribed next step is not actionable."""
        for shape in (
            cp.SHAPE_STEP_ZERO,
            cp.SHAPE_LINEAR,
            cp.SHAPE_EXPONENTIAL,
            cp.SHAPE_ONE_FLAT,
            cp.SHAPE_SPIKE,
            cp.SHAPE_PERIODIC,
            cp.SHAPE_NOISE,
        ):
            assert shape in cp.SHAPE_ACTIONS
            assert cp.SHAPE_ACTIONS[shape].strip()

    def test_nonfinite_target_loss_is_caught(self):
        ref = dict(zip(self.STEPS, self.REF))
        tgt = dict(ref)
        tgt[3] = float("nan")
        res = cp.compare_loss(ref, tgt)
        assert not res.ok
        assert any("not finite" in f for f in res.findings)

    def test_missing_steps_reported(self):
        ref = dict(zip(self.STEPS, self.REF))
        tgt = {k: v for k, v in ref.items() if k != 5}
        res = cp.compare_loss(ref, tgt)
        assert not res.ok
        assert any("only in REF" in f for f in res.findings)


class TestLossSeriesFormats:
    def test_dict_form(self, tmp_path):
        p = tmp_path / "l.json"
        p.write_text(json.dumps({"0": 1.5, "1": 1.4}), encoding="utf-8")
        assert cp.load_loss_series(p) == {0: 1.5, 1: 1.4}

    def test_records_form(self, tmp_path):
        p = tmp_path / "l.json"
        p.write_text(
            json.dumps([{"step": 0, "loss": 1.5}, {"step": 1, "loss": 1.4}]),
            encoding="utf-8",
        )
        assert cp.load_loss_series(p) == {0: 1.5, 1: 1.4}

    def test_nested_losses_key(self, tmp_path):
        p = tmp_path / "l.json"
        p.write_text(json.dumps({"losses": {"3": 2.0}}), encoding="utf-8")
        assert cp.load_loss_series(p) == {3: 2.0}

    def test_lm_loss_alias(self, tmp_path):
        p = tmp_path / "l.json"
        p.write_text(json.dumps([{"iteration": 7, "lm_loss": 3.25}]), encoding="utf-8")
        assert cp.load_loss_series(p) == {7: 3.25}


class TestCLIContract:
    def test_json_output_is_parseable(self, tmp_path):
        """--json must emit ONLY json: no banner line ahead of it."""
        ref = tmp_path / "r.json"
        tgt = tmp_path / "t.json"
        ref.write_text(json.dumps({"0": 1.0, "1": 0.9}), encoding="utf-8")
        tgt.write_text(json.dumps({"0": 1.0, "1": 0.9}), encoding="utf-8")
        res = run_tool("loss", "--ref", str(ref), "--tgt", str(tgt), "--json")
        assert res.returncode == 0, res.stdout + res.stderr
        payload = json.loads(res.stdout)
        assert payload["ok"] is True
        assert payload["vacuous"] is False

    def test_missing_file_exits_2_not_1(self, tmp_path):
        """An input problem must be distinguishable from 'found a difference'."""
        res = run_tool(
            "loss",
            "--ref",
            str(tmp_path / "nope.json"),
            "--tgt",
            str(tmp_path / "nope2.json"),
        )
        assert res.returncode == 2, res.stdout + res.stderr

    def test_malformed_json_exits_2(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        res = run_tool("loss", "--ref", str(bad), "--tgt", str(bad))
        assert res.returncode == 2, res.stdout + res.stderr

    def test_difference_exits_1(self, tmp_path):
        ref = tmp_path / "r.json"
        tgt = tmp_path / "t.json"
        ref.write_text(json.dumps({"0": 1.0, "1": 0.9}), encoding="utf-8")
        tgt.write_text(json.dumps({"0": 1.0, "1": 2.0}), encoding="utf-8")
        assert run_tool("loss", "--ref", str(ref), "--tgt", str(tgt)).returncode == 1
