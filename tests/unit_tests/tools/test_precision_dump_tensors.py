# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tensor-stats half of the precision-dump contract. Requires torch.

Split out from ``test_precision_dump.py`` on the same principle as
``test_raw_tensor_bitexact.py``: the torch requirement is isolated so the
stdlib-only tests stay runnable on a laptop with no torch, which is where
this code is edited.

The central test here is :func:`test_digest_matches_the_comparers_digest`.
``precision_dump.tensor_digest`` is a deliberate copy of the comparer's
implementation -- ``tools/`` is a script directory, not an importable
package, and a training entrypoint must not depend on it. A copy can drift,
and if it drifts the two sides produce different digests for identical
tensors, so every strict comparison reports a difference that is not real.
That failure is invisible in any test that exercises only one side; it needs
both implementations run against the same tensor.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "compare_precision.py"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flagos_robo.utils import precision_dump as pd

torch = pytest.importorskip("torch", reason="tensor dump path requires torch")


def _load_tool():
    """Load the comparer as a module -- it is a script, not an installed package."""
    spec = importlib.util.spec_from_file_location("_cp_for_dump", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cp_for_dump"] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load_tool()


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: torch.arange(12, dtype=torch.float32).reshape(3, 4), id="f32"),
        pytest.param(lambda: torch.ones(5, dtype=torch.bfloat16), id="bf16"),
        pytest.param(lambda: torch.zeros(2, 3, dtype=torch.float16), id="f16"),
        pytest.param(lambda: torch.arange(6, dtype=torch.int64), id="i64"),
        pytest.param(lambda: torch.tensor([True, False, True]), id="bool"),
        pytest.param(lambda: torch.empty(0, dtype=torch.float32), id="empty"),
        pytest.param(lambda: torch.tensor([2.5], dtype=torch.float32), id="single"),
    ],
)
def test_digest_matches_the_comparers_digest(factory):
    """The two copies of the digest must agree, across every training dtype.

    bfloat16 is in the list for a specific reason: it is the training dtype
    and numpy cannot represent it, so an implementation that reached for
    ``numpy().tobytes()`` instead of ``view(torch.uint8)`` raises TypeError
    here rather than returning a wrong answer.
    """
    tensor = factory()

    assert pd.tensor_digest(tensor) == cp.tensor_digest(tensor)


def test_stats_fields_match_the_comparers_reader(tmp_path):
    """A dumped stats file must expose exactly the keys the comparer reads.

    An extra key would be ignored, but a MISSING or renamed one silently
    drops that statistic from the comparison -- the comparison still prints
    PASS, just on less evidence than it claims.
    """
    out = tmp_path / "ref"
    pd.dump_tensors(out, 0, {"logits": torch.arange(8, dtype=torch.float32)})

    dumped = json.loads((out / pd.step_relpath(0)).read_text(encoding="utf-8"))
    stats = dumped["logits"]
    reference = cp.load_tensor_stats(out / pd.step_relpath(0))["logits"]

    assert set(stats) == set(pd.STATS_FIELDS)
    assert set(reference) == set(pd.STATS_FIELDS)
    assert stats == reference


def test_empty_and_single_element_tensors_do_not_produce_nan(tmp_path):
    """NaN never equals itself, so a NaN stat makes a tensor differ from its own copy.

    torch returns NaN for ``mean`` of an empty tensor and for ``std`` of a
    single element, which is why both are special-cased to 0.0 on both sides.
    """
    out = tmp_path / "ref"
    pd.dump_tensors(
        out,
        0,
        {"empty": torch.empty(0), "single": torch.tensor([3.0])},
    )
    dumped = json.loads((out / pd.step_relpath(0)).read_text(encoding="utf-8"))

    assert dumped["empty"]["mean"] == 0.0
    assert dumped["empty"]["max"] == 0.0
    assert dumped["single"]["std"] == 0.0
    for record in (dumped["empty"], dumped["single"]):
        for field in ("sum", "mean", "std", "max", "norm"):
            assert record[field] == record[field], f"{field} is NaN"


def test_one_ulp_change_is_caught_by_the_digest(tmp_path):
    """The escape the digest field exists for.

    Summary statistics are lossy: perturbing one element of a float32 tensor
    by 1 ULP can leave ``sum`` exactly equal at float32 precision, so a
    stats-only strict comparison reports PASS on tensors that are not
    identical. The byte digest is what actually detects it.
    """
    base = torch.arange(32, dtype=torch.float32)
    nudged = base.clone()
    nudged[7] = torch.nextafter(nudged[7], torch.tensor(float("inf")))

    assert not torch.equal(base, nudged)
    assert pd.tensor_digest(base) != pd.tensor_digest(nudged)


def test_dumped_tree_is_paired_and_compared_by_the_real_tool(tmp_path):
    """End to end: two dump trees, compared by the comparer as a subprocess.

    Identical tensors must PASS and a perturbed one must FAIL. Both halves
    matter -- a tool that always passed would satisfy the first assertion.
    """
    ref_dir, tgt_dir = tmp_path / "ref", tmp_path / "tgt"
    for step in range(3):
        tensors = {"logits": torch.arange(8, dtype=torch.float32) + step}
        pd.dump_tensors(ref_dir, step, tensors)
        pd.dump_tensors(tgt_dir, step, tensors)

    def compare():
        return subprocess.run(
            [
                sys.executable,
                str(TOOL_PATH),
                "tensors",
                "--ref",
                str(ref_dir),
                "--tgt",
                str(tgt_dir),
                "--strict",
            ],
            capture_output=True,
            text=True,
        )

    passing = compare()
    assert passing.returncode == 0, passing.stdout + passing.stderr

    # Perturb one step on the target side; the pairing is by relative path,
    # so this lands on the same file the comparer pairs with ref step 1.
    pd.dump_tensors(tgt_dir, 1, {"logits": torch.arange(8, dtype=torch.float32) + 99})
    failing = compare()
    assert failing.returncode != 0
    assert "step_000001" in failing.stdout


def test_relative_layout_contains_no_host_or_time_dependent_names(tmp_path):
    """Pairing is by path relative to each root, so the layout must be reproducible.

    A hostname, timestamp, or framework name in the path would differ between
    the reference and target sides and every file would land as an unpaired
    MISS -- which reads as "nothing compared" rather than as an error.
    """
    out = tmp_path / "ref"
    path = pd.dump_tensors(out, 42, {"x": torch.zeros(2)}, rank=0)

    assert path.relative_to(out).as_posix() == "step_000042/rank_0.json"
