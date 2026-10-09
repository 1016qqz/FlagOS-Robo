# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Regression tests for the raw-tensor (.pt) bit-exactness path.

These exist because of a real escape found by probing the .pt branch with
actual torch tensors: strict mode used to compare only the five summary
statistics (sum/mean/std/max/norm). A 1-ULP delta in a single element leaves
a float32 ``sum`` EXACTLY equal, so the tool printed PASS on a tensor that
was demonstrably not identical.

Every test below is kept in its own module (rather than appended to
test_compare_precision.py) so that the torch requirement is isolated: this
file skips wholesale where torch is absent, while the stat-dump tests stay
runnable anywhere.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "compare_precision.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("_cp_bitexact", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cp_bitexact"] = mod
    spec.loader.exec_module(mod)
    return mod


cp = _load_tool()

torch = pytest.importorskip("torch", reason="raw-tensor path requires torch")


def _dump(d: Path, name: str, t) -> None:
    d.mkdir(parents=True, exist_ok=True)
    torch.save({name: t}, d / "step0.pt")


def _one_ulp_up(t):
    out = t.clone()
    flat = out.flatten()
    flat[0] = torch.nextafter(flat[0], torch.tensor(float("inf"), dtype=t.dtype))
    return flat.reshape(t.shape)


class TestSummaryStatsCannotProveBitExactness:
    """The premise of the whole fix, asserted rather than assumed."""

    def test_float32_sum_is_blind_to_one_ulp(self):
        a = torch.randn(4, 8)
        b = _one_ulp_up(a)
        assert not torch.equal(a, b), "the two tensors must actually differ"
        assert a.sum().item() == b.sum().item(), (
            "premise: a lossy reduction absorbs a 1-ULP single-element delta. "
            "If this ever stops holding, the elementwise path must stay anyway."
        )

    def test_mean_and_norm_are_also_blind(self):
        a = torch.randn(4, 8)
        b = _one_ulp_up(a)
        assert a.mean().item() == b.mean().item()
        assert a.norm().item() == b.norm().item()


class TestStrictCatchesOneUlp:
    def test_one_ulp_fails_strict(self, tmp_path):
        a = torch.randn(4, 8)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a)
        _dump(tgt, "hidden", _one_ulp_up(a))
        res = cp.compare_tensor_dir(ref, tgt, strict=True)
        assert not res.ok, "1-ULP drift must not pass a bit-exact check"
        assert res.findings

    def test_one_ulp_passes_relaxed(self, tmp_path):
        """relaxed exists precisely to tolerate sub-tolerance byte differences."""
        a = torch.randn(4, 8)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a)
        _dump(tgt, "hidden", _one_ulp_up(a))
        assert cp.compare_tensor_dir(ref, tgt, strict=False, rtol=1e-5).ok

    def test_identical_passes_strict(self, tmp_path):
        a = torch.randn(4, 8)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a.clone())
        _dump(tgt, "hidden", a.clone())
        res = cp.compare_tensor_dir(ref, tgt, strict=True)
        assert res.ok, res.render()
        assert not res.vacuous, "identical-but-empty must never read as PASS"

    def test_elementwise_path_is_reported_as_taken(self, tmp_path):
        """Proof the elementwise comparison ran, not silently skipped."""
        a = torch.randn(2, 3)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a.clone())
        _dump(tgt, "hidden", a.clone())
        res = cp.compare_tensor_dir(ref, tgt, strict=True)
        blob = " ".join(res.notes)
        assert "elementwise" in blob, f"notes did not mention elementwise: {res.notes}"

    def test_finding_localises_how_many_elements_differ(self, tmp_path):
        """A bare 'not equal' is not actionable; the count localises it."""
        a = torch.zeros(10)
        b = a.clone()
        b[3] = 1.0
        b[7] = 1.0
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a)
        _dump(tgt, "hidden", b)
        res = cp.compare_tensor_dir(ref, tgt, strict=True)
        assert not res.ok
        blob = " ".join(res.findings)
        assert "2" in blob and "10" in blob, f"count not reported: {res.findings}"


class TestByteDigest:
    """``numpy().tobytes()`` raises on bfloat16 -- the dtype this workload
    actually trains in -- so the digest must go through ``view(torch.uint8)``.
    """

    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool],
    )
    def test_digest_works_for_every_training_dtype(self, dtype):
        t = torch.ones(4, 3, dtype=dtype)
        h = cp.tensor_digest(t)
        assert isinstance(h, str) and len(h) == 64, f"not a sha256 hex: {h!r}"
        assert h == cp.tensor_digest(t.clone()), "digest must be deterministic"

    def test_numpy_tobytes_really_fails_on_bfloat16(self):
        """Documents WHY view(uint8) was chosen over the obvious numpy route."""
        t = torch.ones(2, 2, dtype=torch.bfloat16)
        with pytest.raises(TypeError):
            t.numpy().tobytes()

    def test_digest_distinguishes_one_ulp(self):
        a = torch.randn(4, 8)
        assert cp.tensor_digest(a) != cp.tensor_digest(_one_ulp_up(a))

    def test_digest_distinguishes_reshape(self):
        """Payload bytes are identical after a reshape, so shape must be mixed
        into the digest or the two collide."""
        a = torch.randn(4, 8)
        assert cp.tensor_digest(a) != cp.tensor_digest(a.reshape(8, 4))

    def test_digest_distinguishes_dtype(self):
        a = torch.ones(4, 8, dtype=torch.float32)
        assert cp.tensor_digest(a) != cp.tensor_digest(a.to(torch.bfloat16))

    def test_digest_survives_non_contiguous(self):
        """A transposed view has the same storage; the digest must reflect the
        logical tensor, not the raw storage order."""
        a = torch.randn(4, 8)
        t = a.t()
        assert not t.is_contiguous()
        assert cp.tensor_digest(t) == cp.tensor_digest(t.contiguous())


class TestStructuralDifferencesOnRawTensors:
    def test_dtype_change_caught(self, tmp_path):
        a = torch.randn(4, 8)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a)
        _dump(tgt, "hidden", a.to(torch.bfloat16))
        assert not cp.compare_tensor_dir(ref, tgt, strict=True).ok

    def test_shape_change_caught(self, tmp_path):
        a = torch.randn(4, 8)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a)
        _dump(tgt, "hidden", a.reshape(8, 4))
        assert not cp.compare_tensor_dir(ref, tgt, strict=True).ok

    def test_bfloat16_identical_passes_strict(self, tmp_path):
        """bf16 is the real training dtype, so it gets its own end-to-end case."""
        a = torch.randn(4, 8).to(torch.bfloat16)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a.clone())
        _dump(tgt, "hidden", a.clone())
        res = cp.compare_tensor_dir(ref, tgt, strict=True)
        assert res.ok, res.render()

    def test_bfloat16_one_ulp_caught_strict(self, tmp_path):
        a = torch.randn(4, 8).to(torch.bfloat16)
        ref, tgt = tmp_path / "ref", tmp_path / "tgt"
        _dump(ref, "hidden", a)
        _dump(tgt, "hidden", _one_ulp_up(a))
        assert not cp.compare_tensor_dir(ref, tgt, strict=True).ok
