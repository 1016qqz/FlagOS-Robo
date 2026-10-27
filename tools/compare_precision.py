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
"""Precision alignment comparator for FlagScale -> FlagOS-Robo migration.

Implements the comparison half of the 6-level alignment ladder documented in
``docs/PRECISION_BASELINE.md``. The *instrumentation* half lives in the training
loops themselves (see that doc, section 4): this tool only compares artifacts
that both sides dumped. That split is deliberate - a standalone script that
"simulates" training cannot reproduce dataloader order, DP rank sharding or
grad-accumulation boundaries, so any agreement it reports would be fiction.

Why bit-exact is the default here
---------------------------------
FlagOS-Robo migrates FlagScale code **verbatim** and runs on the **same** H800
box. Framework, kernels and hardware are all identical; only import paths
change. So unlike a normal cross-framework port, there is no legitimate source
of numerical difference, and any nonzero delta means the migration broke
something. ``--strict`` (atol=0, rtol=0) is therefore the default for L3/L4/L5.

Subcommands
-----------
params   L1 - parameter tables: 1:1 mapping + identical element totals
config   L2 - hyperparameters: batch-size identity + 3-way difference triage
tensors  L3/L4/L6 - dumped tensor stats, bit-exact by default
loss     L5 - per-step loss, plus divergence-SHAPE classification

Exit codes: 0 aligned / 1 difference found / 2 usage or input problem.

Pure stdlib. ``torch`` is imported lazily and only when a ``.pt`` dump is
actually read, so ``params``/``config``/``loss`` and ``--self-test`` all run on
a laptop with no torch and no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Difference triage (L2). Straight from the standard methodology: not every
# difference is a bug, and collapsing these three classes is how real bugs get
# waved through as "just numerics".
# --------------------------------------------------------------------------

#: Mathematically equivalent: same result, different execution. Record only.
MATH_EQUIVALENT_KEYS = frozenset(
    {
        "zero_stage",
        "use_distributed_optimizer",
        "overlap_grad_reduce",
        "overlap_param_gather",
        "gradient_accumulation_fusion",
        "sequence_parallel",
    }
)

#: Numerically different: changes rounding, not semantics. Tolerated in a
#: generic port -- but NOT here, because identical code on identical hardware
#: should not hit a different kernel. Flagged as a defect unless the batch is
#: registered in MIGRATION_MAP.md as "implementation rewritten".
NUMERIC_DIFFERENT_KEYS = frozenset(
    {
        "use_flash_attn",
        "attention_backend",
        "transformer_impl",
        "fp8",
        "fp8_format",
        "apply_rope_fusion",
        "bias_activation_fusion",
        "masked_softmax_fusion",
    }
)

#: Keys whose value is a tolerance/epsilon. A difference here is SEMANTIC: it
#: changes the maths, not just the rounding. Always must-fix.
_EPSILON_KEY_RE = re.compile(r"(eps|epsilon|tolerance)$", re.IGNORECASE)

MATH_EQUIVALENT = "math-equivalent"
NUMERIC_DIFFERENT = "numeric-different"
SEMANTIC_DIFFERENT = "semantic-different"


def classify_difference(key: str) -> str:
    """Triage one differing config key into the 3 standard classes."""
    leaf = key.rsplit(".", 1)[-1]
    if _EPSILON_KEY_RE.search(leaf):
        return SEMANTIC_DIFFERENT
    if leaf in MATH_EQUIVALENT_KEYS:
        return MATH_EQUIVALENT
    if leaf in NUMERIC_DIFFERENT_KEYS:
        return NUMERIC_DIFFERENT
    return SEMANTIC_DIFFERENT


# --------------------------------------------------------------------------
# Determinism snippet (prerequisite for strict L3-L5)
# --------------------------------------------------------------------------

DETERMINISM_SNIPPET = """\
# --- strict-alignment determinism block -------------------------------------
# CUBLAS_WORKSPACE_CONFIG MUST be set before CUDA initialises. Setting it after
# the first CUDA call silently does nothing -- a real trap.
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
# Also required: attention_dropout=0 and hidden_dropout=0 in the model config.
# ---------------------------------------------------------------------------
"""


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------


@dataclass
class Result:
    """One level's verdict. ``findings`` are failures, ``notes`` are FYI."""

    level: str
    findings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Set when the comparison had nothing to compare. A vacuous PASS is the
    #: single most dangerous failure mode of a diffing tool, so it is tracked
    #: explicitly rather than being allowed to look like success.
    vacuous: bool = False

    @property
    def ok(self) -> bool:
        return not self.findings and not self.vacuous

    def render(self) -> str:
        lines = [f"[{self.level}]"]
        for n in self.notes:
            lines.append(f"  . {n}")
        for f in self.findings:
            lines.append(f"  ! {f}")
        if self.vacuous:
            lines.append("  ! VACUOUS: nothing was actually compared")
        lines.append(f"  -> {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# L1 - parameter table
# --------------------------------------------------------------------------

_PARAM_LINE_RE = re.compile(r"^\s*(?P<name>[\w.\[\]]+)\s+(?P<shape>[\d,x\s()\[\]]+?)\s*$")


def parse_param_table(text: str) -> dict[str, tuple[int, ...]]:
    """Parse ``name  shape`` lines, as printed from ``named_parameters()``.

    Accepts ``a.b.weight torch.Size([4, 8])``, ``a.b.weight (4, 8)`` and
    ``a.b.weight 4x8``. Blank lines and ``#`` comments are ignored.
    """
    out: dict[str, tuple[int, ...]] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        line = line.replace("torch.Size", "")
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name, shape_s = parts
        nums = re.findall(r"\d+", shape_s)
        if not nums:
            continue
        out[name] = tuple(int(n) for n in nums)
    return out


def numel(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def compare_params(ref_text: str, tgt_text: str, mapping: dict[str, str] | None = None) -> Result:
    """L1: every parameter must map 1:1 and total element count must match."""
    res = Result("L1 model structure")
    ref = parse_param_table(ref_text)
    tgt = parse_param_table(tgt_text)

    if not ref and not tgt:
        res.vacuous = True
        return res

    mapping = mapping or {}
    # Apply the fusion/rename map to the reference side, so e.g. separate
    # q/k/v -> fused qkv is expressible without editing the dumps.
    mapped_ref: dict[str, tuple[int, ...]] = {}
    for name, shape in ref.items():
        mapped_ref[mapping.get(name, name)] = shape

    ref_only = sorted(set(mapped_ref) - set(tgt))
    tgt_only = sorted(set(tgt) - set(mapped_ref))
    for name in ref_only:
        res.findings.append(f"only in REF (unmapped): {name} {mapped_ref[name]}")
    for name in tgt_only:
        res.findings.append(f"only in TGT (unexpected): {name} {tgt[name]}")

    for name in sorted(set(mapped_ref) & set(tgt)):
        if mapped_ref[name] != tgt[name]:
            res.findings.append(f"shape mismatch {name}: ref={mapped_ref[name]} tgt={tgt[name]}")

    ref_total = sum(numel(s) for s in ref.values())
    tgt_total = sum(numel(s) for s in tgt.values())
    res.notes.append(f"ref params={len(ref)} elements={ref_total}")
    res.notes.append(f"tgt params={len(tgt)} elements={tgt_total}")
    if ref_total != tgt_total:
        res.findings.append(
            f"total element count differs: ref={ref_total} tgt={tgt_total} "
            f"(delta={tgt_total - ref_total})"
        )
    return res


# --------------------------------------------------------------------------
# L2 - hyperparameters
# --------------------------------------------------------------------------


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted keys. Lists are compared whole."""
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(flatten(v, key))
            else:
                out[key] = v
    else:
        out[prefix] = obj
    return out


def load_config(path: Path) -> dict[str, Any]:
    """Load a config dump. JSON always; YAML only if pyyaml is present.

    Instrumentation should dump JSON precisely so this stays dependency-free.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".json"}:
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - env dependent
        raise SystemExit(
            f"error: {path} needs pyyaml to parse; dump configs as .json instead"
        ) from exc
    return yaml.safe_load(text)


def check_batch_identity(cfg: dict[str, Any], side: str) -> list[str]:
    """Verify global = micro x DP x grad_accum, the single most error-prone
    equation when parallelism changes. Returns problem strings (empty = ok)."""
    flat = flatten(cfg)

    def get(*names: str) -> Any:
        for n in names:
            for k, v in flat.items():
                if k.rsplit(".", 1)[-1] == n:
                    return v
        return None

    gbs = get("global_batch_size")
    mbs = get("micro_batch_size")
    world = get("world_size")
    tp = get("tensor_model_parallel_size", "tensor_parallel_size") or 1
    pp = get("pipeline_model_parallel_size", "pipeline_parallel_size") or 1
    accum = get("gradient_accumulation_steps", "grad_accum_steps")

    problems: list[str] = []
    if gbs is None or mbs is None:
        return [f"{side}: cannot verify batch identity (missing global/micro)"]
    dp = get("data_parallel_size")
    if dp is None and world is not None:
        denom = int(tp) * int(pp)
        if denom == 0 or int(world) % denom != 0:
            problems.append(f"{side}: world_size={world} not divisible by TPxPP={denom}")
            return problems
        dp = int(world) // denom
    if dp is None:
        return [f"{side}: cannot verify batch identity (no DP and no world_size)"]
    if accum is None:
        if int(mbs) * int(dp) == 0 or int(gbs) % (int(mbs) * int(dp)) != 0:
            problems.append(
                f"{side}: global={gbs} not divisible by micro x DP={int(mbs) * int(dp)}"
            )
            return problems
        accum = int(gbs) // (int(mbs) * int(dp))
    lhs, rhs = int(gbs), int(mbs) * int(dp) * int(accum)
    if lhs != rhs:
        problems.append(
            f"{side}: batch identity violated: global={lhs} != "
            f"micro({mbs}) x DP({dp}) x accum({accum})={rhs}"
        )
    return problems


def compare_config(ref: dict[str, Any], tgt: dict[str, Any]) -> Result:
    """L2: triage every differing key; batch identity must hold on both sides."""
    res = Result("L2 hyperparameters")
    rf, tf = flatten(ref), flatten(tgt)
    if not rf and not tf:
        res.vacuous = True
        return res

    for side, cfg in (("ref", ref), ("tgt", tgt)):
        for p in check_batch_identity(cfg, side):
            res.findings.append(p)

    ref_gbs = flatten(ref).get("global_batch_size")
    tgt_gbs = flatten(tgt).get("global_batch_size")
    if ref_gbs is not None and tgt_gbs is not None and ref_gbs != tgt_gbs:
        res.findings.append(f"global_batch_size differs: ref={ref_gbs} tgt={tgt_gbs}")

    only_ref = sorted(set(rf) - set(tf))
    only_tgt = sorted(set(tf) - set(rf))
    for k in only_ref:
        res.findings.append(f"key only in REF: {k}={rf[k]!r}")
    for k in only_tgt:
        res.findings.append(f"key only in TGT: {k}={tf[k]!r}")

    for k in sorted(set(rf) & set(tf)):
        if rf[k] == tf[k]:
            continue
        cls = classify_difference(k)
        msg = f"{cls}: {k}: ref={rf[k]!r} tgt={tf[k]!r}"
        if cls == MATH_EQUIVALENT:
            res.notes.append(msg + "  (record only)")
        else:
            # numeric-different is a finding HERE: identical code on identical
            # hardware must not select a different kernel. See the doc.
            res.findings.append(msg)
    return res


# --------------------------------------------------------------------------
# L3 / L4 / L6 - tensor dumps
# --------------------------------------------------------------------------


def load_tensor_stats(path: Path) -> dict[str, dict[str, Any]]:
    """Read one dump file into ``{tensor_name: stats}``.

    Two formats, both produced by ``dump_step``:

    * ``.json`` - precomputed stats (sum/mean/std/max/norm/shape/dtype).
      Preferred: no torch needed, so comparison runs anywhere.
    * ``.pt`` - raw tensors. torch imported lazily, only here.
    """
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {k: v for k, v in raw.items() if isinstance(v, dict)}
    if path.suffix in {".pt", ".pth", ".bin"}:
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                f"error: reading {path.name} needs torch; dump .json stats "
                f"instead to compare without torch"
            ) from exc
        blob = torch.load(path, map_location="cpu", weights_only=True)
        out: dict[str, dict[str, Any]] = {}
        for name, t in blob.items():
            ft = t.float()
            out[name] = {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "sum": ft.sum().item(),
                "mean": ft.mean().item() if t.numel() else 0.0,
                "std": ft.std().item() if t.numel() > 1 else 0.0,
                "max": ft.max().item() if t.numel() else 0.0,
                "norm": ft.norm().item(),
                # Bit-exactness CANNOT be decided from summary statistics: a
                # 1-ULP change in one element is absorbed by a float32 sum.
                # The digest is over the raw bytes, so it does detect it.
                "sha256": tensor_digest(t),
            }
        return out
    raise SystemExit(f"error: unsupported dump format: {path}")


def tensor_digest(t: Any) -> str:
    """SHA-256 over a tensor's raw bytes -- a bit-exactness fingerprint.

    Why this exists: summary statistics are lossy. Measured on real tensors --
    perturbing one element of a 32-element float32 tensor by 1 ULP left
    ``sum`` *exactly* equal at float32 precision, so a stats-only strict
    comparison printed PASS on a tensor that was NOT bit-identical. The byte
    digest caught it. Observed, not theorised.

    ``view(torch.uint8)`` is used rather than ``numpy().tobytes()`` because
    numpy has no bfloat16: measured, ``numpy()`` raises
    ``TypeError: Got unsupported ScalarType BFloat16``, and bf16 is the
    training dtype here. ``view(uint8)`` returned the correct byte count for
    float32 / bfloat16 / float16 / int64 / bool.
    """
    import torch  # type: ignore

    c = t.detach().cpu().contiguous().flatten()
    h = hashlib.sha256()
    # Shape and dtype join the digest so a reshape or a dtype change cannot
    # collide with identical payload bytes.
    h.update(f"{tuple(t.shape)}|{t.dtype}|".encode())
    if c.numel():
        h.update(c.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_raw_tensors(path: Path) -> dict[str, Any] | None:
    """Load raw tensors from a ``.pt`` dump, or ``None`` for stats-only dumps.

    When both sides provide raw tensors the comparison is elementwise, which is
    the only way to honour a bit-exact claim.
    """
    if path.suffix not in {".pt", ".pth", ".bin"}:
        return None
    import torch  # type: ignore

    blob = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict):
        return None
    return {k: v for k, v in blob.items() if hasattr(v, "shape")}


_STAT_KEYS = ("sum", "mean", "std", "max", "norm")


def _rel(a: float, b: float) -> float:
    """Relative difference, safe at zero."""
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b))
    return math.inf if denom == 0 else abs(a - b) / denom


def compare_tensor_dir(
    ref_dir: Path, tgt_dir: Path, strict: bool = True, rtol: float = 1e-6
) -> Result:
    """L3/L4/L6: compare every dump file present on both sides.

    ``strict`` (default) demands exact equality: for verbatim-migrated code on
    the same hardware, bit-exact is the right bar and any delta is a defect.
    """
    label = "L3/L4 tensors (strict)" if strict else f"L3/L4 tensors (rtol={rtol})"
    res = Result(label)

    def index(d: Path) -> dict[str, Path]:
        if not d.is_dir():
            raise SystemExit(f"error: not a directory: {d}")
        return {
            p.relative_to(d).as_posix(): p
            for p in sorted(d.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".pt", ".pth", ".bin"}
        }

    ref_files, tgt_files = index(ref_dir), index(tgt_dir)
    common = sorted(set(ref_files) & set(tgt_files))

    for name in sorted(set(ref_files) - set(tgt_files)):
        res.findings.append(f"dump only in REF: {name}")
    for name in sorted(set(tgt_files) - set(ref_files)):
        res.findings.append(f"dump only in TGT: {name}")

    if not common:
        # No overlapping dumps at all -> there is nothing to verify. Without
        # this guard the tool would print PASS on two empty directories.
        res.vacuous = True
        return res

    compared = 0
    elementwise = 0
    for name in common:
        r = load_tensor_stats(ref_files[name])
        t = load_tensor_stats(tgt_files[name])

        # When BOTH sides carry raw tensors, compare elementwise. Summary
        # statistics cannot decide bit-exactness (a 1-ULP delta vanishes in a
        # float32 sum), so a stats-only strict check would wrongly PASS.
        raw_r = load_raw_tensors(ref_files[name])
        raw_t = load_raw_tensors(tgt_files[name])
        if raw_r is not None and raw_t is not None:
            import torch  # type: ignore

            for key in sorted(set(raw_r) & set(raw_t)):
                a_t, b_t = raw_r[key], raw_t[key]
                if a_t.shape != b_t.shape or a_t.dtype != b_t.dtype:
                    continue  # reported by the stats pass below
                elementwise += 1
                if strict:
                    if not torch.equal(a_t, b_t):
                        d = (a_t.float() - b_t.float()).abs()
                        nbad = int((a_t != b_t).sum().item())
                        res.findings.append(
                            f"{name}:{key}: not bit-exact elementwise - "
                            f"{nbad}/{a_t.numel()} element(s) differ, "
                            f"max|delta|={d.max().item():.6e}"
                        )
                else:
                    d = (a_t.float() - b_t.float()).abs()
                    scale = torch.maximum(a_t.float().abs(), b_t.float().abs())
                    rel = (d / scale.clamp_min(1e-30)).max().item()
                    if rel > rtol:
                        res.findings.append(
                            f"{name}:{key}: max rel={rel:.3e} > rtol={rtol} "
                            f"(max|delta|={d.max().item():.6e})"
                        )

        for key in sorted(set(r) - set(t)):
            res.findings.append(f"{name}: tensor only in REF: {key}")
        for key in sorted(set(t) - set(r)):
            res.findings.append(f"{name}: tensor only in TGT: {key}")
        for key in sorted(set(r) & set(t)):
            rs, ts = r[key], t[key]
            if rs.get("shape") != ts.get("shape"):
                res.findings.append(
                    f"{name}:{key}: shape ref={rs.get('shape')} tgt={ts.get('shape')}"
                )
                continue
            if rs.get("dtype") != ts.get("dtype"):
                res.findings.append(
                    f"{name}:{key}: dtype ref={rs.get('dtype')} tgt={ts.get('dtype')}"
                )
            # Byte digest: only meaningful under strict, and only when both
            # dumps carry one. Deliberately NOT checked in relaxed mode, where
            # differing bytes within rtol are the point.
            if strict and "sha256" in rs and "sha256" in ts:
                compared += 1
                if rs["sha256"] != ts["sha256"]:
                    res.findings.append(
                        f"{name}:{key}: byte digest differs "
                        f"ref={rs['sha256'][:12]} tgt={ts['sha256'][:12]}"
                    )
            for sk in _STAT_KEYS:
                if sk not in rs or sk not in ts:
                    continue
                a, b = float(rs[sk]), float(ts[sk])
                compared += 1
                if strict:
                    if a != b:
                        res.findings.append(
                            f"{name}:{key}.{sk}: not bit-exact "
                            f"ref={a!r} tgt={b!r} (rel={_rel(a, b):.3e})"
                        )
                elif _rel(a, b) > rtol:
                    res.findings.append(
                        f"{name}:{key}.{sk}: rel={_rel(a, b):.3e} > rtol={rtol} ref={a!r} tgt={b!r}"
                    )

    res.notes.append(
        f"compared {len(common)} dump file(s), {compared} statistic(s), "
        f"{elementwise} tensor(s) elementwise"
    )
    if elementwise:
        res.notes.append(
            "elementwise comparison active (raw tensors on both sides) - "
            "this is what makes a bit-exact claim verifiable"
        )
    if compared == 0 and elementwise == 0:
        # Files matched by name but carried no comparable statistics.
        res.vacuous = True
    return res


# --------------------------------------------------------------------------
# L5 - loss curve + divergence SHAPE classification
# --------------------------------------------------------------------------

#: Divergence shapes, mapped to the cause/action table in PRECISION_BASELINE.md.
SHAPE_STEP_ZERO = "diverges-at-step-0"
SHAPE_LINEAR = "linear-growth"
SHAPE_EXPONENTIAL = "exponential-growth"
SHAPE_ONE_FLAT = "one-flat-one-descending"
SHAPE_SPIKE = "spike-then-recover"
SHAPE_PERIODIC = "periodic-large-error"
SHAPE_NOISE = "small-noise"

SHAPE_ACTIONS = {
    SHAPE_STEP_ZERO: "init not aligned -> go back to L4",
    SHAPE_LINEAR: "accumulating numeric diff -> unacceptable here, go to L6",
    SHAPE_EXPONENTIAL: "computation bug -> L6",
    SHAPE_ONE_FLAT: "broken gradient/operator -> L6, check backward first",
    SHAPE_SPIKE: "numerical instability -> check loss scaling / grad clip",
    SHAPE_PERIODIC: "data order or LR schedule mismatch -> back to L2/L3",
    SHAPE_NOISE: "within noise -> inspect, do not assume OK under strict mode",
}


def load_loss_series(path: Path) -> dict[int, float]:
    """Read ``{step: loss}``. Accepts a dict, or a list of step/loss records."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, float] = {}
    if isinstance(raw, dict):
        src = raw.get("losses", raw)
        if isinstance(src, dict):
            for k, v in src.items():
                out[int(k)] = float(v)
            return out
        raw = src
    if isinstance(raw, list):
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                step = int(item.get("step", item.get("iteration", i)))
                val = item.get("loss", item.get("lm_loss"))
                if val is None:
                    continue
                out[step] = float(val)
            else:
                out[i] = float(item)
    return out


def classify_divergence(
    steps: list[int], ref: list[float], tgt: list[float]
) -> tuple[str, dict[str, float]]:
    """Classify the SHAPE of the ref/tgt gap. Shape points at the cause.

    Checked in order of diagnostic priority, not magnitude: a step-0 mismatch
    means init is wrong and nothing downstream can be trusted, so it wins even
    if later deltas are bigger.
    """
    deltas = [abs(a - b) for a, b in zip(ref, tgt)]
    peak = max(deltas) if deltas else 0.0
    stats = {
        "max_abs_delta": peak,
        "mean_abs_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
        "first_delta": deltas[0] if deltas else 0.0,
        "last_delta": deltas[-1] if deltas else 0.0,
    }
    if not deltas or peak == 0.0:
        return SHAPE_NOISE, stats

    # 1. step 0 already differs -> initialisation.
    if deltas[0] > 0 and deltas[0] >= 0.5 * peak:
        return SHAPE_STEP_ZERO, stats

    # 2. one side learns, the other does not.
    def slope(series: list[float]) -> float:
        return series[-1] - series[0] if len(series) > 1 else 0.0

    r_slope, t_slope = slope(ref), slope(tgt)
    span = max(abs(r_slope), abs(t_slope))
    if span > 0 and min(abs(r_slope), abs(t_slope)) < 0.05 * span:
        return SHAPE_ONE_FLAT, stats

    # 3. isolated spike that comes back down.
    if len(deltas) >= 3:
        median = sorted(deltas)[len(deltas) // 2]
        if median > 0 and peak > 10 * median and deltas[-1] < 2 * median:
            return SHAPE_SPIKE, stats
        # 4. recurring large errors -> schedule/data periodicity.
        big = [i for i, d in enumerate(deltas) if d > 5 * median] if median else []
        if len(big) >= 3:
            gaps = {big[i + 1] - big[i] for i in range(len(big) - 1)}
            if len(gaps) == 1 and gaps != {1}:
                return SHAPE_PERIODIC, stats

    # 5. growth shape: exponential (ratio roughly constant >1) vs linear.
    #
    # Anchor on the first NONZERO delta, not on deltas[0]. The most likely real
    # shape in this migration is exactly zero at step 0 (identical init, as L4
    # demands) followed by accumulating drift -- requiring deltas[0] > 0 here
    # misfiled that case as noise, which the planted fixture caught.
    first_nz = next((i for i, d in enumerate(deltas) if d > 0), None)
    if first_nz is not None and len(deltas) - first_nz >= 3 and deltas[-1] > deltas[first_nz]:
        tail = deltas[first_nz:]
        ratios = [tail[i + 1] / tail[i] for i in range(len(tail) - 1) if tail[i] > 0]
        if ratios and sum(ratios) / len(ratios) > 1.5:
            return SHAPE_EXPONENTIAL, stats
        return SHAPE_LINEAR, stats

    return SHAPE_NOISE, stats


def compare_loss(
    ref_series: dict[int, float],
    tgt_series: dict[int, float],
    strict: bool = True,
    rtol: float = 1e-6,
) -> Result:
    """L5: per-step loss comparison plus a shape diagnosis when it diverges."""
    res = Result("L5 loss curve (strict)" if strict else f"L5 loss curve (rtol={rtol})")
    common = sorted(set(ref_series) & set(tgt_series))
    for s in sorted(set(ref_series) - set(tgt_series)):
        res.findings.append(f"step {s} only in REF")
    for s in sorted(set(tgt_series) - set(ref_series)):
        res.findings.append(f"step {s} only in TGT")
    if not common:
        res.vacuous = True
        return res

    ref = [ref_series[s] for s in common]
    tgt = [tgt_series[s] for s in common]
    for s, a, b in zip(common, ref, tgt):
        if math.isnan(b) or math.isinf(b):
            res.findings.append(f"step {s}: TGT loss is not finite: {b!r}")
    bad = [
        (s, a, b) for s, a, b in zip(common, ref, tgt) if (a != b if strict else _rel(a, b) > rtol)
    ]
    res.notes.append(f"compared {len(common)} step(s)")
    if bad:
        shape, stats = classify_divergence(common, ref, tgt)
        res.notes.append(
            "shape="
            + shape
            + "  max_abs_delta={max_abs_delta:.6e} mean_abs_delta={mean_abs_delta:.6e}".format(
                **stats
            )
        )
        res.notes.append(f"action: {SHAPE_ACTIONS[shape]}")
        for s, a, b in bad[:5]:
            res.findings.append(f"step {s}: ref={a!r} tgt={b!r} rel={_rel(a, b):.3e}")
        if len(bad) > 5:
            res.findings.append(f"... and {len(bad) - 5} more differing step(s)")
    return res


# --------------------------------------------------------------------------
# Self-test: prove each level can FAIL, and that identical input PASSES.
#
# A diffing tool's worst failure mode is returning PASS unconditionally
# (unreadable file treated as empty, both-empty treated as equal, mismatched
# shapes skipped). That gives green lights all the way to W3, when the model
# does not converge and nothing was ever actually aligned. So every level gets
# a planted defect it must catch, PLUS a false-positive control.
# --------------------------------------------------------------------------


def _self_test() -> int:
    checks: list[tuple[str, bool, str]] = []

    def expect(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    tmp = Path(tempfile.mkdtemp(prefix="cmp_precision_selftest_"))
    try:
        # ---- L1 ----------------------------------------------------------
        ref_params = "model.a.weight torch.Size([4, 8])\nmodel.b.bias (8)\n"
        same = compare_params(ref_params, ref_params)
        expect("L1 fp-control: identical tables PASS", same.ok, same.render())

        extra = ref_params + "model.c.weight (2, 2)\n"
        r = compare_params(ref_params, extra)
        expect(
            "L1 catches an extra TGT parameter",
            not r.ok and any("only in TGT" in f for f in r.findings),
            r.render(),
        )

        reshaped = "model.a.weight torch.Size([8, 4])\nmodel.b.bias (8)\n"
        r = compare_params(ref_params, reshaped)
        expect(
            "L1 catches a transposed shape (same numel!)",
            not r.ok and any("shape mismatch" in f for f in r.findings),
            r.render(),
        )

        fused_ref = "l.q.weight (4, 4)\nl.k.weight (4, 4)\nl.v.weight (4, 4)\n"
        fused_tgt = "l.qkv.weight (12, 4)\n"
        r = compare_params(fused_ref, fused_tgt)
        expect(
            "L1 fusion: numel matches but mapping is reported missing",
            not r.ok and not any("total element count" in f for f in r.findings),
            r.render(),
        )

        r = compare_params("", "")
        expect("L1 vacuity: two empty tables are NOT a PASS", not r.ok, r.render())

        # ---- L2 ----------------------------------------------------------
        base_cfg = {
            "global_batch_size": 64,
            "micro_batch_size": 2,
            "world_size": 8,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "gradient_accumulation_steps": 8,
            "optimizer": {"name": "adam", "eps": 1e-8, "betas": [0.9, 0.95]},
        }
        r = compare_config(base_cfg, json.loads(json.dumps(base_cfg)))
        expect("L2 fp-control: identical configs PASS", r.ok, r.render())

        eps_cfg = json.loads(json.dumps(base_cfg))
        eps_cfg["optimizer"]["eps"] = 1e-18
        r = compare_config(base_cfg, eps_cfg)
        expect(
            "L2 classifies eps 1e-8 vs 1e-18 as SEMANTIC (must-fix)",
            not r.ok and any(SEMANTIC_DIFFERENT in f for f in r.findings),
            r.render(),
        )

        math_cfg = json.loads(json.dumps(base_cfg))
        math_cfg["use_distributed_optimizer"] = True
        base_plus = json.loads(json.dumps(base_cfg))
        base_plus["use_distributed_optimizer"] = False
        r = compare_config(base_plus, math_cfg)
        expect(
            "L2 classifies distributed-optimizer as MATH-EQUIVALENT (note only)",
            r.ok and any(MATH_EQUIVALENT in n for n in r.notes),
            r.render(),
        )

        fa_cfg = json.loads(json.dumps(base_cfg))
        fa_cfg["use_flash_attn"] = True
        base_fa = json.loads(json.dumps(base_cfg))
        base_fa["use_flash_attn"] = False
        r = compare_config(base_fa, fa_cfg)
        expect(
            "L2 flags flash-attn switch as NUMERIC-DIFFERENT finding here",
            not r.ok and any(NUMERIC_DIFFERENT in f for f in r.findings),
            r.render(),
        )

        bad_batch = json.loads(json.dumps(base_cfg))
        bad_batch["gradient_accumulation_steps"] = 7  # 2*4*7=56 != 64
        r = compare_config(bad_batch, bad_batch)
        expect(
            "L2 catches a violated batch identity even when ref==tgt",
            not r.ok and any("batch identity" in f for f in r.findings),
            r.render(),
        )

        r = compare_config({}, {})
        expect("L2 vacuity: two empty configs are NOT a PASS", not r.ok, r.render())
        return _self_test_tensors_and_loss(tmp, checks, expect)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write_stats(path: Path, name: str, **stats: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: {"shape": [2, 2], "dtype": "torch.float32", **stats}}
    path.write_text(json.dumps(payload), encoding="utf-8")


def _self_test_tensors_and_loss(tmp: Path, checks, expect) -> int:
    # ---- L3 / L4 ---------------------------------------------------------
    ref_d, tgt_d = tmp / "ref", tmp / "tgt"
    for d in (ref_d, tgt_d):
        _write_stats(d / "step0.json", "input_ids", sum=100.0, mean=25.0, norm=50.0)
    r = compare_tensor_dir(ref_d, tgt_d, strict=True)
    expect("L3 fp-control: identical dumps PASS in strict mode", r.ok, r.render())

    # One ULP of float64 at 100.0 -- the smallest possible planted defect.
    nudged = math.nextafter(100.0, math.inf)
    _write_stats(tgt_d / "step0.json", "input_ids", sum=nudged, mean=25.0, norm=50.0)
    r = compare_tensor_dir(ref_d, tgt_d, strict=True)
    expect(
        "L4 strict catches a 1-ULP difference (bit-exact really is bit-exact)",
        not r.ok and any("not bit-exact" in f for f in r.findings),
        r.render(),
    )
    r = compare_tensor_dir(ref_d, tgt_d, strict=False, rtol=1e-6)
    expect(
        "L4 relaxed tolerates that same 1-ULP difference",
        r.ok,
        r.render(),
    )

    _write_stats(tgt_d / "step0.json", "input_ids", sum=100.0, mean=25.0, norm=50.0)
    _write_stats(tgt_d / "step1.json", "input_ids", sum=1.0, mean=1.0, norm=1.0)
    r = compare_tensor_dir(ref_d, tgt_d, strict=True)
    expect(
        "L3 catches a dump present only on one side",
        not r.ok and any("only in TGT" in f for f in r.findings),
        r.render(),
    )

    empty_a, empty_b = tmp / "ea", tmp / "eb"
    empty_a.mkdir()
    empty_b.mkdir()
    r = compare_tensor_dir(empty_a, empty_b, strict=True)
    expect(
        "L3 vacuity: two EMPTY dump dirs are NOT a PASS",
        not r.ok and r.vacuous,
        r.render(),
    )

    # Dumps that exist but carry no comparable statistics must also be vacuous:
    # name-matching alone must never be mistaken for verification.
    for d in (tmp / "na", tmp / "nb"):
        d.mkdir()
        (d / "step0.json").write_text(
            json.dumps({"input_ids": {"shape": [2, 2], "dtype": "torch.float32"}}),
            encoding="utf-8",
        )
    r = compare_tensor_dir(tmp / "na", tmp / "nb", strict=True)
    expect(
        "L3 vacuity: stat-less dumps are NOT a PASS",
        not r.ok and r.vacuous,
        r.render(),
    )

    # ---- L5 --------------------------------------------------------------
    steps = list(range(8))
    descending = [10.0 - 0.5 * i for i in steps]

    r = compare_loss(dict(zip(steps, descending)), dict(zip(steps, descending)))
    expect("L5 fp-control: identical curves PASS", r.ok, r.render())

    shapes = {
        SHAPE_STEP_ZERO: [v + 2.0 for v in descending],
        SHAPE_LINEAR: [v + 0.01 * i for i, v in enumerate(descending)],
        SHAPE_EXPONENTIAL: [v + 1e-4 * (3.0**i) for i, v in enumerate(descending)],
        SHAPE_ONE_FLAT: [10.0] * len(steps),
    }
    for want, series in shapes.items():
        got, _ = classify_divergence(steps, descending, series)
        expect(f"L5 classifies planted '{want}' correctly (got '{got}')", got == want)

    nan_tgt = list(descending)
    nan_tgt[3] = float("nan")
    r = compare_loss(dict(zip(steps, descending)), dict(zip(steps, nan_tgt)))
    expect(
        "L5 catches a non-finite target loss",
        not r.ok and any("not finite" in f for f in r.findings),
        r.render(),
    )

    r = compare_loss({}, {})
    expect("L5 vacuity: two empty curves are NOT a PASS", not r.ok, r.render())

    # ---- report ----------------------------------------------------------
    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        if not ok and detail:
            for line in detail.splitlines():
                print(f"        {line}")
    print()
    if failed:
        print(f"SELF-TEST FAILED: {len(failed)}/{len(checks)} check(s) failed")
        return 1
    print(
        f"SELF-TEST PASSED: {len(checks)}/{len(checks)} - every level detects a "
        f"planted defect, identical input passes, and empty input is reported "
        f"VACUOUS rather than PASS"
    )
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="compare_precision.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run the planted-defect self-test and exit (no inputs needed)",
    )
    p.add_argument(
        "--emit-determinism-snippet",
        action="store_true",
        help="print the determinism block to paste into both training scripts",
    )
    # `--json` must work on BOTH sides of the subcommand: a caller naturally
    # writes `compare_precision.py loss --ref A --tgt B --json`. argparse does
    # not share top-level flags with subparsers, so it is declared twice --
    # once here, and once per subcommand with default=SUPPRESS so that the
    # subparser does not clobber a value given before the subcommand.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="emit machine-readable json",
    )

    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("params", help="L1 - model structure", parents=[common])
    sp.add_argument("--ref", type=Path, required=True)
    sp.add_argument("--tgt", type=Path, required=True)
    sp.add_argument(
        "--mapping",
        type=Path,
        help="json {ref_name: tgt_name} for fused/renamed params",
    )

    sc = sub.add_parser("config", help="L2 - hyperparameters", parents=[common])
    sc.add_argument("--ref", type=Path, required=True)
    sc.add_argument("--tgt", type=Path, required=True)

    st = sub.add_parser("tensors", help="L3/L4/L6 - tensor dumps", parents=[common])
    st.add_argument("--ref", type=Path, required=True)
    st.add_argument("--tgt", type=Path, required=True)
    st.add_argument("--rtol", type=float, default=1e-6)

    sl = sub.add_parser("loss", help="L5 - loss curve", parents=[common])
    sl.add_argument("--ref", type=Path, required=True)
    sl.add_argument("--tgt", type=Path, required=True)
    sl.add_argument("--rtol", type=float, default=1e-6)

    for q in (st, sl):
        mode = q.add_mutually_exclusive_group()
        mode.add_argument(
            "--strict",
            dest="strict",
            action="store_true",
            default=True,
            help="bit-exact comparison (DEFAULT for verbatim migration)",
        )
        mode.add_argument(
            "--relaxed",
            dest="strict",
            action="store_false",
            help="allow --rtol; only for batches registered as 'implementation "
            "rewritten' in MIGRATION_MAP.md",
        )
    p.add_argument("--json", action="store_true", help="emit machine-readable json")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.emit_determinism_snippet:
        print(DETERMINISM_SNIPPET, end="")
        return 0
    if args.self_test:
        return _self_test()
    if not args.cmd:
        parser.print_help()
        return 2

    try:
        if args.cmd == "params":
            mapping = json.loads(args.mapping.read_text(encoding="utf-8")) if args.mapping else None
            res = compare_params(
                args.ref.read_text(encoding="utf-8"),
                args.tgt.read_text(encoding="utf-8"),
                mapping,
            )
        elif args.cmd == "config":
            res = compare_config(load_config(args.ref), load_config(args.tgt))
        elif args.cmd == "tensors":
            res = compare_tensor_dir(args.ref, args.tgt, args.strict, args.rtol)
        else:
            res = compare_loss(
                load_loss_series(args.ref),
                load_loss_series(args.tgt),
                args.strict,
                args.rtol,
            )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: malformed json input: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "level": res.level,
                    "ok": res.ok,
                    "vacuous": res.vacuous,
                    "findings": res.findings,
                    "notes": res.notes,
                },
                indent=2,
            )
        )
    else:
        print(res.render())
    return 0 if res.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
