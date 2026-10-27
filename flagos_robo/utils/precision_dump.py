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
"""Precision dumping for training-loop instrumentation.

This is the *producer* side of precision alignment. The *consumer* is
``tools/compare_precision.py``, and its reader functions define the output
format -- not this module. Two consumer contracts are honoured here:

``loss`` subcommand (``load_loss_series``)
    Reads a whole file with ``json.loads``, so a JSON Lines file canNOT be fed
    to it directly. But a training run that is killed mid-step must not lose
    the steps already written, and that is exactly what append-per-line buys.
    So the dump is JSONL and :func:`losses_to_json` converts it to the single
    document array the comparer wants. The two formats are not redundant; each
    one is required by a different consumer.

``tensors`` subcommand (``load_tensor_stats``)
    Reads a directory tree via ``rglob`` and pairs the two sides by the path
    *relative to* each root, so the relative layout written here must be
    byte-identical on the reference and target sides. Hence the layout is
    derived purely from ``step`` and ``rank`` -- never from a hostname, a
    timestamp, or a framework name, all of which differ across the two sides
    and would make every file an unpaired MISS.

Why stats JSON rather than raw ``.pt``: the comparer needs no torch to read
it, so comparison runs on a laptop instead of occupying a GPU node.

Usage in a training loop::

    from flagos_robo.utils.precision_dump import dump_step

    dump_step(out_dir, step, loss=loss, tensors={"logits": logits})

Enabled by the caller via ``dump_dir_from_env()``, which reads
``FLAGOS_PRECISION_DUMP``. That is the only env-reading function here; the
dump functions themselves take an explicit directory so they stay testable.
Both the FlagOS-Robo and the reference training loop must use the SAME gate,
otherwise a comparison run can silently dump on one side only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "DEFAULT_LOSS_FILENAME",
    "dump_step",
    "dump_loss",
    "dump_tensors",
    "losses_to_json",
    "read_loss_jsonl",
    "tensor_digest",
    "tensor_stats",
    "step_relpath",
    "DUMP_ENV_VAR",
    "dump_dir_from_env",
    "STEP_LOG_ENV_VAR",
    "DEFAULT_STEP_LOG_FILENAME",
    "step_log_dir_from_env",
    "format_step_log_line",
    "append_step_log",
]

#: Env var that turns dumping on. Value is the output directory.
DUMP_ENV_VAR = "FLAGOS_PRECISION_DUMP"

#: Env var that turns the human-readable per-step log on. Value is the
#: directory the log file is written into. Deliberately separate from
#: ``DUMP_ENV_VAR``: the machine-readable JSONL and the human/grep-readable
#: log serve different consumers and a run may want either alone.
STEP_LOG_ENV_VAR = "FLAGOS_STEP_LOG"

#: Loss records live in one append-only JSONL file per run directory.
DEFAULT_LOSS_FILENAME = "loss.jsonl"

#: Per-step log filename. Rank 0 is the only writer (see ``append_step_log``),
#: so the name states the rank rather than pretending to be a merged log.
DEFAULT_STEP_LOG_FILENAME = "rank0.log"

#: Field order of a tensor stats record. Kept explicit and identical to
#: ``tools/compare_precision.py::load_tensor_stats`` so a field rename on
#: either side shows up as a test failure rather than as a silent MISS.
STATS_FIELDS = (
    "shape",
    "dtype",
    "sum",
    "mean",
    "std",
    "max",
    "norm",
    "sha256",
)


def tensor_digest(t: Any) -> str:
    """SHA-256 over a tensor's raw bytes -- a bit-exactness fingerprint.

    Deliberately the same algorithm as
    ``tools/compare_precision.py::tensor_digest``. It is duplicated rather
    than imported because ``tools/`` is a script directory, not an importable
    package, and a training entrypoint must not depend on it. The duplication
    is pinned by a unit test that asserts both implementations agree -- if one
    drifts, the digests stop matching and every strict comparison would report
    a false difference.

    Two details carry the correctness:

    * ``shape`` and ``dtype`` are hashed in, so a reshape or a dtype change
      cannot collide with identical payload bytes.
    * ``view(torch.uint8)`` is used instead of ``numpy().tobytes()`` because
      numpy has no bfloat16 -- it raises
      ``TypeError: Got unsupported ScalarType BFloat16`` -- and bf16 is the
      training dtype here.
    """
    import torch  # Local: keeps module import torch-free until actually used.

    c = t.detach().cpu().contiguous().flatten()
    h = hashlib.sha256()
    h.update(f"{tuple(t.shape)}|{t.dtype}|".encode())
    if c.numel():
        h.update(c.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def tensor_stats(t: Any) -> dict[str, Any]:
    """Summarise one tensor into the comparer's stats record.

    The empty / single-element guards mirror the comparer exactly: ``mean``
    and ``max`` of an empty tensor are reported as ``0.0`` rather than NaN,
    and ``std`` needs more than one element. Without these, torch returns NaN
    and a NaN never equals itself, so a tensor would be reported as differing
    from an identical copy of itself.
    """
    ft = t.float()
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "sum": ft.sum().item(),
        "mean": ft.mean().item() if t.numel() else 0.0,
        "std": ft.std().item() if t.numel() > 1 else 0.0,
        "max": ft.max().item() if t.numel() else 0.0,
        "norm": ft.norm().item(),
        "sha256": tensor_digest(t),
    }


def step_relpath(step: int, rank: int = 0) -> str:
    """Relative path of a step's tensor dump, as ``step_000003/rank_0.json``.

    Zero-padded so a plain lexicographic ``sorted()`` -- which is what the
    comparer's ``rglob`` listing uses -- also orders the steps numerically.
    Six digits covers a million steps; beyond that the padding simply grows
    and ordering still holds within a run of consistent width.
    """
    return f"step_{step:06d}/rank_{rank}.json"


def dump_dir_from_env(env: Mapping[str, str] | None = None) -> Path | None:
    """Return the dump directory if dumping is enabled, else ``None``.

    This is the single gate both training loops call, so the two sides cannot
    drift into dumping under different conditions. Returns ``None`` when the
    var is unset or empty, which makes instrumentation inert by default -- a
    normal training run pays nothing.

    ``env`` is injectable so tests need not mutate ``os.environ``.
    """
    source = os.environ if env is None else env
    raw = source.get(DUMP_ENV_VAR)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    # A bare "0"/"false" is a common way to try to switch a flag off; honour it
    # rather than creating a directory literally named "0".
    if value.lower() in {"0", "false", "no", "off"}:
        return None
    return Path(value).expanduser()


def step_log_dir_from_env(env: Mapping[str, str] | None = None) -> Path | None:
    """Return the per-step log directory if step logging is enabled, else ``None``.

    Same off-value handling as :func:`dump_dir_from_env` so the two gates cannot
    disagree about what "off" means -- a run that writes the JSONL but silently
    skips the log (or vice versa) is the failure this symmetry prevents.
    """
    source = os.environ if env is None else env
    raw = source.get(STEP_LOG_ENV_VAR)
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    if value.lower() in {"0", "false", "no", "off"}:
        return None
    return Path(value).expanduser()


def format_step_log_line(
    step: int,
    *,
    loss: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Render one step as ``[step N] loss=... grad_norm=...``.

    The ``[step N]`` prefix starts at column 0 on purpose: the acceptance check
    greps ``^\\[step (199|200)\\]``, and a logging-framework timestamp prefix
    would push it off the line start and make the check unmatchable. That is
    also why this is a plain file append rather than a ``logger`` call.

    ``repr`` is used for floats so the line carries full precision -- a log
    rounded to three decimals cannot be cross-checked against the JSONL.
    """
    parts = [f"[step {int(step)}]"]
    if loss is not None:
        parts.append(f"loss={_as_float(loss)!r}")
    if extra:
        for key in sorted(extra):
            if key in ("step", "loss"):
                continue
            value = extra[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                parts.append(f"{key}={value}")
            else:
                parts.append(f"{key}={float(value)!r}")
    return " ".join(parts)


def append_step_log(
    out_dir: str | os.PathLike[str],
    step: int,
    *,
    loss: Any = None,
    extra: Mapping[str, Any] | None = None,
    rank: int = 0,
    only_rank: int | None = 0,
    filename: str = DEFAULT_STEP_LOG_FILENAME,
) -> Path | None:
    """Append one ``[step N] ...`` line to the run's rank-0 log.

    Returns the path written, or ``None`` when this rank was skipped. Flushed
    and fsynced for the same reason as :func:`dump_loss`: the check reads this
    file to prove the run reached its last step, so a hard kill must not be
    able to swallow the final lines that are the evidence.
    """
    if only_rank is not None and rank != only_rank:
        return None
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    line = format_step_log_line(step, loss=loss, extra=extra)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


def _as_float(value: Any) -> float:
    """Coerce a loss to a plain float, accepting a 0-dim tensor or a number.

    ``.item()`` is tried first so a tensor never reaches ``float()`` -- on a
    multi-element tensor ``float()`` raises, and that is the right outcome: a
    loss is scalar by definition and silently taking element 0 would dump a
    wrong number that looks plausible.
    """
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)


def dump_loss(
    out_dir: str | os.PathLike[str],
    step: int,
    loss: Any,
    *,
    rank: int = 0,
    extra: Mapping[str, Any] | None = None,
    filename: str = DEFAULT_LOSS_FILENAME,
) -> Path:
    """Append one ``{"step": ..., "loss": ...}`` record to the run's JSONL.

    Append mode is the point: a run killed at step 900 keeps the 900 records
    already on disk. A single JSON document would have to be rewritten whole
    on every step and would be left truncated and unparseable by a kill.

    ``flush`` + ``fsync`` are explicit because the common failure here is a
    hard kill (OOM killer, ``scancel``, node fault) which does not run
    interpreter exit handlers, so buffered lines would be lost.
    """
    path = Path(out_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"step": int(step), "loss": _as_float(loss)}
    if rank:
        record["rank"] = int(rank)
    if extra:
        for key, value in extra.items():
            if key not in ("step", "loss"):
                record[key] = value
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return path


def dump_tensors(
    out_dir: str | os.PathLike[str],
    step: int,
    tensors: Mapping[str, Any],
    *,
    rank: int = 0,
) -> Path:
    """Write one step's tensor stats to ``step_NNNNNN/rank_R.json``.

    Scalar metadata (``step``, ``rank``) is written alongside the per-tensor
    records. The comparer keeps only the dict-valued top-level keys, so these
    scalars are visible to a human reading the file but are skipped by the
    comparison -- they carry no risk of being reported as a difference.
    """
    path = Path(out_dir) / step_relpath(step, rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"step": int(step), "rank": int(rank)}
    for name, tensor in tensors.items():
        payload[name] = tensor_stats(tensor)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def dump_step(
    out_dir: str | os.PathLike[str],
    step: int,
    *,
    loss: Any = None,
    tensors: Mapping[str, Any] | None = None,
    rank: int = 0,
    only_rank: int | None = 0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """One call per training step -- the function training loops import.

    ``only_rank`` defaults to rank 0 because a dump from every rank of a
    6-way data-parallel run produces six files per step whose contents differ
    legitimately (different microbatches), which the comparer would pair up and
    report as differences. Pass ``only_rank=None`` to dump every rank when
    that is genuinely wanted, e.g. inspecting a DP sharding bug.

    Returns the paths actually written, so a caller can log them; an empty
    dict means this rank was skipped.
    """
    if only_rank is not None and rank != only_rank:
        return {}
    written: dict[str, Path] = {}
    if loss is not None:
        written["loss"] = dump_loss(out_dir, step, loss, rank=rank, extra=extra)
    if tensors:
        written["tensors"] = dump_tensors(out_dir, step, tensors, rank=rank)
    return written


def read_loss_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Parse a loss JSONL file, tolerating a truncated final line.

    A hard kill can leave a partial last line. Dropping only that line keeps
    the preceding steps usable -- which is the entire reason for appending
    line by line, so refusing to parse the file would defeat the design.
    Earlier malformed lines are a different matter: they indicate interleaved
    writers, so they raise rather than being skipped silently.
    """
    src = Path(path)
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # Truncated tail from a kill: keep what came before.
            raise
    return out


def losses_to_json(
    jsonl_path: str | os.PathLike[str],
    json_path: str | os.PathLike[str] | None = None,
    *,
    records: Iterable[Mapping[str, Any]] | None = None,
) -> Path:
    """Convert the JSONL dump into the single-document array the comparer reads.

    This exists because ``compare_precision.py`` calls ``json.loads`` on the
    whole file: JSONL input fails there with a malformed-JSON error. The
    output is a ``list`` of ``{"step", "loss"}`` objects, which is the shape
    ``load_loss_series`` handles natively.

    Sorted by step so an out-of-order append (resume from checkpoint, retried
    step) still yields a monotonically ordered series for the comparer's
    divergence-shape classification, which reads the series positionally.
    """
    rows = list(records) if records is not None else read_loss_jsonl(jsonl_path)
    out = Path(json_path) if json_path else Path(jsonl_path).with_suffix(".json")
    payload = [
        {"step": int(r["step"]), "loss": float(r["loss"])}
        for r in sorted(rows, key=lambda r: int(r["step"]))
        if r.get("loss") is not None
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
