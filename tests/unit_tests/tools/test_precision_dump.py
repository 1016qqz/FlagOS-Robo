# Copyright (c) 2025 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the producer side of precision alignment.

``flagos_robo/utils/precision_dump.py`` writes what
``tools/compare_precision.py`` reads, so the contract between the two is what
these tests pin down -- not the producer's internal behaviour in isolation.
A test that only checked "dump_loss wrote a file" would pass while the
comparer rejected every one of those files.

So the comparer is driven as a real subprocess against real dumps, and its
exit code is the assertion. The exit codes are the tool's documented
interface: 0 PASS, 1 FAIL, 2 bad input.

The torch-dependent half (tensor stats and the byte digest) lives in
``test_precision_dump_tensors.py`` so this module stays runnable on a machine
with no torch installed, which is where the code is normally edited.
"""

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


def _run_loss_cmp(ref: Path, tgt: Path) -> subprocess.CompletedProcess:
    """Drive the real comparer. Its exit code is the contract under test."""
    return subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "loss",
            "--ref",
            str(ref),
            "--tgt",
            str(tgt),
        ],
        capture_output=True,
        text=True,
    )


def test_converted_dump_is_accepted_by_comparer(tmp_path):
    """The round trip the whole design exists for: dump -> convert -> compare."""
    run = tmp_path / "ref"
    for step in range(20):
        pd.dump_step(run, step, loss=2.5 - 0.1 * step)

    converted = pd.losses_to_json(run / pd.DEFAULT_LOSS_FILENAME)
    result = _run_loss_cmp(converted, converted)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout


def test_raw_jsonl_is_rejected_by_comparer(tmp_path):
    """Why the conversion step is not optional.

    The comparer reads a whole file with ``json.loads``, so feeding it the
    append-only JSONL fails as malformed input. This test documents the
    reason the two formats both exist -- if someone "simplifies" the producer
    to emit a single JSON document and drops the JSONL, a killed run starts
    losing every step it had already completed.
    """
    run = tmp_path / "ref"
    for step in range(3):
        pd.dump_step(run, step, loss=1.0)

    result = _run_loss_cmp(run / pd.DEFAULT_LOSS_FILENAME, run / pd.DEFAULT_LOSS_FILENAME)

    assert result.returncode == 2
    assert "malformed json" in (result.stdout + result.stderr)


def test_comparer_fails_on_a_real_divergence(tmp_path):
    """Guard against a vacuous PASS: a changed loss must be caught and named."""
    run = tmp_path / "ref"
    for step in range(20):
        pd.dump_step(run, step, loss=2.5 - 0.1 * step)
    ref = pd.losses_to_json(run / pd.DEFAULT_LOSS_FILENAME)

    rows = json.loads(ref.read_text(encoding="utf-8"))
    rows[10]["loss"] += 0.4
    tgt = tmp_path / "tgt.json"
    tgt.write_text(json.dumps(rows), encoding="utf-8")

    result = _run_loss_cmp(ref, tgt)

    assert result.returncode == 1
    assert "step 10" in result.stdout


def test_append_survives_a_truncated_tail_line(tmp_path):
    """A hard kill (OOM killer, scancel) can cut the last line mid-write.

    Every step before the cut must remain readable -- that is the whole
    payoff of appending line by line. Refusing to parse the file would make
    the design pointless.
    """
    run = tmp_path / "ref"
    for step in range(4):
        pd.dump_loss(run, step, 0.5)
    path = run / pd.DEFAULT_LOSS_FILENAME
    path.write_text(path.read_text(encoding="utf-8") + '{"step": 4, "lo', encoding="utf-8")

    assert len(pd.read_loss_jsonl(path)) == 4


def test_malformed_line_in_the_middle_raises(tmp_path):
    """A broken line that is NOT the tail means two writers interleaved.

    That is data corruption rather than a clean truncation, so it must be
    loud. Skipping it silently would hand the comparer a series with a hole
    in it and the resulting "divergence" would send the next person hunting a
    numerical bug that does not exist.
    """
    run = tmp_path / "ref"
    for step in range(3):
        pd.dump_loss(run, step, 0.5)
    path = run / pd.DEFAULT_LOSS_FILENAME
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(1, "{broken")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        pd.read_loss_jsonl(path)


def test_out_of_order_steps_are_sorted_on_conversion(tmp_path):
    """Resume-from-checkpoint and retried steps append out of order.

    The comparer reads the series positionally for its divergence-shape
    classification, so an unsorted array would be diagnosed as a wild
    oscillation instead of a clean match.
    """
    run = tmp_path / "ref"
    for step in (5, 3, 9, 1):
        pd.dump_loss(run, step, 1.0 + step)

    rows = json.loads(pd.losses_to_json(run / pd.DEFAULT_LOSS_FILENAME).read_text(encoding="utf-8"))

    assert [r["step"] for r in rows] == [1, 3, 5, 9]


def test_dump_step_skips_non_zero_ranks_by_default(tmp_path):
    """Under 6-way DP every rank sees a different microbatch.

    Dumping all of them would make the comparer pair up files whose contents
    differ legitimately and report differences that are not errors. Rank 0
    only is the default; opting in to all ranks stays possible.
    """
    assert pd.dump_step(tmp_path / "a", 0, loss=1.0, rank=3) == {}

    written = pd.dump_step(tmp_path / "b", 0, loss=1.0, rank=3, only_rank=None)
    assert "loss" in written
    record = pd.read_loss_jsonl(written["loss"])[0]
    assert record["rank"] == 3


def test_dump_creates_missing_directories(tmp_path):
    """The dump dir is normally a fresh per-experiment path that does not exist."""
    deep = tmp_path / "exp001_L5" / "ref_flagscale"
    path = pd.dump_loss(deep, 0, 1.25)

    assert path.exists()
    assert pd.read_loss_jsonl(path) == [{"step": 0, "loss": 1.25}]


def test_step_relpath_is_zero_padded_for_lexicographic_order(tmp_path):
    """The comparer pairs files by sorted rglob order, which is lexicographic.

    Unpadded names would order step 10 before step 2 and the reported "first
    diverging step" would be wrong.
    """
    names = [pd.step_relpath(s) for s in (2, 10, 100)]

    assert names == [
        "step_000002/rank_0.json",
        "step_000010/rank_0.json",
        "step_000100/rank_0.json",
    ]
    assert sorted(names) == names


def test_extra_metadata_cannot_overwrite_step_or_loss(tmp_path):
    """``extra`` is caller-supplied; it must not shadow the two load-bearing keys."""
    path = pd.dump_loss(tmp_path / "r", 7, 0.25, extra={"step": 999, "loss": 999.0, "lr": 1e-4})
    record = pd.read_loss_jsonl(path)[0]

    assert record["step"] == 7
    assert record["loss"] == 0.25
    assert record["lr"] == pytest.approx(1e-4)
