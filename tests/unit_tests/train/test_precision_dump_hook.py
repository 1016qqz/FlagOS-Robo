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
"""Regression tests for the precision-dump hook in the gr00t training loop.

The hook itself is three lines in ``train_gr00t_n1_5.py``, but its CORRECTNESS
lives entirely in *where* those lines sit and *which* attribute they read.
Both are easy to "clean up" into silent wrongness, and neither would make any
other test fail -- hence this file.

Two invariants are pinned:

1. ORDER -- the dump must run after ``train_tracker.step()`` and BEFORE the
   ``step % log_freq`` block, because that block calls ``reset_averages()``
   which zeroes every meter's ``.val``. Dumping after it records ``loss=0.0``
   on exactly the log_freq steps, producing a loss curve that looks plausible
   (most steps fine) while being wrong every Nth step.

2. ``.val`` not ``.avg`` -- ``MetricsTracker.__getattr__`` hands back the
   ``AverageMeter``, and ``to_dict()`` defaults to ``use_avg=True``. A
   per-step precision comparison needs the step's own value, not a running
   window mean.

These use the real ``MetricsTracker``/``AverageMeter`` rather than fakes, so
an upstream change to their semantics surfaces here.
"""

from __future__ import annotations

import sys
import types

import pytest

from flagos_robo.utils.precision_dump import (
    DEFAULT_LOSS_FILENAME,
    DUMP_ENV_VAR,
    dump_dir_from_env,
    dump_step,
    read_loss_jsonl,
)

LOG_FREQ = 5
STEPS = 20

# Both training entry points are same-origin copies, and every invariant below
# applies to both. Parametrising over this list is what makes the two sides
# unable to drift apart: the hooks were added to the gr00t loop first and the
# pi loop went without them for a whole milestone precisely because nothing
# compared the two. Adding a third entry point means adding it here.
ENTRY_POINTS = ["train_gr00t_n1_5.py", "train_pi.py"]


def _logging_utils():
    """Import MetricsTracker, stubbing the torch.distributed it needs.

    ``MetricsTracker.step()`` does a function-local ``import torch.distributed``
    and calls only ``is_initialized()``/``get_world_size()``. Stubbing exactly
    those two keeps this test runnable without torch (the comparison tooling is
    deliberately torch-free) while preserving single-process semantics.
    """
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        dist_mod = types.ModuleType("torch.distributed")
        dist_mod.is_initialized = lambda: False
        dist_mod.get_world_size = lambda: 1
        torch_mod.distributed = dist_mod
        sys.modules["torch"] = torch_mod
        sys.modules["torch.distributed"] = dist_mod
    from flagos_robo.train.utils import logging_utils

    return logging_utils


def _make_tracker():
    lu = _logging_utils()
    return lu.MetricsTracker(
        batch_size=2,
        num_frames=100,
        num_episodes=10,
        metrics={
            "loss": lu.AverageMeter("loss", ":.3f"),
            "grad_norm": lu.AverageMeter("grdn", ":.3f"),
            "lr": lu.AverageMeter("lr", ":0.1e"),
        },
    )


def _run_loop(out_dir, *, dump_before_log):
    """Replay the shape of the gr00t loop around the dump hook.

    ``dump_before_log=True`` is the shipped placement; ``False`` is the
    tempting-but-wrong placement this test exists to reject.
    """
    tracker = _make_tracker()
    step = 0
    for i in range(STEPS):
        # stands in for update_policy(): assignment routes through
        # __setattr__ -> metrics[name].update(value)
        tracker.loss = 2.0 - 0.01 * i
        tracker.grad_norm = 1.5
        tracker.lr = 1e-4

        step += 1
        tracker.step()

        if dump_before_log:
            dump_step(
                out_dir,
                step,
                loss=float(tracker.loss.val),
                rank=0,
                extra={"grad_norm": float(tracker.grad_norm.val)},
            )

        if step % LOG_FREQ == 0:
            tracker.reset_averages()

        if not dump_before_log:
            dump_step(
                out_dir,
                step,
                loss=float(tracker.loss.val),
                rank=0,
                extra={"grad_norm": float(tracker.grad_norm.val)},
            )

    return read_loss_jsonl(out_dir / DEFAULT_LOSS_FILENAME)


def test_shipped_order_never_dumps_a_zeroed_loss(tmp_path):
    records = _run_loop(tmp_path, dump_before_log=True)

    assert len(records) == STEPS
    assert [r["step"] for r in records] == list(range(1, STEPS + 1))
    assert [r["step"] for r in records if r["loss"] == 0.0] == []
    # step numbering is 1-based so dumped steps line up with the log line
    assert records[0]["step"] == 1
    assert records[0]["loss"] == pytest.approx(2.0)
    assert records[-1]["loss"] == pytest.approx(2.0 - 0.01 * (STEPS - 1))
    # extras ride along on the same record
    assert records[0]["grad_norm"] == pytest.approx(1.5)


def test_dumping_after_the_log_block_zeroes_every_log_freq_step(tmp_path):
    """Pins WHY the hook sits where it does.

    If someone moves the dump below the logging block, this is the damage:
    reset_averages() has already zeroed .val, so those steps record 0.0.
    """
    records = _run_loop(tmp_path, dump_before_log=False)

    zeroed = [r["step"] for r in records if r["loss"] == 0.0]
    assert zeroed == [LOG_FREQ * k for k in range(1, STEPS // LOG_FREQ + 1)]
    # the quiet part: most steps still look fine, which is what makes this bug
    # survive a glance at the log
    assert len(records) - len(zeroed) == STEPS - STEPS // LOG_FREQ


def test_tracker_attribute_is_a_meter_not_a_float():
    """Pins the `.val` half of the hook.

    Reading the attribute yields an AverageMeter; `.avg` is a window mean that
    would silently smooth the dumped curve.
    """
    tracker = _make_tracker()
    tracker.loss = 1.25
    tracker.loss = 3.75

    assert not isinstance(tracker.loss, float)
    assert tracker.loss.val == pytest.approx(3.75)  # this step
    assert tracker.loss.avg == pytest.approx(2.5)  # window mean -- wrong here
    # and the dict form defaults to the averages, hence the explicit .val
    assert tracker.to_dict()["loss"] == pytest.approx(2.5)
    assert tracker.to_dict(use_avg=False)["loss"] == pytest.approx(3.75)


def test_hook_is_inert_unless_the_env_var_is_set(monkeypatch):
    """A normal training run must pay nothing and write nothing."""
    monkeypatch.delenv(DUMP_ENV_VAR, raising=False)
    assert dump_dir_from_env() is None

    for off in ("", "   ", "0", "false", "FALSE", "no", "off"):
        monkeypatch.setenv(DUMP_ENV_VAR, off)
        assert dump_dir_from_env() is None, f"{off!r} should disable dumping"


def test_env_var_value_is_the_output_directory(monkeypatch, tmp_path):
    monkeypatch.setenv(DUMP_ENV_VAR, f"  {tmp_path}  ")
    resolved = dump_dir_from_env()

    assert resolved == tmp_path  # surrounding whitespace stripped

    # end to end: gate -> dump -> read back
    dump_step(resolved, 1, loss=0.5, rank=0)
    records = read_loss_jsonl(resolved / DEFAULT_LOSS_FILENAME)
    assert [r["step"] for r in records] == [1]
    assert records[0]["loss"] == pytest.approx(0.5)


def test_only_rank_zero_writes_by_default(tmp_path):
    """Six DP ranks must not produce six divergent files per step."""
    assert dump_step(tmp_path, 1, loss=1.0, rank=3) == {}
    assert not (tmp_path / DEFAULT_LOSS_FILENAME).exists()

    written = dump_step(tmp_path, 1, loss=1.0, rank=0)
    assert "loss" in written
    assert (tmp_path / DEFAULT_LOSS_FILENAME).exists()


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_the_real_training_loop_has_the_hook_in_the_right_place(entry_point):
    """Checks the shipped file, not a replay of its shape.

    The tests above pin the *semantics* of the placement; this one pins that
    each training entry point actually honours them. Source inspection rather
    than import because importing those modules pulls in torch, transformers
    and flash_attn.
    """
    import ast

    # parse rather than grep so a mention inside a comment cannot satisfy this
    source, tree = _train_loop_tree(entry_point)

    dump_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dump_step"
    ]
    reset_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reset_averages"
    ]

    assert len(dump_lines) == 1, f"expected exactly one dump_step call, got {dump_lines}"

    # The log block is the invariant that holds on *both* entry points: the dump
    # must happen before it. Only the gr00t loop additionally calls
    # reset_averages() inside that block (the pi loop prints the running average
    # without resetting, inherited from upstream), so that call is asserted
    # conditionally -- demanding it on both sides would pin a difference this
    # test has no business pinning.
    log_at = _log_block_lineno(tree)
    assert dump_lines[0] < log_at, (
        f"dump_step at line {dump_lines[0]} must precede the log_freq block at "
        f"line {log_at}, otherwise log_freq steps dump a reset or stale value"
    )
    if reset_lines:
        assert dump_lines[0] < min(reset_lines), (
            f"dump_step at line {dump_lines[0]} must precede reset_averages() at "
            f"line {min(reset_lines)}, otherwise log_freq steps dump loss=0.0"
        )

    # and it must read .val, never the bare attribute or the averaged dict
    hook = _call_source(source, tree, "dump_step")
    assert "train_tracker.loss.val" in hook, hook
    assert "to_dict()" not in hook, "to_dict() returns window averages"


def _train_loop_tree(filename="train_gr00t_n1_5.py"):
    """Parse a shipped training loop. See the note in the test above."""
    import ast
    from pathlib import Path

    src_path = Path(__file__).resolve().parents[3] / "flagos_robo" / "train" / filename
    source = src_path.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _call_lines(tree, name):
    import ast

    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        == name
    )


def _call_source(source, tree, name):
    """Exact source text of the sole call to ``name``.

    ``ast.get_source_segment`` rather than slicing ``source`` at a hardcoded
    indent such as ``"\\n        )"``: the two entry points nest the hook at
    different depths and separate it with different blank-line runs, so a
    literal slice silently matches the wrong span on one of them (or raises).
    """
    import ast

    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            node.func.attr
            if isinstance(node.func, ast.Attribute)
            else getattr(node.func, "id", None)
        )
        == name
    ]
    assert len(nodes) == 1, f"expected exactly one {name} call, got {len(nodes)}"
    segment = ast.get_source_segment(source, nodes[0])
    assert segment, f"could not recover source for the {name} call"
    return segment


def _log_block_lineno(tree):
    """Line of the ``if step % ...log_freq == 0:`` block.

    This is the shared anchor for the ordering invariant: both entry points
    have such a block, and whatever it does to the meters (reset, or merely
    print the running average) must happen *after* the dump.
    """
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "log_freq" in ast.dump(node.test):
            return node.lineno
    raise AssertionError("no `step % log_freq == 0` block in this entry point")


def _main_lineno(tree):
    import ast

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node.lineno
    raise AssertionError("no main() in this training entry point")


def _device_init_lines(tree):
    """Lines inside ``main()`` that can bring a CUDA context into existence.

    The two entry points express this differently: the gr00t loop calls
    ``set_device``/``init_process_group`` inline, while the pi loop hides both
    behind an ``init_distributed()`` helper *defined earlier in the file*. A
    bare line-number comparison against ``set_device`` therefore proves nothing
    for the pi loop -- the definition sits above ``main()`` while the call sits
    below it. Filtering to calls at or after ``main()`` makes the comparison
    mean the same thing on both sides.
    """
    main_at = _main_lineno(tree)
    lines = []
    for name in ("set_device", "init_process_group", "init_distributed"):
        lines += [ln for ln in _call_lines(tree, name) if ln >= main_at]
    return sorted(lines)


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_determinism_is_enabled_before_any_cuda_handle_exists(entry_point):
    """D1-7 ordering invariant, pinned against the shipped files.

    ``enable_determinism`` exports ``CUBLAS_WORKSPACE_CONFIG``, which cuBLAS
    reads when it creates its handle. Move this call below ``set_device`` /
    ``init_process_group`` and the export lands too late:
    ``use_deterministic_algorithms(True)`` then raises from inside some matmul
    rather than at startup -- an error whose message points nowhere near the
    line that actually caused it.
    """
    _, tree = _train_loop_tree(entry_point)

    det = _call_lines(tree, "enable_determinism")
    seed = _call_lines(tree, "set_seed")
    device_init = _device_init_lines(tree)

    assert len(det) == 1, f"expected exactly one enable_determinism call, got {det}"
    assert seed, "set_seed call vanished -- re-check this invariant"
    assert device_init, "no set_device/init_process_group/init_distributed in main()"
    assert max(seed) < det[0], (
        f"enable_determinism at line {det[0]} must come after set_seed at "
        f"{max(seed)}, so its stricter settings win"
    )
    assert det[0] < min(device_init), (
        f"enable_determinism at line {det[0]} must precede device/process-group "
        f"init at {min(device_init)}; CUBLAS_WORKSPACE_CONFIG is read at cuBLAS "
        f"handle creation"
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_dataloader_seeding_is_gated_not_unconditional(entry_point):
    """Both seeding kwargs must be present AND switched off by default.

    Passing ``worker_init_fn``/``generator`` unconditionally would change
    ordinary training runs (worker RNG streams move, and a fixed generator
    pins shuffling), which is a behaviour change nobody asked for. Passing
    neither would leave worker RNGs uncontrolled and make the two sides
    diverge for a reason that has nothing to do with the model.
    """
    source, tree = _train_loop_tree(entry_point)

    loader = _call_lines(tree, "DataLoader")
    assert len(loader) == 1, f"expected one DataLoader call, got {loader}"

    call = source[source.index("torch.utils.data.DataLoader(") :]
    call = call[: call.index("\n    )") + 6]
    assert "worker_init_fn=seed_worker if" in call, call
    assert "generator=make_dataloader_generator(seed) if" in call, call
    # the gate must be the shared one, not a locally invented condition
    assert "determinism_enabled()" in source


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_step_log_hook_shares_the_dump_hook_point(entry_point):
    """A5 ordering invariant, pinned against the shipped file.

    ``append_step_log`` must sit in the same window as ``dump_step``: after
    ``train_tracker.step()`` and before ``reset_averages()``. If it drifts past
    the log block it writes ``loss=0.0`` on every ``log_freq`` step -- and
    unlike the JSONL, nothing downstream re-parses this file as numbers, so a
    zeroed log would pass the grep and still be wrong.
    """
    source, tree = _train_loop_tree(entry_point)

    log_lines = _call_lines(tree, "append_step_log")
    dump_lines = _call_lines(tree, "dump_step")
    reset_lines = _call_lines(tree, "reset_averages")
    step_lines = _call_lines(tree, "step")
    log_at = _log_block_lineno(tree)

    assert len(log_lines) == 1, f"expected exactly one append_step_log, got {log_lines}"
    assert dump_lines and step_lines

    assert min(step_lines) < log_lines[0] < log_at, (
        f"append_step_log at {log_lines[0]} must fall between tracker.step() at "
        f"{min(step_lines)} and the log_freq block at {log_at}"
    )
    if reset_lines:
        assert log_lines[0] < min(reset_lines), (
            f"append_step_log at {log_lines[0]} must precede reset_averages() at {min(reset_lines)}"
        )
    assert dump_lines[0] < log_lines[0], "keep the two artifacts adjacent and ordered"


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_step_log_hook_reads_the_same_meters_as_the_dump(entry_point):
    """Two artifacts of one run disagreeing is worse than one missing."""
    source, tree = _train_loop_tree(entry_point)

    call = _call_source(source, tree, "append_step_log")

    assert "train_tracker.loss.val" in call, call
    assert "train_tracker.grad_norm.val" in call, call
    assert "train_tracker.lr.val" in call, call
    assert "to_dict()" not in call, "to_dict() returns window averages"
    assert "rank=rank" in call, "every rank appending to one file interleaves"


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_step_log_is_inert_unless_its_env_var_is_set(entry_point):
    """The gate must be the shared helper, not a locally invented condition."""
    source, _ = _train_loop_tree(entry_point)

    assert "step_log_dir_from_env()" in source
    assert "if step_log_dir is not None:" in source
