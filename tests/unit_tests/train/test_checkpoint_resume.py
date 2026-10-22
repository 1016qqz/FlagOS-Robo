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
"""Resume/save symmetry invariants for every training entry point.

``config.system.checkpoint.resume_from`` is part of the config schema
(``train_config.py``), so a loop that silently ignores it starts from step 0
while the operator believes the run resumed -- no error, no log line, just a
quietly restarted training run.

The harder half is that resume and save are *one* contract. ``save_checkpoint``
writes ``optimizer_state.safetensors`` only when it is handed an
``optimizer_state_dict``, while ``load_training_state_fsdp2`` loads that file
unconditionally. An entry point that saves without the optimizer state
therefore produces checkpoints that look complete and fail on resume. Both
halves are pinned here so neither can drift away from the other.

Source inspection rather than import: these modules pull in torch,
transformers and flash_attn.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ENTRY_POINTS = ["train_gr00t_n1_5.py", "train_pi.py"]

TRAIN_DIR = Path(__file__).resolve().parents[3] / "flagos_robo" / "train"


def _tree(entry_point):
    source = (TRAIN_DIR / entry_point).read_text(encoding="utf-8")
    return source, ast.parse(source)


def _called_name(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _call_lines(tree, name):
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == name
    )


def _sole_call(tree, name):
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == name
    ]
    assert len(nodes) == 1, f"expected exactly one {name} call, got {len(nodes)}"
    return nodes[0]


def _kwargs(call):
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


def _main(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("no main() in this training entry point")


def _name_lines(tree, name):
    return sorted(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id == name
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_resume_from_is_actually_consumed(entry_point):
    """The config field must reach ``load_training_state_fsdp2``.

    Reading the attribute is not enough -- it has to be the thing that gates
    the load and supplies its path.
    """
    source, tree = _tree(entry_point)

    assert "config.system.checkpoint.resume_from" in source, (
        "resume_from is in the config schema but this loop never reads it: a "
        "resume request would be silently ignored and the run would restart "
        "from step 0"
    )

    load = _sole_call(tree, "load_training_state_fsdp2")
    assert "resume_from" in {n.id for n in ast.walk(load) if isinstance(n, ast.Name)}, (
        "load_training_state_fsdp2 must be fed resume_from, not a hardcoded path"
    )

    # ... and the load must be guarded, or an ordinary from-scratch run would
    # try to read a checkpoint that does not exist.
    guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "resume_from" in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        and node.lineno <= load.lineno <= node.body[-1].end_lineno
    ]
    assert guards, "the resume load must sit under an `if resume_from:` guard"


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_resume_restores_the_epoch_not_only_the_step(entry_point):
    """``set_epoch`` seeds the sampler permutation.

    Restoring ``step`` but leaving ``epoch`` at 0 re-serves epoch 0's order on
    a resumed run, so the resumed and uninterrupted runs see different data in
    a different order -- invisible in the loss curve, fatal to alignment.
    """
    _, tree = _tree(entry_point)

    epoch_recovery = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "resume_from" in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        and "samples_per_epoch"
        in {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
    ]
    assert epoch_recovery, (
        "expected `if resume_from and samples_per_epoch > 0:` recovering epoch "
        "from step; without the samples_per_epoch guard this is also a "
        "division by zero on a dataset smaller than one global batch"
    )

    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "epoch" for t in node.targets)
        for node in ast.walk(epoch_recovery[0])
    ), "the guarded block must assign epoch"

    # set_epoch has to run after that recovery, otherwise it seeds with 0
    set_epoch_lines = _call_lines(tree, "set_epoch")
    assert set_epoch_lines, "no set_epoch() call -- sampler would never reshuffle"
    assert any(ln > epoch_recovery[0].lineno for ln in set_epoch_lines), (
        "set_epoch() must be called after the epoch is recovered from step"
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_dataloader_replay_does_not_burn_the_restored_rng(entry_point):
    """Fast-forwarding the iterator consumes RNG; it must be put back.

    The sampler has no resumable state, so reaching step N's data position
    means replaying ``next(dl_iter)`` N times. That draws from the RNG the
    checkpoint just restored, so without a snapshot/restore around the replay
    the resumed run diverges from the uninterrupted one immediately.
    """
    _, tree = _tree(entry_point)

    ser = _call_lines(tree, "serialize_rng_state")
    deser = _call_lines(tree, "deserialize_rng_state")
    assert ser and deser, (
        "expected serialize_rng_state()/deserialize_rng_state() bracketing the "
        "dataloader replay"
    )

    # `range(step)` with a SINGLE argument -- the main training loop is
    # `range(step, train_steps)` and also calls next(dl_iter), so an arity
    # check is what separates the replay from the loop it fast-forwards to.
    replays = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and _called_name(node.iter) == "range"
        and len(node.iter.args) == 1
        and isinstance(node.iter.args[0], ast.Name)
        and node.iter.args[0].id == "step"
        and any(
            _called_name(c) == "next" for c in ast.walk(node) if isinstance(c, ast.Call)
        )
    ]
    assert len(replays) == 1, (
        "expected exactly one `for _ in range(step): next(dl_iter)` replay "
        f"loop, found {len(replays)}"
    )

    replay = replays[0]
    assert any(ln < replay.lineno for ln in ser), "snapshot the RNG before replaying"
    assert any(
        ln > replay.end_lineno for ln in deser
    ), "restore the RNG after replaying"


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_save_persists_the_optimizer_state_resume_will_demand(entry_point):
    """Resume and save are one contract, and this is the half that drifts.

    ``save_training_state`` writes ``optimizer_state.safetensors`` only when
    handed an ``optimizer_state_dict``, but ``load_training_state_fsdp2``
    reads that file unconditionally. A loop that saves without it produces
    checkpoints that look complete, pass every other test, and raise
    ``FileNotFoundError`` the first time anyone resumes.
    """
    _, tree = _tree(entry_point)

    save = _sole_call(tree, "save_checkpoint")
    assert "optimizer_state_dict" in _kwargs(save), (
        "save_checkpoint() must be passed optimizer_state_dict, otherwise the "
        "checkpoint has no optimizer_state.safetensors and resume raises "
        "FileNotFoundError"
    )

    getter = _sole_call(tree, "get_optimizer_state_dict")
    # Collective: every rank must call it, so it may not live inside a
    # rank-0-only block. Getting this wrong hangs the job instead of failing.
    for node in ast.walk(_main(tree)):
        if not isinstance(node, ast.If):
            continue
        test_names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        test_attrs = {
            n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)
        }
        if not ({"is_main_process", "rank"} & (test_names | test_attrs)):
            continue
        assert not (
            node.body[0].lineno <= getter.lineno <= node.body[-1].end_lineno
        ), (
            "get_optimizer_state_dict is a collective -- inside a rank-gated "
            f"block (line {node.lineno}) it deadlocks every non-zero rank"
        )
