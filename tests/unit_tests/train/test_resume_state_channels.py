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
"""W5 B3 - the two hidden-state channels a resume has to carry.

``test_checkpoint_resume.py`` pins the *replay* contract, which is now the
fallback for checkpoints written before this change. This file pins the two
channels that replaced it, because each fails silently rather than loudly:

* **Sampler offset.** ``StatefulDistributedSampler`` records how far into the
  epoch a rank got, so a resume is one offset assignment instead of N calls to
  ``next(dl_iter)``. If the entry point builds a plain ``DistributedSampler``
  again, or forgets to hand ``sampler.state_dict()`` to ``save_checkpoint``,
  resume quietly falls back to O(step) replay -- correct, but linear, which is
  exactly what the resume-cost requirement rules out.

* **Per-rank RNG.** ``save_checkpoint`` runs under ``if is_main_process`` because
  it writes state gathered through FSDP2 collectives. RNG state cannot be
  gathered that way: it is per-rank. So a main-process-only save leaves every
  other rank restoring rank 0's stream -- no error, just a divergent run on
  anything above one GPU.

The AST half inspects source rather than importing, matching
``test_checkpoint_resume.py``: these entry points pull in torch, transformers
and flash_attn. The runtime half is skipped where torch is absent.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ENTRY_POINTS = ["train_gr00t_n1_5.py", "train_pi.py"]

REPO_ROOT = Path(__file__).resolve().parents[3]
TRAIN_DIR = REPO_ROOT / "flagos_robo" / "train"
RANDOM_UTILS = TRAIN_DIR / "utils" / "random_utils.py"


def _tree(entry_point):
    source = (TRAIN_DIR / entry_point).read_text(encoding="utf-8")
    return source, ast.parse(source)


def _called_name(node):
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def _calls(tree, name):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _called_name(node) == name
    ]


def _kwarg(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


# --------------------------------------------------------------------------
# Sampler offset channel
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_entry_point_builds_the_stateful_sampler(entry_point):
    """A plain DistributedSampler has no offset to restore.

    Reverting this one line is the cheapest way to silently lose O(1) resume:
    the ``hasattr(sampler, "load_state_dict")`` guard in the resume block would
    simply stop matching and fall through to replay.
    """
    _, tree = _tree(entry_point)

    assert _calls(tree, "StatefulDistributedSampler"), (
        "expected StatefulDistributedSampler() -- a plain DistributedSampler "
        "cannot report how far into the epoch this rank got"
    )
    assert not _calls(tree, "DistributedSampler"), (
        "a bare DistributedSampler() call remains; it yields no resumable state"
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_sampler_state_is_handed_to_save_checkpoint(entry_point):
    """The offset has to reach the checkpoint, not just exist in memory.

    ``save_checkpoint`` writes ``dataloader_state.json`` only when it is given
    a ``dataloader_state``; omit the keyword and every checkpoint looks complete
    but resumes through the replay fallback.
    """
    _, tree = _tree(entry_point)

    saves = _calls(tree, "save_checkpoint")
    assert saves, "no save_checkpoint() call"

    for save in saves:
        state = _kwarg(save, "dataloader_state")
        assert state is not None, (
            f"save_checkpoint() at line {save.lineno} passes no dataloader_state"
        )
        assert _calls(ast.Module(body=[ast.Expr(value=state)], type_ignores=[]),
                      "state_dict"), (
            "dataloader_state must come from the sampler's state_dict()"
        )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_resume_prefers_the_saved_offset_over_replay(entry_point):
    """Both paths must exist, and the offset one must be the preferred branch.

    Keeping only the fast path would break checkpoints written before the
    sampler carried state; keeping only replay would make resume cost grow with
    ``step``. So the shape asserted here is if/else, with ``load_state_dict``
    in the ``if`` body and ``replay_dataloader_to_step`` in the ``else``.
    """
    _, tree = _tree(entry_point)

    load_calls = _calls(tree, "load_state_dict")
    assert load_calls, "resume never calls sampler.load_state_dict()"

    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and node.orelse
        and any(
            _called_name(c) == "load_state_dict"
            for stmt in node.body
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call)
        )
        and any(
            _called_name(c) == "replay_dataloader_to_step"
            for stmt in node.orelse
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call)
        )
    ]
    assert branches, (
        "expected `if <saved offset>: sampler.load_state_dict(...) "
        "else: replay_dataloader_to_step(...)` so old checkpoints stay resumable"
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_offset_path_does_not_snapshot_the_rng(entry_point):
    """Restoring an offset draws nothing, so bracketing it with a snapshot is wrong.

    The snapshot exists because *replaying* consumes RNG. If it migrated up to
    wrap the fast path too, it would restore a stale RNG over the one the
    checkpoint just loaded.
    """
    _, tree = _tree(entry_point)

    # Only the inner if/else counts. The enclosing `if resume_from:` contains
    # both branches, so matching on "body mentions load_state_dict" alone would
    # pick it up and see the fallback's snapshot as if it were on the fast path.
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            _called_name(c) == "load_state_dict"
            for stmt in node.body
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call)
        )
        and any(
            _called_name(c) == "replay_dataloader_to_step"
            for stmt in node.orelse
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call)
        )
    ]
    assert branches, "no `if <saved offset>: ... else: replay ...` branch found"

    for branch in branches:
        in_fast_path = {
            _called_name(c)
            for stmt in branch.body
            for c in ast.walk(stmt)
            if isinstance(c, ast.Call)
        }
        assert "serialize_rng_state" not in in_fast_path, (
            "the offset path draws nothing from the dataloader; snapshotting "
            "the RNG there would overwrite the restored state"
        )


# --------------------------------------------------------------------------
# Per-rank RNG channel
# --------------------------------------------------------------------------


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_non_main_ranks_write_their_own_rng(entry_point):
    """rank 0 writing RNG for everyone means everyone resumes rank 0's stream."""
    _, tree = _tree(entry_point)

    assert _calls(tree, "save_rng_state_only"), (
        "no save_rng_state_only() call -- only rank 0's RNG would be saved, so "
        "every other rank resumes from rank 0's stream"
    )


@pytest.mark.parametrize("entry_point", ENTRY_POINTS)
def test_checkpoint_dir_is_computed_outside_the_main_process_guard(entry_point):
    """The non-main ranks need the same directory to deposit RNG into.

    This is the easy half to get wrong: hoisting ``save_rng_state_only`` out of
    the guard is useless if the path it writes to is still computed inside it.
    """
    _, tree = _tree(entry_point)

    dir_calls = _calls(tree, "get_step_checkpoint_dir")
    assert dir_calls, "no get_step_checkpoint_dir() call"

    main_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "is_main_process"
    ]
    guarded_spans = [
        (g.body[0].lineno, g.body[-1].end_lineno) for g in main_guards if g.body
    ]

    for call in dir_calls:
        inside = [s for s in guarded_spans if s[0] <= call.lineno <= s[1]]
        assert not inside, (
            f"get_step_checkpoint_dir() at line {call.lineno} sits inside an "
            "`if is_main_process:` body, so non-main ranks have no directory to "
            "write their RNG file into"
        )


def test_rng_filename_is_rank_scoped_with_a_legacy_fallback():
    """Source-level check of the naming contract, importable without torch."""
    source = RANDOM_UTILS.read_text(encoding="utf-8")
    tree = ast.parse(source)

    names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    assert "rng_state_filename" in names, "no rank-scoped RNG filename helper"

    for fn in ("save_rng_state", "load_rng_state"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == fn)
        args = [a.arg for a in node.args.args]
        assert "rank" in args, f"{fn}() takes no rank argument"
        # Callers in train_utils.py pass save_dir only, so rank must default.
        assert len(node.args.defaults) >= 1, (
            f"{fn}(save_dir) must stay callable with one argument -- "
            "train_utils.py calls it that way"
        )

    load = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_rng_state"
    )
    raises = [n for n in ast.walk(load) if isinstance(n, ast.Raise)]
    assert raises, (
        "load_rng_state() must fail loudly when neither the rank-scoped nor the "
        "legacy file is present, rather than silently leaving the RNG as-is"
    )
