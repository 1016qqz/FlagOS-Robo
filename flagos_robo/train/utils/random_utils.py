# Adapted from https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/utils/random_utils.py

#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from flagos_robo.models.utils.constants import RNG_STATE
from flagos_robo.platforms import get_platform
from flagos_robo.train.datasets.utils import flatten_dict, unflatten_dict


def _optional_platform():
    """Return the accelerator platform, or ``None`` when there is none.

    The two RNG helpers below both ask the same question -- "is there an
    accelerator whose RNG state I should also save/restore?" -- and both guard
    the answer with ``is_available()``. On a host with no accelerator the
    registry is empty and ``get_platform()`` raises RuntimeError
    (``platform_manager.py:89``), which turns that question into a crash
    instead of a "no". Checkpointing on CPU is a supported path
    (``device_processor.get_safe_torch_device`` accepts "cpu"), so the honest
    answer there is ``None`` and the caller skips the accelerator half.

    This is deliberately narrower than ``_cur_platform()``: anything that
    genuinely *needs* a device must keep getting the RuntimeError.
    """
    try:
        return get_platform()
    except RuntimeError:
        return None


def _cur_platform():
    """Resolve the accelerator platform on demand, not at import time.

    This used to be a module-level ``cur_platform = get_platform()``. That made
    importing this module impossible on a machine with no accelerator, because
    ``get_platform()`` raises RuntimeError when the registry is empty
    (``platform_manager.py:89``) -- registration is gated on ``is_available()``.
    The no-accelerator CI job could therefore not even collect the tests that
    import this file.

    Only the two RNG helpers below need the platform, and both of them guard
    the use with ``is_available()``, so deferring the lookup costs nothing and
    does not change behaviour on a machine that does have an accelerator.
    """
    return get_platform()


def __getattr__(name):
    """Keep ``random_utils.cur_platform`` readable as a module attribute.

    ``tests/unit_tests/platforms/test_platform_rng.py:108`` asserts this
    attribute is a PlatformBase instance -- it guards the variable-bound call
    shape that the W1 API survey's grep for ``get_platform().<method>`` could
    not see. Resolving it through the module ``__getattr__`` hook (PEP 562)
    keeps that assertion meaningful while still deferring the lookup, so the
    RuntimeError now surfaces when the attribute is read rather than when the
    module is imported.
    """
    if name == "cur_platform":
        return get_platform()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def serialize_python_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `random` in the form of a flat dict[str, torch.Tensor] to be saved using
    `safetensors.save_file()` or `torch.save()`.
    """
    py_state = random.getstate()
    return {
        "py_rng_version": torch.tensor([py_state[0]], dtype=torch.int64),
        "py_rng_state": torch.tensor(py_state[1], dtype=torch.int64),
    }


def deserialize_python_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `random` from a dictionary produced by `serialize_python_rng_state()`.
    """
    py_state = (
        rng_state_dict["py_rng_version"].item(),
        tuple(rng_state_dict["py_rng_state"].tolist()),
        None,
    )
    random.setstate(py_state)


def serialize_numpy_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `numpy` in the form of a flat dict[str, torch.Tensor] to be saved using
    `safetensors.save_file()` or `torch.save()`.
    """
    np_state = np.random.get_state()
    # Ensure no breaking changes from numpy
    assert np_state[0] == "MT19937"
    return {
        "np_rng_state_values": torch.tensor(np_state[1], dtype=torch.int64),
        "np_rng_state_index": torch.tensor([np_state[2]], dtype=torch.int64),
        "np_rng_has_gauss": torch.tensor([np_state[3]], dtype=torch.int64),
        "np_rng_cached_gaussian": torch.tensor([np_state[4]], dtype=torch.float32),
    }


def deserialize_numpy_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `numpy` from a dictionary produced by `serialize_numpy_rng_state()`.
    """
    np_state = (
        "MT19937",
        rng_state_dict["np_rng_state_values"].numpy(),
        rng_state_dict["np_rng_state_index"].item(),
        rng_state_dict["np_rng_has_gauss"].item(),
        rng_state_dict["np_rng_cached_gaussian"].item(),
    )
    np.random.set_state(np_state)


def serialize_torch_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `torch` in the form of a flat dict[str, torch.Tensor] to be saved using
    `safetensors.save_file()` or `torch.save()`.
    """
    torch_rng_state_dict = {"torch_rng_state": torch.get_rng_state()}
    cur_platform = _optional_platform()
    if cur_platform is not None and cur_platform.is_available():
        torch_rng_state_dict["torch_cuda_rng_state"] = cur_platform.get_rng_state()
    return torch_rng_state_dict


def deserialize_torch_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `torch` from a dictionary produced by `serialize_torch_rng_state()`.
    """
    torch.set_rng_state(rng_state_dict["torch_rng_state"])
    cur_platform = _optional_platform()
    if (
        cur_platform is not None
        and cur_platform.is_available()
        and "torch_cuda_rng_state" in rng_state_dict
    ):
        cur_platform.set_rng_state(rng_state_dict["torch_cuda_rng_state"])


def serialize_rng_state() -> dict[str, torch.Tensor]:
    """
    Returns the rng state for `random`, `numpy`, and `torch`, in the form of a flat
    dict[str, torch.Tensor] to be saved using `safetensors.save_file()` `torch.save()`.
    """
    py_rng_state_dict = serialize_python_rng_state()
    np_rng_state_dict = serialize_numpy_rng_state()
    torch_rng_state_dict = serialize_torch_rng_state()

    return {
        **py_rng_state_dict,
        **np_rng_state_dict,
        **torch_rng_state_dict,
    }


def deserialize_rng_state(rng_state_dict: dict[str, torch.Tensor]) -> None:
    """
    Restores the rng state for `random`, `numpy`, and `torch` from a dictionary produced by
    `serialize_rng_state()`.
    """
    py_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("py")}
    np_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("np")}
    torch_rng_state_dict = {k: v for k, v in rng_state_dict.items() if k.startswith("torch")}

    deserialize_python_rng_state(py_rng_state_dict)
    deserialize_numpy_rng_state(np_rng_state_dict)
    deserialize_torch_rng_state(torch_rng_state_dict)


def rng_state_filename(rank: int) -> str:
    """Return the RNG state filename for ``rank``.

    Every rank advances its own python/numpy/torch streams: the dataloader hands
    each rank a different shard, and dropout/noise draw per device. A single
    shared file therefore cannot describe the cluster — whichever rank wrote it
    last would win, and on resume every rank would adopt that one rank's stream.
    So each rank gets its own file.

    ``rank`` is required. The legacy shared name is deliberately NOT reachable
    through this function: new code never writes it, and ``load_rng_state``
    reads it straight from the ``RNG_STATE`` constant. A ``None`` sentinel here
    would also clash with ``save_rng_state(rank=None)``, where ``None`` means
    "detect my own rank" rather than "use the legacy name".
    """
    stem, _, suffix = RNG_STATE.rpartition(".")
    return f"{stem}_rank{rank}.{suffix}"


def _current_rank() -> int:
    """Rank of this process, or 0 when not running under torch.distributed."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def save_rng_state(save_dir: Path, rank: int | None = None) -> None:
    """Persist this rank's RNG state.

    Must be called by EVERY rank, not just the main process: the file name is
    rank-scoped, so a main-process-only call would leave the other ranks with
    no state to restore.
    """
    if rank is None:
        rank = _current_rank()
    save_dir.mkdir(parents=True, exist_ok=True)
    rng_state_dict = serialize_rng_state()
    flat_rng_state_dict = flatten_dict(rng_state_dict)
    save_file(flat_rng_state_dict, save_dir / rng_state_filename(rank))


def load_rng_state(save_dir: Path, rank: int | None = None) -> None:
    """Restore this rank's RNG state.

    Prefers the rank-scoped file. Falls back to the legacy shared file so that
    checkpoints written before per-rank saving existed remain resumable; raises
    FileNotFoundError only when neither is present.
    """
    if rank is None:
        rank = _current_rank()
    path = save_dir / rng_state_filename(rank)
    if not path.exists():
        legacy = save_dir / RNG_STATE
        if not legacy.exists():
            raise FileNotFoundError(
                f"no RNG state for rank {rank}: neither {path.name} nor {legacy.name} "
                f"exists in {save_dir}"
            )
        path = legacy
    flat_rng_state_dict = load_file(path)
    rng_state_dict = unflatten_dict(flat_rng_state_dict)
    deserialize_rng_state(rng_state_dict)
