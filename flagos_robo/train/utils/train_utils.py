# Adapted from https://github.com/huggingface/lerobot/blob/2b304eeb841ae6c371e3dd341bbbb9dd254b07cb/src/lerobot/utils/train_utils.py

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
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.optim import Optimizer

from flagos_robo.models.utils.constants import (
    CHECKPOINTS_DIR,
    DATALOADER_STATE,
    LAST_CHECKPOINT_LINK,
    OPTIMIZER_PARAM_GROUPS,
    OPTIMIZER_STATE,
    PRETRAINED_MODEL_DIR,
    SAFETENSORS_FILE,
    SCHEDULER_STATE,
    TRAINING_STATE_DIR,
    TRAINING_STEP,
)
from flagos_robo.train.datasets.utils import flatten_dict, load_json, unflatten_dict, write_json
from flagos_robo.train.utils.io_utils import deserialize_json_into_object
from flagos_robo.train.utils.random_utils import load_rng_state, save_rng_state


class StatefulDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """DistributedSampler that can restart part-way into an epoch.

    The resume offset is supplied by the caller rather than counted here, and
    that is the whole point of the class. Counting indices as ``__iter__``
    hands them out -- the obvious implementation -- is wrong under a
    multi-worker DataLoader: ``_MultiProcessingDataLoaderIter`` pre-fetches
    ``num_workers * prefetch_factor`` batches when it is built and keeps the
    pipeline full afterwards. Those indices have left the sampler but have not
    reached the training loop, so a checkpoint would record an offset running
    ahead of real progress by ``num_workers * prefetch_factor * batch_size``
    indices, and the resumed run would silently skip that many batches -- no
    error, just a different data order than the uninterrupted run.
    (``torchdata``'s StatefulDataLoader rolls in-flight prefetch back for you;
    it is not installed in this environment.)

    So ``state_dict`` takes the number of indices this rank has actually
    consumed. Callers derive it from the training loop's ``step`` via
    :func:`consumed_indices_for_step`: ``step`` only advances after a batch has
    been used, which makes it immune to prefetch.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._offset = 0

    def __iter__(self):
        # Take the offset on the way in and clear it immediately, rather than
        # resetting after the yield loop: cycle() abandons the exhausted
        # generator instead of running it to completion, so a trailing reset
        # would never execute and the offset would be re-applied to every
        # later pass.
        offset, self._offset = self._offset, 0
        full_indices = list(super().__iter__())
        yield from full_indices[offset:]

    def state_dict(self, consumed_indices: int):
        """Snapshot the sampler. ``consumed_indices`` is required, see class doc."""
        return {"epoch": self.epoch, "yielded": consumed_indices}

    def load_state_dict(self, state_dict):
        self.set_epoch(state_dict["epoch"])
        self._offset = state_dict["yielded"]


def consumed_indices_for_step(step: int, batches_per_pass: int, batch_size: int) -> int:
    """How many indices this rank has consumed inside the current dataloader pass.

    ``batches_per_pass`` is ``len(dataloader)`` -- the batch count of one full
    pass, which ``DistributedSampler.__len__`` keeps constant regardless of any
    resume offset. Note this is *not* ``samples_per_epoch`` as the training
    loops compute it (``num_frames // (batch_size * world_size)``, a step
    count): the per-pass batch count is ``ceil(num_samples / batch_size)`` on
    this rank alone.

    The modulo self-corrects across a resume. With 10 batches per pass and a
    resume at step 13, the offset is 3 batches, the shortened pass yields 7
    batches, and ``13 + 7 == 20`` lands exactly on the next pass boundary.
    """
    if batches_per_pass <= 0:
        return 0
    return (step % batches_per_pass) * batch_size


def get_step_identifier(step: int, total_steps: int) -> str:
    num_digits = max(6, len(str(total_steps)))
    return f"{step:0{num_digits}d}"


def get_step_checkpoint_dir(output_dir: Path, total_steps: int, step: int) -> Path:
    """Returns the checkpoint sub-directory corresponding to the step number."""
    step_identifier = get_step_identifier(step, total_steps)
    return output_dir / CHECKPOINTS_DIR / step_identifier


def save_training_step(step: int, save_dir: Path) -> None:
    write_json({"step": step}, save_dir / TRAINING_STEP)


def load_training_step(save_dir: Path) -> int:
    training_step = load_json(save_dir / TRAINING_STEP)
    return training_step["step"]


def update_last_checkpoint(checkpoint_dir: Path) -> Path:
    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink():
        last_checkpoint_dir.unlink()
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    last_checkpoint_dir.symlink_to(relative_target)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    config,
    policy,
    optimizer_state_dict: dict | None = None,
    lr_scheduler=None,
    preprocessor=None,
    dataloader_state: dict | None = None,
    postprocessor=None,
    state_dict: dict | None = None,
) -> None:
    """Save a full training checkpoint.

    Creates the following directory structure:

    005000/
    ├── pretrained_model/
    │   ├── config.json                     # policy config
    │   ├── model.safetensors               # policy weights
    │   ├── vlm_config/                     # VLM artifacts (model-specific)
    │   ├── train_config.json               # training config
    │   ├── policy_preprocessor.json        # preprocessor config
    │   ├── policy_preprocessor_step_*.safetensors
    │   ├── policy_postprocessor.json       # postprocessor config
    │   └── policy_postprocessor_step_*.safetensors
    └── training_state/
        ├── training_step.json
        ├── optimizer_state.safetensors
        ├── optimizer_param_groups.json
        ├── scheduler_state.json
        └── rng_state.safetensors

    Args:
        checkpoint_dir: Root checkpoint directory (e.g., checkpoints/005000).
        step: Current training step.
        config: TrainConfig instance.
        policy: The policy model.
        optimizer_state_dict: Pre-gathered optimizer state dict (e.g. from
            get_optimizer_state_dict for FSDP2).
        lr_scheduler: Optional LR scheduler.
        preprocessor: Optional preprocessor pipeline.
        postprocessor: Optional postprocessor pipeline.
        state_dict: Pre-gathered model state dict (e.g. from FSDP2). If None, uses
            policy.state_dict().
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    policy.save_pretrained(pretrained_dir, state_dict=state_dict)
    config._save_pretrained(pretrained_dir)
    if preprocessor is not None:
        preprocessor.save_pretrained(pretrained_dir)
    if postprocessor is not None:
        postprocessor.save_pretrained(pretrained_dir)
    save_training_state(checkpoint_dir, step, optimizer_state_dict, lr_scheduler, dataloader_state)


def save_training_state(
    checkpoint_dir: Path,
    train_step: int,
    optimizer_state_dict: dict | None = None,
    scheduler=None,
    dataloader_state: dict | None = None,
) -> None:
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir)
    # Saves this rank's RNG file. Callers that run save_checkpoint under an
    # `if is_main_process` guard must additionally call save_rng_state() on the
    # other ranks (see save_rng_state_only), or those ranks resume from
    # rank 0's stream.
    save_rng_state(save_dir)
    if optimizer_state_dict is not None:
        save_optimizer_state(optimizer_state_dict, save_dir)
    if scheduler is not None:
        save_scheduler_state(scheduler, save_dir)
    if dataloader_state is not None:
        save_dataloader_state(dataloader_state, save_dir)


def save_rng_state_only(checkpoint_dir: Path) -> None:
    """Write just this rank's RNG file into a checkpoint's training_state dir.

    Exists for the non-main ranks. ``save_checkpoint`` writes model/optimizer
    state through FSDP2 collectives and is therefore called on rank 0 only, but
    RNG state is per-rank and cannot be gathered that way — so every other rank
    calls this to deposit its own file alongside. Safe to call concurrently:
    the directory is created with ``exist_ok`` and each rank writes a distinct
    filename.
    """
    save_dir = checkpoint_dir / TRAINING_STATE_DIR
    save_rng_state(save_dir)


def load_model_state_fsdp2(model: nn.Module, pretrained_dir: Path) -> None:
    """Load a full (non-sharded) safetensors checkpoint into an FSDP2-wrapped model."""
    full_sd = load_file(str(pretrained_dir / SAFETENSORS_FILE))
    set_model_state_dict(model, full_sd, options=StateDictOptions(full_state_dict=True))


def load_optimizer_state_fsdp2(model: nn.Module, optimizer: Optimizer, save_dir: Path) -> None:
    """Load optimizer state into an FSDP2-wrapped model's optimizer.

    FSDP2 optimizer state uses FQN string keys (not integer indices).
    Params that never received gradients won't have state entries;
    pad them with empty dicts so set_optimizer_state_dict doesn't KeyError.
    """
    flat_optim = load_file(str(save_dir / OPTIMIZER_STATE))
    optim_state = unflatten_dict(flat_optim)
    param_groups = load_json(save_dir / OPTIMIZER_PARAM_GROUPS)
    all_pg_params = {p for g in param_groups for p in g["params"]}
    for p in all_pg_params - set(optim_state.get("state", {}).keys()):
        optim_state.setdefault("state", {})[p] = {}
    optim_state["param_groups"] = param_groups
    set_optimizer_state_dict(
        model, optimizer, optim_state, options=StateDictOptions(full_state_dict=True)
    )


def load_training_state_fsdp2(
    checkpoint_dir: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
) -> tuple[int, dict | None]:
    """Load full training state into an FSDP2-wrapped model and optimizer.

    Returns a tuple of (training step, dataloader state dict or None).
    """
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    training_state_dir = checkpoint_dir / TRAINING_STATE_DIR

    load_model_state_fsdp2(model, pretrained_dir)
    step = load_training_step(training_state_dir)
    load_rng_state(training_state_dir)
    load_optimizer_state_fsdp2(model, optimizer, training_state_dir)

    if scheduler is not None:
        load_scheduler_state(scheduler, training_state_dir)

    dl_state = None
    dl_state_path = training_state_dir / DATALOADER_STATE
    if dl_state_path.exists():
        dl_state = load_dataloader_state(training_state_dir)

    return step, dl_state


def replay_dataloader_to_step(dl_iter, sampler, step: int, samples_per_epoch: int) -> int:
    """Fast-forward a cycling dataloader iterator to ``step``, faithfully.

    ``DistributedSampler.__iter__`` seeds a generator private to the sampler
    with ``self.seed + self.epoch`` (see torch/utils/data/distributed.py), so
    its permutation is a function of the EPOCH ALONE -- global RNG state does
    not enter into it. Replaying the iterator without advancing the sampler
    epoch therefore re-serves epoch 0's permutation on every epoch boundary
    ``cycle()`` crosses, while the uninterrupted run was already on epoch
    1, 2, ... at those same boundaries. The resumed run then trains on a
    different data order than the run it claims to continue -- no error, no
    visible symptom in the loss curve.

    Upstream's own warning is the other half of this: ``set_epoch`` must be
    called BEFORE the iterator is created to take effect, so each boundary
    sets the epoch and then lets ``cycle()`` build the next iterator.

    Returns the epoch the sampler has been left on, which is the epoch the
    training loop must continue from.
    """
    epoch = 0
    for i in range(step):
        next(dl_iter)
        if samples_per_epoch > 0 and (i + 1) % samples_per_epoch == 0:
            epoch += 1
            # Takes effect when cycle() re-iters the exhausted dataloader,
            # which is exactly the next epoch boundary.
            sampler.set_epoch(epoch)
    return epoch


def save_optimizer_state(state_dict: dict, save_dir: Path) -> None:
    state_dict = dict(state_dict)
    param_groups = state_dict.pop("param_groups")
    flat_state = flatten_dict(state_dict)
    save_file(flat_state, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)


def save_scheduler_state(scheduler, save_dir: Path) -> None:
    write_json(scheduler.state_dict(), save_dir / SCHEDULER_STATE)


def load_scheduler_state(scheduler, save_dir: Path):
    state_dict = deserialize_json_into_object(save_dir / SCHEDULER_STATE, scheduler.state_dict())
    scheduler.load_state_dict(state_dict)
    return scheduler


def save_dataloader_state(state: dict, save_dir: Path) -> None:
    write_json(state, save_dir / DATALOADER_STATE)


def load_dataloader_state(save_dir: Path) -> dict:
    return load_json(save_dir / DATALOADER_STATE)
