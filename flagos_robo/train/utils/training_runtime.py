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

"""Shared distributed data and checkpoint lifecycle for model training loops."""

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
)
from torch.optim import Optimizer

from flagos_robo.logger import logger
from flagos_robo.train.utils.random_utils import deserialize_rng_state, serialize_rng_state
from flagos_robo.train.utils.train_utils import (
    StatefulDistributedSampler,
    consumed_indices_for_step,
    get_step_checkpoint_dir,
    load_training_state_fsdp2,
    replay_dataloader_to_step,
    save_checkpoint,
    save_rng_state_only,
    update_last_checkpoint,
)
from flagos_robo.utils.determinism import (
    determinism_enabled,
    make_dataloader_generator,
    seed_worker,
)


def build_training_dataloader(dataset, system, seed: int):
    """Build the same rank-aware, resumable DataLoader for every model."""
    sampler = StatefulDistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=system.shuffle,
        drop_last=False,
    )
    deterministic = determinism_enabled()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=system.num_workers,
        batch_size=system.batch_size,
        shuffle=False,
        sampler=sampler,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if system.num_workers > 0 else None,
        worker_init_fn=seed_worker if deterministic else None,
        generator=make_dataloader_generator(seed) if deterministic else None,
    )
    return dataloader, sampler


def resume_training(
    resume_from: str | None,
    policy,
    optimizer: Optimizer,
    lr_scheduler: Any,
    dl_iter,
    sampler: StatefulDistributedSampler,
    samples_per_epoch: int,
) -> tuple[int, int]:
    """Restore model and data position, including old checkpoints without sampler state.

    Loading the FSDP2 state is collective: every rank must call this function.
    Replaying old checkpoints consumes global RNG, so restore it afterward.
    """
    if not resume_from:
        return 0, 0

    step, dl_state = load_training_state_fsdp2(Path(resume_from), policy, optimizer, lr_scheduler)
    if dl_state is not None and hasattr(sampler, "load_state_dict"):
        sampler.load_state_dict(dl_state)
        epoch = dl_state.get("epoch", 0)
    else:
        saved_rng = serialize_rng_state()
        epoch = replay_dataloader_to_step(dl_iter, sampler, step, samples_per_epoch)
        deserialize_rng_state(saved_rng)
    logger.info(f"Resumed from checkpoint at step {step} (epoch {epoch})")
    return step, epoch


def save_training_checkpoint(
    *,
    config,
    step: int,
    policy,
    policy_to_save,
    optimizer: Optimizer,
    lr_scheduler: Any,
    preprocessor,
    postprocessor,
    dataloader,
    sampler: StatefulDistributedSampler,
    is_main_process: bool,
) -> None:
    """Gather state on all ranks, save shared files on rank 0 and RNG on each rank."""
    dist.barrier()
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state_dict = get_model_state_dict(policy, options=options)
    optimizer_state_dict = get_optimizer_state_dict(policy, optimizer, options=options)
    output_dir = Path(config.system.checkpoint.output_directory)
    checkpoint_dir = get_step_checkpoint_dir(output_dir, config.system.train_steps, step)

    if is_main_process:
        logger.info(f"Saving checkpoint at step {step}")
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=step,
            config=config,
            policy=policy_to_save,
            optimizer_state_dict=optimizer_state_dict,
            lr_scheduler=lr_scheduler,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            state_dict=state_dict,
            # The sampler runs ahead by DataLoader prefetch depth. Use the
            # consumed training step, rather than its current iterator offset.
            dataloader_state=sampler.state_dict(
                consumed_indices_for_step(step, len(dataloader), config.system.batch_size)
            )
            if hasattr(sampler, "state_dict")
            else None,
        )
        update_last_checkpoint(checkpoint_dir)
    else:
        save_rng_state_only(checkpoint_dir)
    dist.barrier()
