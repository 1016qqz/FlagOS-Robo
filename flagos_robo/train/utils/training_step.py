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

"""Shared training-step mechanics; policies keep their own forward and loss formats."""

import time
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from flagos_robo.platforms import get_platform
from flagos_robo.train.utils.logging_utils import MetricsTracker


def cycle_dataloader(iterable: Any) -> Iterator[Any]:
    """Restart a DataLoader without caching its batches or worker processes."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def run_training_step(
    train_metrics: MetricsTracker,
    policy: nn.Module,
    batch: Any,
    optimizer: Optimizer,
    *,
    loss_from_output: Callable[[Any], Tensor],
    update_target: nn.Module,
    use_amp: bool,
    grad_clip_norm: float,
    lr_scheduler: Any = None,
    lock: Any = None,
    zero_grad_before: bool = False,
    train_mode_each_step: bool = False,
) -> MetricsTracker:
    """Run the common optimizer lifecycle around a model-specific loss adapter.

    PI clears gradients after each update and enters train mode on every step;
    GR00T clears gradients before each update. Keep those established orders
    explicit while sharing the forward/backward/clip/step/metrics sequence.
    ``policy(batch)`` must be used so FSDP's forward hooks still run.
    """
    start_time = time.perf_counter()
    if train_mode_each_step:
        policy.train()
    if zero_grad_before:
        optimizer.zero_grad()

    autocast_context = (
        torch.amp.autocast(get_platform().amp_device_type(), dtype=torch.bfloat16)
        if use_amp
        else nullcontext()
    )
    with autocast_context:
        loss = loss_from_output(policy(batch))
    loss.backward()

    clip_value = grad_clip_norm if grad_clip_norm > 0 else float("inf")
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_value)

    with lock if lock is not None else nullcontext():
        optimizer.step()
    if not zero_grad_before:
        optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    update = getattr(update_target, "update", None)
    if callable(update):
        update()

    train_metrics.loss = loss.item()
    if hasattr(grad_norm, "full_tensor"):
        grad_norm = grad_norm.full_tensor()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics
