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

from types import SimpleNamespace

import pytest
import torch

from flagos_robo.train.utils.training_step import cycle_dataloader, run_training_step


class _Policy(torch.nn.Module):
    def __init__(self, events, output_format, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, device=device))
        self.events = events
        self.output_format = output_format

    def forward(self, batch):
        self.events.append("forward")
        loss = (self.weight * batch).square()
        if self.output_format == "pi":
            return loss, {"loss": loss.item()}
        return {"loss": loss}

    def training_loss(self, output):
        return output[0] if self.output_format == "pi" else output["loss"]

    def update(self):
        self.events.append("update")


class _Optimizer(torch.optim.SGD):
    def __init__(self, params, events):
        super().__init__(params, lr=0.1)
        self.events = events

    def zero_grad(self, *args, **kwargs):
        self.events.append("zero_grad")
        return super().zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.events.append("step")
        return super().step(*args, **kwargs)


@pytest.mark.parametrize(
    ("output_format", "zero_grad_before", "expected_events"),
    [
        ("pi", False, ["forward", "step", "zero_grad", "update"]),
        ("gr00t", True, ["zero_grad", "forward", "step", "update"]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_training_step_keeps_model_output_and_gradient_order(
    output_format, zero_grad_before, expected_events, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    events = []
    policy = _Policy(events, output_format, device)
    optimizer = _Optimizer(policy.parameters(), events)
    metrics = SimpleNamespace()

    result = run_training_step(
        metrics,
        policy,
        torch.tensor(3.0, device=device),
        optimizer,
        loss_from_output=policy.training_loss,
        update_target=policy,
        use_amp=device == "cuda",
        grad_clip_norm=0.0,
        zero_grad_before=zero_grad_before,
    )

    assert result is metrics
    assert metrics.loss == pytest.approx(36.0)
    assert metrics.lr == pytest.approx(0.1)
    assert policy.weight.item() < 2.0
    assert events == expected_events


def test_cycle_dataloader_recreates_iterator_without_caching_batches():
    class Loader:
        def __init__(self):
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter([self.iterations])

    loader = Loader()
    batches = cycle_dataloader(loader)
    assert [next(batches) for _ in range(3)] == [1, 2, 3]
