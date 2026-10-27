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

import numpy as np
import pytest
import torch

from flagos_robo.serve.processor.image_layout_processor import ImageLayoutProcessorStep
from flagos_robo.serve.processor.numpy_to_tensor_processor import NumpyToTensorProcessorStep
from flagos_robo.train.processor import ProcessorStepRegistry


@pytest.fixture
def step():
    return NumpyToTensorProcessorStep()


def _make_obs():
    return {
        "observation.images.image": np.zeros((1, 3, 4, 4), dtype=np.float32),
        "observation.state": np.arange(8, dtype=np.float32),
        "task": "put the bowl on the plate",
    }


def test_converts_images_and_state(step):
    out = step.observation(_make_obs())
    assert isinstance(out["observation.images.image"], torch.Tensor)
    assert out["observation.images.image"].dtype == torch.float32
    assert isinstance(out["observation.state"], torch.Tensor)
    assert out["observation.state"].dtype == torch.float32


def test_non_array_values_pass_through(step):
    out = step.observation(_make_obs())
    assert out["task"] == "put the bowl on the plate"


def test_state_stays_1d_so_to_batch_can_unsqueeze(step):
    """AddBatchDimensionObservationStep only unsqueezes a 1-D torch tensor."""
    out = step.observation(_make_obs())
    state = out["observation.state"]
    assert isinstance(state, torch.Tensor) and state.dim() == 1


def test_existing_tensor_is_cast_not_rejected():
    step = NumpyToTensorProcessorStep(dtype="float16")
    obs = {"observation.state": torch.ones(8, dtype=torch.float32)}
    out = step.observation(obs)
    assert out["observation.state"].dtype == torch.float16


def test_list_value_is_converted(step):
    obs = {"observation.state": [1.0, 2.0, 3.0]}
    out = step.observation(obs)
    assert isinstance(out["observation.state"], torch.Tensor)
    assert out["observation.state"].shape == (3,)


def test_explicit_keys_restrict_conversion():
    step = NumpyToTensorProcessorStep(keys=["observation.state"])
    out = step.observation(_make_obs())
    assert isinstance(out["observation.state"], torch.Tensor)
    assert isinstance(out["observation.images.image"], np.ndarray)


def test_invalid_dtype_raises():
    with pytest.raises(ValueError):
        NumpyToTensorProcessorStep(dtype="int8")


def test_registered_under_registry_name():
    assert ProcessorStepRegistry.get("numpy_to_tensor_processor") is NumpyToTensorProcessorStep


def test_layout_step_must_run_first_negative_control():
    """Converting before image_layout makes the layout step skip every image.

    This pins the load-bearing ordering constraint: ImageLayoutProcessorStep gates on
    ``isinstance(img, np.ndarray)``, so the wrong order leaves a 3-D CHW-less tensor.
    """
    layout = ImageLayoutProcessorStep(
        src_layout="hwc", dst_layout="chw", add_batch_dim=True, to_float=True
    )
    tensor_step = NumpyToTensorProcessorStep()
    img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)

    # Wrong order: tensor first -> layout is a no-op, image stays HWC and 3-D.
    wrong = layout.observation(tensor_step.observation({"observation.images.image": img.copy()}))
    assert wrong["observation.images.image"].shape == (4, 4, 3)

    # Right order: layout first -> NCHW float, then converted to a tensor.
    right = tensor_step.observation(layout.observation({"observation.images.image": img.copy()}))
    out = right["observation.images.image"]
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 3, 4, 4)
    assert out.dtype == torch.float32
    assert float(out.max()) <= 1.0


def test_default_serve_preprocessor_covers_pi0_and_pi05():
    """Before the fix, pi0 got no serve preprocessor at all (returned None)."""
    from flagos_robo.serve.run_serve_vla import _default_serve_preprocessor

    for variant in ("pi0", "pi0.5"):
        pipeline = _default_serve_preprocessor(variant)
        assert pipeline is not None, f"{variant} has no default serve preprocessor"
        names = [type(s).__name__ for s in pipeline.steps]
        assert names == ["ImageLayoutProcessorStep", "NumpyToTensorProcessorStep"], names

    # gr00t branch deliberately untouched.
    gr00t = _default_serve_preprocessor("qwengr00t")
    assert [type(s).__name__ for s in gr00t.steps] == ["ImageResizeProcessorStep"]
    assert _default_serve_preprocessor("") is None
