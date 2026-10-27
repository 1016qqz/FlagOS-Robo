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

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from flagos_robo.logger import logger
from flagos_robo.models.configs.types import PipelineFeatureType, PolicyFeature
from flagos_robo.models.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from flagos_robo.train.processor import ObservationProcessorStep, ProcessorStepRegistry
from flagos_robo.train.processor.converters import to_tensor

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
}


@dataclass
@ProcessorStepRegistry.register(name="numpy_to_tensor_processor")
class NumpyToTensorProcessorStep(ObservationProcessorStep):
    """Converts numpy observation entries to ``torch.Tensor``.

    Serve clients send observations as numpy arrays (see ``run_serve_vla.py``
    docstring: HWC ``uint8`` images). The policy preprocessor steps and the model
    forward both gate on ``isinstance(x, torch.Tensor)``, so the numpy values must
    be converted before they reach the policy pipeline.

    Operates on ``observation.images.*``, ``observation.state`` and
    ``observation.environment_state`` by default. Values that are already tensors
    are only cast to ``dtype``; non array-like values are left untouched.

    Note:
        This step MUST be placed AFTER ``image_layout_processor`` in the serve
        preprocessor chain, because that step gates on ``isinstance(img, np.ndarray)``
        and would skip every image if the conversion happened first.

    Args:
        dtype: Target torch dtype name, one of ``float32``/``float16``/``bfloat16``/``float64``.
        keys: Optional explicit key list. When given, only these keys are converted.

    Example serve config::

        serve_preprocessor:
          steps:
            - registry_name: image_layout_processor
              config:
                src_layout: hwc
                dst_layout: chw
                add_batch_dim: true
                to_float: true
            - registry_name: numpy_to_tensor_processor
              config:
                dtype: float32
    """

    dtype: str = "float32"
    keys: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPES:
            raise ValueError(
                f"numpy_to_tensor_processor: unsupported dtype '{self.dtype}', "
                f"expected one of {sorted(_DTYPES)}"
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

    def _targets(self, observation) -> list[str]:
        if self.keys:
            return [key for key in observation.keys() if key in self.keys]
        return [
            key
            for key in observation.keys()
            if key.startswith(OBS_IMAGES) or key in (OBS_STATE, OBS_ENV_STATE)
        ]

    def observation(self, observation):
        dtype = self.torch_dtype
        for key in self._targets(observation):
            value = observation[key]
            if isinstance(value, torch.Tensor):
                observation[key] = value.to(dtype=dtype)
                continue
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            if isinstance(value, np.ndarray):
                observation[key] = to_tensor(value, dtype=dtype)
                continue
            logger.warning(
                f"numpy_to_tensor_processor: skipping '{key}': expected np.ndarray or "
                f"torch.Tensor, got {type(value).__name__}"
            )
        return observation

    def get_config(self) -> dict[str, Any]:
        return {"dtype": self.dtype, "keys": self.keys}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
