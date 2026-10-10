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

# W2-W3: migrated from flagscale/models/vla/ (21 of 27 files); qwen_gr00t/ and
# vlm/ stay in FlagScale.

from .base_policy import TrainablePolicy
from .gr00t_n1_5 import Gr00tN15
from .protocols import ActionModel, VLMBackbone
from .registry import (
    build_action_model,
    build_vlm,
    register_action_model,
    register_vlm,
)
from .utils import get_vlm_config

# qwen_gr00t/ and vlm/ are deliberately NOT part of FlagOS-Robo: they need a
# newer transformers (Qwen3VLForConditionalGeneration) than the PI0/PI0.5 env
# provides, so they stay in FlagScale. The try/except that used to import
# QwenGr00t / Qwen{,25,3}VLBackbone here is gone along with them, and so are
# their four entries in __all__ -- a name left in __all__ with nothing bound
# to it makes `from ... import *` raise AttributeError.

try:
    from flagos_robo.models.pi0.modeling_pi0 import PI0Policy
    from flagos_robo.models.pi05.modeling_pi05 import PI05Policy
except ImportError:
    pass

__all__ = [
    "TrainablePolicy",
    "VLMBackbone",
    "ActionModel",
    "register_vlm",
    "register_action_model",
    "build_vlm",
    "build_action_model",
    "get_vlm_config",
    "Gr00tN15",
    "PI0Policy",
    "PI05Policy",
]
