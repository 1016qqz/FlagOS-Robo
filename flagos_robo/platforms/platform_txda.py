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
#
# Migrated from FlagScale flagscale/platforms/platform_txda.py @ baseline f8e9a62c.

import torch

from .platform_base import PlatformBase


class PlatformTXDA(PlatformBase):
    def __init__(self):
        self._name = "txda"
        try:
            import flag_gems  # noqa: F401
            from torch_txda import transfer_to_txda  # noqa: F401
        except ImportError:
            pass

    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        try:
            import torch_txda  # noqa: F401

            return torch.txda.is_available() and torch.txda.device_count() > 0
        except Exception:
            return False

    def set_device(self, device_index):
        torch.cuda.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("txda", device_index)

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def current_device(self) -> int:
        # txda rides on the cuda runtime, same as set_device/device_count above.
        return torch.cuda.current_device()

    def dist_backend(self) -> str:
        return "flagcx"

    def manual_seed_all(self, seed):
        torch.cuda.manual_seed_all(seed)

    def get_rng_state(self, device=None):
        if device is None:
            return torch.cuda.get_rng_state()

        return torch.cuda.get_rng_state(device)

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch.cuda.set_rng_state(new_state)

        return torch.cuda.set_rng_state(new_state, device_index)

    def amp_device_type(self) -> str:
        return "cuda"
