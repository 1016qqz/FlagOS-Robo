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
# Migrated from FlagScale flagscale/platforms/platform_base.py @ baseline f8e9a62c.

import abc
from abc import ABC


class PlatformBase(ABC):
    """FlagOS-Robo hardware platform abstraction base class.

    Provides a unified interface for device operations across different
    hardware backends (CUDA, NPU, MUSA, etc.).
    """

    @abc.abstractmethod
    def name(self) -> str:
        """Return the platform name, e.g. 'cuda', 'npu', 'musa'."""
        ...

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if this platform is available on the current machine."""
        ...

    # ---- Device APIs ----

    @abc.abstractmethod
    def set_device(self, device_index):
        """Set the current device."""
        ...

    @abc.abstractmethod
    def device(self, device_index=None):
        """Return a torch.device for the given index."""
        ...

    @abc.abstractmethod
    def device_count(self) -> int:
        """Return the number of available devices."""
        ...

    @abc.abstractmethod
    def current_device(self) -> int:
        """Return the index of the device currently selected on this process.

        Added in FlagOS-Robo W1 to close an API drift: ``train_pi.py`` calls
        ``get_platform().current_device()`` to build ``FSDP(device_id=...)``,
        but FlagScale's ``flagscale/platforms/`` never defined it (that entry
        point was bound to ``megatron.plugin.platform`` instead). Without this
        method the PI0 training entry point raises ``AttributeError``.
        """
        ...

    # ---- Distributed ----

    @abc.abstractmethod
    def dist_backend(self) -> str:
        """Return the distributed backend name, e.g. 'nccl', 'hccl', 'mccl'."""
        ...

    # ---- RNG ----

    @abc.abstractmethod
    def manual_seed_all(self, seed):
        """Set the seed for all devices."""
        ...

    # ---- AMP ----

    @abc.abstractmethod
    def amp_device_type(self) -> str:
        """Return the device type string for torch.amp.autocast, e.g. 'cuda', 'npu'."""
        ...

    # ---- Compatibility ----

    def supports_distributions_on_device(self) -> bool:
        """Whether torch.distributions (e.g. Beta) can run directly on this device.

        Returns False for platforms that require CPU fallback (e.g. MUSA).
        Defaults to True.
        """
        return True
