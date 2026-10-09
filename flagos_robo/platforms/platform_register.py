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
# Migrated from FlagScale flagscale/platforms/platform_register.py @ baseline f8e9a62c.

PLATFORMS = {}


def register_platforms() -> None:
    """Register all available platforms."""

    from .platform_cuda import PlatformCUDA

    platform_cuda = PlatformCUDA()
    if platform_cuda.is_available():
        PLATFORMS["cuda"] = platform_cuda

    from .platform_npu import PlatformNPU

    platform_npu = PlatformNPU()
    if platform_npu.is_available():
        PLATFORMS["npu"] = platform_npu

    from .platform_musa import PlatformMUSA

    platform_musa = PlatformMUSA()
    if platform_musa.is_available():
        PLATFORMS["musa"] = platform_musa

    from .platform_txda import PlatformTXDA

    platform_txda = PlatformTXDA()
    if platform_txda.is_available():
        PLATFORMS["txda"] = platform_txda
