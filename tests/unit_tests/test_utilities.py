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

"""Shared device-probing helpers for the FlagOS-Robo unit tests.

Provenance and deliberate deviations from the upstream FlagScale helper
(FlagScale/tests/unit_tests/test_utilities.py, 239 lines):

1. Trimmed to the accelerator-probing surface only. The sole consumer in this
   repo is tests/unit_tests/train/utils/test_train_utils.py, which uses exactly
   three entry points: dist_backend(), has_accelerator() and
   accelerator_device(). Everything else upstream (TestModel, process-group
   setup/teardown, distributed_barrier, ...) exists to drive
   megatron.core.parallel_state and has no consumer here.

2. No megatron import. Upstream line 20 is
   `import megatron.core.parallel_state as ps`, which this repo forbids --
   tools/check_no_megatron.py scans every .py file in the tree. The three
   helpers kept below never touched parallel_state; it was only used by the
   process-group methods that are not carried over.

3. Lazy rank/world-size reads. Upstream evaluates
   `rank = int(os.environ["LOCAL_RANK"])` in the class body, so merely
   importing the module raises KeyError outside a torchrun launch. Plain
   `pytest` is exactly that case, so rank() reads the environment on call and
   falls back to 0. This keeps collection safe for the single-process run.

4. Environment variable prefix is FLAGOS_ROBO_TEST_* rather than
   FLAGSCALE_TEST_*. The upstream names are not referenced anywhere in this
   repo, so renaming them costs nothing and keeps the tree free of the old
   project name.
"""

import os

import torch

# Maps a logical platform name to its torch runtime details. Kept verbatim from
# upstream so that a future cross-vendor test run behaves identically.
PLATFORM_RUNTIME_MAP = {
    "cuda": {
        "dist_backend": "nccl",
        "torch_device_type": "cuda",
        "torch_accelerator_attr": "cuda",
        "optional_runtime_module": None,
    },
    "ascend": {
        "dist_backend": "hccl",
        "torch_device_type": "npu",
        "torch_accelerator_attr": "npu",
        "optional_runtime_module": "torch_npu",
    },
    "metax": {
        "dist_backend": "nccl",
        "torch_device_type": "cuda",
        "torch_accelerator_attr": "cuda",
        "optional_runtime_module": None,
    },
    "musa": {
        "dist_backend": "mccl",
        "torch_device_type": "musa",
        "torch_accelerator_attr": "musa",
        "optional_runtime_module": "torch_musa",
    },
}

DEFAULT_PLATFORM_RUNTIME = PLATFORM_RUNTIME_MAP["cuda"]


class Utils:
    """Accelerator probing shared by the unit tests.

    All methods are static and side-effect free apart from the optional vendor
    runtime import in accelerator(), which is guarded.
    """

    @staticmethod
    def rank():
        """Local rank, defaulting to 0 when not launched under torchrun."""
        return int(os.getenv("LOCAL_RANK", "0"))

    @staticmethod
    def world_size():
        """World size, defaulting to 1 when not launched under torchrun."""
        return int(os.getenv("WORLD_SIZE", "1"))

    @staticmethod
    def platform():
        return os.getenv("FLAGOS_ROBO_TEST_PLATFORM", "cuda")

    @staticmethod
    def platform_runtime():
        return PLATFORM_RUNTIME_MAP.get(Utils.platform(), DEFAULT_PLATFORM_RUNTIME)

    @staticmethod
    def dist_backend():
        return os.getenv(
            "FLAGOS_ROBO_TEST_DIST_BACKEND", Utils.platform_runtime()["dist_backend"]
        )

    @staticmethod
    def torch_device_type():
        return os.getenv(
            "FLAGOS_ROBO_TEST_TORCH_DEVICE_TYPE",
            Utils.platform_runtime()["torch_device_type"],
        )

    @staticmethod
    def accelerator():
        """Return the torch accelerator module for the configured platform.

        Falls back to torch.cuda when the vendor attribute is absent, matching
        upstream behaviour.
        """
        runtime = Utils.platform_runtime()
        optional_runtime_module = runtime.get("optional_runtime_module")
        if optional_runtime_module:
            try:
                __import__(optional_runtime_module)
            except Exception:
                # Vendor runtime not installed on this host; the getattr below
                # then falls back to torch.cuda.
                pass

        accelerator_attr = os.getenv(
            "FLAGOS_ROBO_TEST_TORCH_ACCELERATOR_ATTR", runtime["torch_accelerator_attr"]
        )
        return getattr(torch, accelerator_attr, torch.cuda)

    @staticmethod
    def accelerator_device_count():
        accelerator = Utils.accelerator()
        device_count = getattr(accelerator, "device_count", None)
        if callable(device_count):
            count = device_count()
            return count if count > 0 else 1
        return 1

    @staticmethod
    def has_accelerator():
        accelerator = Utils.accelerator()
        device_count = getattr(accelerator, "device_count", None)
        return callable(device_count) and device_count() > 0

    @staticmethod
    def accelerator_device(rank=None):
        rank = Utils.rank() if rank is None else rank
        index = rank % Utils.accelerator_device_count()
        return torch.device(f"{Utils.torch_device_type()}:{index}")
