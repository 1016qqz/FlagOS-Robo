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
# Migrated from FlagScale flagscale/platforms/platform_manager.py @ baseline f8e9a62c.

import os
import warnings

from .platform_register import PLATFORMS

_current_platform = None


#: Primary override env var for this repo. ``FS_PLATFORM`` is retained as a
#: deprecated alias so configs/scripts carried over from FlagScale keep working.
PLATFORM_ENV_VAR = "FLAGOS_ROBO_PLATFORM"
LEGACY_PLATFORM_ENV_VAR = "FS_PLATFORM"


def _read_platform_env() -> tuple[str, str]:
    """Return ``(var_name, value)`` for the platform override, or ``("", "")``.

    Priority: ``FLAGOS_ROBO_PLATFORM`` > ``FS_PLATFORM`` (deprecated). When only
    the legacy name is set, a DeprecationWarning is emitted; when both are set,
    the new name wins and the conflict is warned about rather than silently
    resolved.
    """
    new = os.environ.get(PLATFORM_ENV_VAR, "").strip().lower()
    legacy = os.environ.get(LEGACY_PLATFORM_ENV_VAR, "").strip().lower()
    if new and legacy and new != legacy:
        warnings.warn(
            f"Both {PLATFORM_ENV_VAR}={new!r} and {LEGACY_PLATFORM_ENV_VAR}={legacy!r} "
            f"are set; using {PLATFORM_ENV_VAR}.",
            RuntimeWarning,
            stacklevel=3,
        )
    if new:
        return PLATFORM_ENV_VAR, new
    if legacy:
        warnings.warn(
            f"{LEGACY_PLATFORM_ENV_VAR} is deprecated, use {PLATFORM_ENV_VAR} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return LEGACY_PLATFORM_ENV_VAR, legacy
    return "", ""


def get_platform():
    """Get the current platform instance.

    Detection priority:
    1. Environment variable ``FLAGOS_ROBO_PLATFORM`` (e.g. "cuda", "npu", "musa")
    2. Deprecated alias ``FS_PLATFORM``
    3. Auto-detect: cuda > npu > musa > txda
    """
    global _current_platform
    if _current_platform is not None:
        return _current_platform

    # 1. Check environment variable override
    env_var, env_platform = _read_platform_env()
    if env_platform:
        if env_platform not in PLATFORMS:
            raise ValueError(
                f"{env_var}='{env_platform}' is not available. "
                f"Registered platforms: {list(PLATFORMS.keys())}"
            )
        _current_platform = PLATFORMS[env_platform]
        return _current_platform

    # 2. Auto-detect in priority order
    for name in ("cuda", "npu", "musa", "txda"):
        if name in PLATFORMS:
            _current_platform = PLATFORMS[name]
            return _current_platform

    raise RuntimeError(
        "No available platform detected. "
        "Ensure CUDA, NPU (torch_npu), or MUSA (torch_musa) is installed."
    )


def set_platform(platform_obj):
    """Manually set the current platform."""
    global _current_platform
    _current_platform = platform_obj


def reset_platform() -> None:
    """Clear the cached platform so the next ``get_platform()`` re-detects.

    Added in FlagOS-Robo W1: ``get_platform()`` memoises into a module global,
    which makes the env-var priority path unobservable in a single process
    (the first call wins forever). Tests and multi-backend harnesses need to
    re-run detection, so the cache gets an explicit escape hatch.
    """
    global _current_platform
    _current_platform = None
