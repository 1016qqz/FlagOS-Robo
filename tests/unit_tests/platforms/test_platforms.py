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
"""W1 D2 - platform abstraction layer.

Covers three things W1 is accountable for:
  1. the migrated layer imports and detects the real backend,
  2. the ``current_device()`` API gap closed in W1 is actually implemented
     by *every* backend (contract test driven off PlatformBase, so a future
     backend cannot silently skip it),
  3. the ``FLAGOS_ROBO_PLATFORM`` override and its deprecated ``FS_PLATFORM``
     alias resolve in the documented priority.
"""

import inspect
import os
import warnings

import pytest
import torch

from flagos_robo.platforms import (
    LEGACY_PLATFORM_ENV_VAR,
    PLATFORMS,
    PLATFORM_ENV_VAR,
    PlatformBase,
    get_platform,
    reset_platform,
    set_platform,
)
from flagos_robo.platforms import platform_manager
from flagos_robo.platforms.platform_cuda import PlatformCUDA
from flagos_robo.platforms.platform_musa import PlatformMUSA
from flagos_robo.platforms.platform_npu import PlatformNPU
from flagos_robo.platforms.platform_txda import PlatformTXDA

ALL_BACKENDS = (PlatformCUDA, PlatformNPU, PlatformMUSA, PlatformTXDA)


@pytest.fixture(autouse=True)
def _clean_platform_state():
    """Each test starts with no cached platform and no override env vars."""
    saved = {k: os.environ.get(k) for k in (PLATFORM_ENV_VAR, LEGACY_PLATFORM_ENV_VAR)}
    for k in saved:
        os.environ.pop(k, None)
    reset_platform()
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    reset_platform()


def _abstract_methods() -> set[str]:
    return {
        name
        for name, fn in inspect.getmembers(PlatformBase, predicate=inspect.isfunction)
        if getattr(fn, "__isabstractmethod__", False)
    }


class TestPlatformContract:
    def test_base_declares_current_device_abstract(self):
        """The W1 API gap: current_device() must be part of the contract."""
        assert "current_device" in _abstract_methods()

    @pytest.mark.parametrize("backend", ALL_BACKENDS, ids=lambda c: c.__name__)
    def test_backend_implements_full_contract(self, backend):
        """Contract test, not a hardcoded list: every abstract method must be
        overridden by every backend, so adding a method to the base without
        implementing it anywhere fails here."""
        missing = _abstract_methods() - {
            n for n, _ in inspect.getmembers(backend, predicate=inspect.isfunction)
            if n in backend.__dict__
        }
        assert missing == set(), f"{backend.__name__} missing {sorted(missing)}"

    @pytest.mark.parametrize("backend", ALL_BACKENDS, ids=lambda c: c.__name__)
    def test_consumer_api_surface_is_complete(self, backend):
        """Pins the exact API the migrated consumers call.

        Harvested from the platform call sites in FlagScale @ f8e9a62c. All
        are callables, not properties: callers write ``get_platform().name()``.

        The original W1 list had 8 names, taken from a grep for the chained
        form ``get_platform().<x>``. That grep is blind to a caller that
        binds the platform to a variable first, which is how
        ``train/utils/random_utils.py`` reaches it - so ``get_rng_state`` and
        ``set_rng_state`` were missing until the W2 ``train/utils/`` batch.
        Both forms are now covered; see ``test_platform_rng.py`` for the
        signature and round-trip checks on the two added methods.
        """
        required = {
            "amp_device_type",
            "current_device",
            "device",
            "dist_backend",
            "get_rng_state",
            "manual_seed_all",
            "name",
            "set_device",
            "set_rng_state",
            "supports_distributions_on_device",
        }
        inst = backend()
        for attr in sorted(required):
            assert hasattr(inst, attr), f"{backend.__name__} lacks {attr}()"
            assert callable(getattr(inst, attr)), f"{backend.__name__}.{attr} must be callable"

    @pytest.mark.parametrize("backend", ALL_BACKENDS, ids=lambda c: c.__name__)
    def test_backend_is_instantiable_subclass(self, backend):
        assert issubclass(backend, PlatformBase)
        assert not inspect.isabstract(backend)


class TestEnvOverride:
    """Override-resolution tests, decoupled from what hardware is present.

    Five of these cases name ``"cuda"`` literally, and registration in
    ``PLATFORMS`` is gated on ``is_available()`` (``platform_manager.py``
    registers a backend only when its runtime reports a device). On a machine
    with no accelerator the registry is therefore empty, and every one of them
    failed for a reason that has nothing to do with the override logic they
    exist to pin: ``get_platform()`` either raised ValueError ("cuda is not
    available") or RuntimeError ("No available platform").

    The fixture below injects a ``"cuda"`` entry only when the real registry
    does not already have one, so the resolution *rules* -- priority, the
    deprecation warning, the conflict warning, trimming and case folding,
    empty-value fall-through -- are what gets tested, on any host. On a real
    CUDA box nothing is injected and the behaviour is unchanged.
    """

    @pytest.fixture(autouse=True)
    def _registry_has_cuda(self, _clean_platform_state, monkeypatch):
        if "cuda" not in platform_manager.PLATFORMS:
            monkeypatch.setitem(platform_manager.PLATFORMS, "cuda", PlatformCUDA())
            reset_platform()

    def test_new_env_var_selects_platform(self):
        os.environ[PLATFORM_ENV_VAR] = "cuda"
        assert get_platform().name() == "cuda"

    def test_legacy_env_var_still_works_but_warns(self):
        os.environ[LEGACY_PLATFORM_ENV_VAR] = "cuda"
        with pytest.warns(DeprecationWarning, match=LEGACY_PLATFORM_ENV_VAR):
            assert get_platform().name() == "cuda"

    def test_new_env_var_wins_over_legacy_and_warns_on_conflict(self):
        os.environ[PLATFORM_ENV_VAR] = "cuda"
        os.environ[LEGACY_PLATFORM_ENV_VAR] = "npu"
        with pytest.warns(RuntimeWarning, match="Both"):
            assert get_platform().name() == "cuda"

    def test_unknown_platform_raises_naming_the_var_used(self):
        os.environ[PLATFORM_ENV_VAR] = "nosuchdev"
        with pytest.raises(ValueError, match=PLATFORM_ENV_VAR):
            get_platform()

    def test_case_and_whitespace_insensitive(self):
        os.environ[PLATFORM_ENV_VAR] = "  CUDA  "
        assert get_platform().name() == "cuda"

    def test_empty_value_falls_through_to_autodetect(self):
        os.environ[PLATFORM_ENV_VAR] = ""
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no spurious deprecation on empty
            assert get_platform().name() in PLATFORMS

    def test_reset_platform_allows_redetection(self):
        os.environ[PLATFORM_ENV_VAR] = "cuda"
        first = get_platform()
        assert get_platform() is first  # memoised
        reset_platform()
        assert get_platform() is not None

    def test_set_platform_overrides_detection(self):
        sentinel = PlatformCUDA()
        set_platform(sentinel)
        assert get_platform() is sentinel

    def test_no_available_platform_raises(self, monkeypatch):
        """Autodetect with an empty registry must raise, not return None."""
        monkeypatch.setattr(platform_manager, "PLATFORMS", {})
        reset_platform()
        with pytest.raises(RuntimeError, match="No available platform"):
            get_platform()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
class TestRealCudaDevice:
    """Runs on the actual GPU node - this is the far-end check that W1's
    platform layer works on hardware, not just that it imports."""

    def test_autodetect_picks_cuda(self):
        p = get_platform()
        assert p.name() == "cuda"
        assert p.is_available() is True
        assert p.dist_backend() == "nccl"
        assert p.amp_device_type() == "cuda"
        assert p.device_count() == torch.cuda.device_count()

    def test_current_device_matches_torch(self):
        p = get_platform()
        assert p.current_device() == torch.cuda.current_device()

    def test_set_device_then_current_device_round_trip(self):
        """The real reason current_device() exists: FSDP(device_id=...) must
        see the device this rank actually selected."""
        p = get_platform()
        if p.device_count() < 2:
            pytest.skip("needs >= 2 devices")
        try:
            p.set_device(1)
            assert p.current_device() == 1
            assert p.device(1).index == 1
        finally:
            p.set_device(0)

    def test_tensor_lands_on_reported_device(self):
        p = get_platform()
        t = torch.zeros(4, device=p.device(p.current_device()))
        assert t.device.type == "cuda"
        assert t.device.index == p.current_device()

    def test_manual_seed_all_is_reproducible(self):
        p = get_platform()
        p.manual_seed_all(1234)
        a = torch.randn(8, device="cuda")
        p.manual_seed_all(1234)
        b = torch.randn(8, device="cuda")
        assert torch.equal(a, b)
