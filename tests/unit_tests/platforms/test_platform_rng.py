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

"""W2 B2 - platform RNG state API, added with the ``train/utils/`` batch.

Why this file exists: the W1 API survey harvested call sites with a grep for
the chained form ``get_platform().<method>``, which structurally cannot see a
caller that binds the platform to a variable first. ``random_utils.py`` does
exactly that::

    cur_platform = _cur_platform()
    ...
    cur_platform.get_rng_state()      # line 125
    cur_platform.set_rng_state(...)   # line 136

(W6 made that binding lazy -- it was a module-level
``cur_platform = get_platform()`` resolved at import, which no accelerator-less
host could import. The variable-bound *shape* this file guards is unchanged.)

So ``get_rng_state`` / ``set_rng_state`` were absent from ``PlatformBase``
until W2 pulled that file in. Signatures follow Megatron-LM-FL
``megatron/plugin/platform/platform_base.py:102,106``.

``test_platforms.py::test_backend_implements_full_contract`` already covers
"every backend overrides every abstract method" generically. This file pins
the parts that generic test cannot express: that the two methods are abstract
at all, that their signatures accept the no-argument call the consumer makes,
and that a real round-trip reproduces a draw.
"""

import inspect

import pytest
import torch

from flagos_robo.platforms import PlatformBase, get_platform
from flagos_robo.platforms.platform_cuda import PlatformCUDA
from flagos_robo.platforms.platform_musa import PlatformMUSA
from flagos_robo.platforms.platform_npu import PlatformNPU
from flagos_robo.platforms.platform_txda import PlatformTXDA

ALL_BACKENDS = (PlatformCUDA, PlatformNPU, PlatformMUSA, PlatformTXDA)
RNG_METHODS = ("get_rng_state", "set_rng_state")


class TestRngContract:
    @pytest.mark.parametrize("name", RNG_METHODS)
    def test_declared_abstract_on_base(self, name):
        """Abstract, not a concrete no-op.

        A no-op default would let a new backend forget the method and fail
        only at checkpoint time; abstract makes the omission fail at
        instantiation.
        """
        assert name in PlatformBase.__abstractmethods__

    @pytest.mark.parametrize("backend", ALL_BACKENDS, ids=lambda c: c.__name__)
    @pytest.mark.parametrize("name", RNG_METHODS)
    def test_backend_overrides_in_own_dict(self, backend, name):
        """The override must be on the backend itself, not inherited."""
        assert name in backend.__dict__, f"{backend.__name__} does not define {name}"

    @pytest.mark.parametrize("backend", ALL_BACKENDS, ids=lambda c: c.__name__)
    def test_get_rng_state_signature(self, backend):
        """``random_utils.py:94`` calls this with no argument."""
        sig = inspect.signature(backend.get_rng_state)
        assert list(sig.parameters) == ["self", "device"]
        assert sig.parameters["device"].default is None

    @pytest.mark.parametrize("backend", ALL_BACKENDS, ids=lambda c: c.__name__)
    def test_set_rng_state_signature(self, backend):
        """``random_utils.py:104`` passes only the state tensor."""
        sig = inspect.signature(backend.set_rng_state)
        assert list(sig.parameters) == ["self", "new_state", "device_index"]
        assert sig.parameters["new_state"].default is inspect.Parameter.empty
        assert sig.parameters["device_index"].default is None

    def test_incomplete_subclass_cannot_instantiate(self):
        """Negative control: makes the abstract declaration load-bearing."""

        class Incomplete(PlatformBase):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()


class TestVariableBoundCallSite:
    """Guards the specific shape of caller that the W1 grep missed."""

    @pytest.fixture
    def registry_has_cuda(self, monkeypatch):
        """Make a platform resolvable regardless of what hardware is present.

        ``PLATFORMS`` is populated only for backends whose ``is_available()``
        returns True, so on a host with no accelerator it is empty and
        ``get_platform()`` raises. Injecting an entry keeps the assertion about
        *call shape* testable there. Requested explicitly, not autouse: the
        import test above must see the registry untouched.
        """
        from flagos_robo.platforms import platform_manager, reset_platform

        if "cuda" not in platform_manager.PLATFORMS:
            monkeypatch.setitem(platform_manager.PLATFORMS, "cuda", PlatformCUDA())
        reset_platform()
        yield
        reset_platform()

    def test_importing_the_consumer_needs_no_accelerator(self):
        """Importing ``random_utils`` must not resolve a platform.

        It used to, via a module-level ``cur_platform = get_platform()``, which
        made the module unimportable wherever the registry is empty (no
        accelerator) -- ``get_platform()`` raises RuntimeError there. That is a
        plain import-time side effect, not a requirement of the module: both
        consumers guard their use with ``is_available()``. This test is the
        negative control for that fix, so a future edit cannot quietly put the
        eager lookup back.
        """
        pytest.importorskip("datasets", reason="random_utils imports datasets.utils")
        import importlib

        from flagos_robo.train.utils import random_utils

        importlib.reload(random_utils)  # a fresh import, registry state untouched

    def test_consumer_calls_via_bound_variable(self, registry_has_cuda):
        """``random_utils`` must still reach the platform through a variable.

        If someone rewrites it to the chained form this test is free to be
        deleted - but then the W1 blind spot no longer applies either. Keeping
        it asserts the module really exercises the variable-bound path.

        The lookup is now lazy (module ``__getattr__``, PEP 562), so reading
        the attribute is what resolves the platform -- hence the registry entry
        has to exist at read time, which is what the injection below provides on
        a host with no real device.
        """
        pytest.importorskip("datasets", reason="random_utils imports datasets.utils")
        from flagos_robo.train.utils import random_utils

        assert isinstance(random_utils.cur_platform, PlatformBase)
        for name in RNG_METHODS:
            assert callable(getattr(random_utils.cur_platform, name))

    def test_unknown_attribute_still_raises(self):
        """The ``__getattr__`` hook must not swallow genuine typos."""
        pytest.importorskip("datasets", reason="random_utils imports datasets.utils")
        from flagos_robo.train.utils import random_utils

        with pytest.raises(AttributeError, match="no_such_attribute"):
            random_utils.no_such_attribute


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real accelerator")
class TestRealDeviceRoundTrip:
    def test_get_returns_uint8_tensor(self):
        state = get_platform().get_rng_state()
        assert isinstance(state, torch.Tensor)
        assert state.dtype == torch.uint8

    def test_set_restores_the_draw(self):
        """Restoring state must reproduce the next random draw bit-for-bit."""
        plat = get_platform()
        state = plat.get_rng_state()
        before = torch.rand(8, device=plat.device())
        plat.set_rng_state(state)
        after = torch.rand(8, device=plat.device())
        assert torch.equal(before, after)

    def test_explicit_device_argument_is_accepted(self):
        """The non-None branch of both methods, which the consumer never hits
        but a future caller may."""
        plat = get_platform()
        idx = plat.current_device()
        state = plat.get_rng_state(plat.device(idx))
        assert isinstance(state, torch.Tensor)
        plat.set_rng_state(state, idx)

    def test_serialize_deserialize_rng_state_round_trip(self):
        """End-to-end through the actual consumer in ``random_utils``."""
        pytest.importorskip("datasets", reason="random_utils imports datasets.utils")
        from flagos_robo.train.utils.random_utils import (
            deserialize_torch_rng_state,
            serialize_torch_rng_state,
        )

        plat = get_platform()
        snapshot = serialize_torch_rng_state()
        assert "torch_rng_state" in snapshot
        assert "torch_cuda_rng_state" in snapshot, "accelerator state must be captured"

        before = torch.rand(8, device=plat.device())
        deserialize_torch_rng_state(snapshot)
        after = torch.rand(8, device=plat.device())
        assert torch.equal(before, after)
