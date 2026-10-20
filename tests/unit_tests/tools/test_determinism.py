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
"""Tests for the shared determinism switches (W4 D1-7)."""

from __future__ import annotations

import os
import random

import pytest

from flagos_robo.utils.determinism import (
    CUBLAS_ENV_VAR,
    CUBLAS_REQUIRED_VALUE,
    DETERMINISM_ENV_VAR,
    DETERMINISM_STRICT_VALUE,
    FLASH_ATTN_ENV_VAR,
    FLASH_ATTN_REQUIRED_VALUE,
    determinism_enabled,
    enable_determinism,
    make_dataloader_generator,
    seed_worker,
    strict_determinism_enabled,
)
from flagos_robo.utils.precision_dump import DUMP_ENV_VAR, dump_dir_from_env

# (raw env value, expected enabled). None means the variable is absent.
GATE_CASES = [
    (None, False),
    ("", False),
    ("   ", False),
    ("0", False),
    ("false", False),
    ("False", False),
    ("no", False),
    ("OFF", False),
    ("1", True),
    ("true", True),
    ("yes", True),
    ("  1  ", True),
    ("strict", True),
]

# (raw env value, expected strict). Strict must be a *sub-state* of enabled.
STRICT_CASES = [
    (None, False),
    ("", False),
    ("0", False),
    ("off", False),
    ("1", False),
    ("true", False),
    ("yes", False),
    ("strict", True),
    ("STRICT", True),
    ("Strict", True),
    ("  strict  ", True),
    # Near-misses must NOT be strict: a typo has to degrade to the
    # non-aborting tier, never silently make a training run fatal.
    ("stric", False),
    ("strict1", False),
    ("strictly", False),
]


@pytest.mark.parametrize(("raw", "expected"), GATE_CASES)
def test_gate_interprets_env(raw, expected):
    env = {} if raw is None else {DETERMINISM_ENV_VAR: raw}
    assert determinism_enabled(env) is expected


@pytest.mark.parametrize(("raw", "_expected"), GATE_CASES)
def test_gate_agrees_with_dump_gate(raw, _expected):
    """The two gates must read the environment identically.

    Both training loops call these side by side; if one treated "0" as on and
    the other as off we would dump on one side only and chase a phantom
    divergence. This pins them together rather than trusting that two
    copies of the parsing logic stay in sync.
    """
    det_env = {} if raw is None else {DETERMINISM_ENV_VAR: raw}
    dump_env = {} if raw is None else {DUMP_ENV_VAR: raw}
    assert determinism_enabled(det_env) is (dump_dir_from_env(dump_env) is not None)


def test_disabled_by_default(monkeypatch):
    """A normal training run must pay nothing for this module."""
    monkeypatch.delenv(DETERMINISM_ENV_VAR, raising=False)
    assert enable_determinism(42) is False


def test_disabled_does_not_touch_cublas_env(monkeypatch):
    monkeypatch.delenv(DETERMINISM_ENV_VAR, raising=False)
    monkeypatch.delenv(CUBLAS_ENV_VAR, raising=False)
    enable_determinism(42)
    import os

    assert CUBLAS_ENV_VAR not in os.environ


def test_cublas_value_is_one_torch_accepts():
    """torch.use_deterministic_algorithms(True) raises unless this matches."""
    assert CUBLAS_REQUIRED_VALUE in {":4096:8", ":16:8"}


def test_enable_sets_cublas_workspace(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "1")
    monkeypatch.delenv(CUBLAS_ENV_VAR, raising=False)
    assert enable_determinism(42) is True
    import os

    assert os.environ[CUBLAS_ENV_VAR] == CUBLAS_REQUIRED_VALUE


def test_enable_overrides_hostile_cublas_value(monkeypatch):
    """A stale export must not be left in place to crash a matmul later."""
    pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "1")
    monkeypatch.setenv(CUBLAS_ENV_VAR, ":2:2")
    enable_determinism(42)
    import os

    assert os.environ[CUBLAS_ENV_VAR] == CUBLAS_REQUIRED_VALUE


def test_enable_keeps_other_accepted_cublas_value(monkeypatch):
    """:16:8 is also valid; don't churn an already-correct setting."""
    pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "1")
    monkeypatch.setenv(CUBLAS_ENV_VAR, ":16:8")
    enable_determinism(42)
    import os

    assert os.environ[CUBLAS_ENV_VAR] == ":16:8"


def test_seed_worker_makes_python_rng_reproducible():
    """Same worker seed => same draws; this is the whole point of the hook."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    seed_worker(0)
    first = [random.random() for _ in range(5)]
    torch.manual_seed(1234)
    seed_worker(0)
    assert [random.random() for _ in range(5)] == first


def test_seed_worker_differs_with_base_seed():
    """Reproducible must not mean constant: a new base seed must move it."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    seed_worker(0)
    first = [random.random() for _ in range(5)]
    torch.manual_seed(2)
    seed_worker(0)
    assert [random.random() for _ in range(5)] != first


def test_seed_worker_seed_fits_numpy_range():
    """NumPy rejects seeds >= 2**32; torch seeds are 64-bit."""
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    torch.manual_seed(2**40 + 7)
    seed_worker(3)  # would raise if the mask were missing
    assert np.random.randint(0, 10) >= 0


def test_generator_is_seeded_and_reproducible():
    torch = pytest.importorskip("torch")
    a = make_dataloader_generator(7)
    b = make_dataloader_generator(7)
    assert a.initial_seed() == b.initial_seed() == 7
    ta = torch.randperm(10, generator=a)
    tb = torch.randperm(10, generator=b)
    assert torch.equal(ta, tb)
    c = make_dataloader_generator(8)
    assert not torch.equal(torch.randperm(10, generator=c), ta)


@pytest.mark.parametrize(("raw", "expected"), STRICT_CASES)
def test_strict_gate_interprets_env(raw, expected):
    env = {} if raw is None else {DETERMINISM_ENV_VAR: raw}
    assert strict_determinism_enabled(env) is expected


@pytest.mark.parametrize(("raw", "_expected"), STRICT_CASES)
def test_strict_implies_enabled(raw, _expected):
    """Strict is a tier of "on", never an orthogonal flag.

    If strict could be true while the plain gate was false, the training loop
    would skip ``enable_determinism`` entirely and the strict request would be
    silently dropped -- the exact class of bug that cost W4 D2 three rounds.
    """
    env = {} if raw is None else {DETERMINISM_ENV_VAR: raw}
    if strict_determinism_enabled(env):
        assert determinism_enabled(env) is True


def test_strict_value_is_not_an_off_value():
    """``strict`` must not collide with the off-vocabulary."""
    assert DETERMINISM_STRICT_VALUE not in {"0", "false", "no", "off", ""}


def test_warn_only_defaults_to_tier_non_strict(monkeypatch):
    """``FLAGOS_DETERMINISTIC=1`` keeps warn_only=True (never aborts)."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "1")
    seen = {}

    def spy(flag, warn_only=False):
        seen["flag"] = flag
        seen["warn_only"] = warn_only

    monkeypatch.setattr(torch, "use_deterministic_algorithms", spy)
    assert enable_determinism(42) is True
    assert seen == {"flag": True, "warn_only": True}


def test_warn_only_defaults_to_tier_strict(monkeypatch):
    """``FLAGOS_DETERMINISTIC=strict`` must pass warn_only=False.

    This is the load-bearing assertion of the W4 D2 fix: warn_only=False is
    what makes torch select the *deterministic* Flash Attention backward.
    With warn_only=True the nondeterministic kernel still runs and only warns,
    which measured as run-to-run divergence from step 2 onwards.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "strict")
    seen = {}

    def spy(flag, warn_only=True):
        seen["flag"] = flag
        seen["warn_only"] = warn_only

    monkeypatch.setattr(torch, "use_deterministic_algorithms", spy)
    assert enable_determinism(42) is True
    assert seen == {"flag": True, "warn_only": False}


def test_explicit_warn_only_overrides_env_tier(monkeypatch):
    """An explicit kwarg wins over the env tier, both directions."""
    torch = pytest.importorskip("torch")
    seen = {}
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda flag, warn_only=None: seen.update(warn_only=warn_only),
    )

    monkeypatch.setenv(DETERMINISM_ENV_VAR, "strict")
    enable_determinism(42, warn_only=True)
    assert seen["warn_only"] is True

    monkeypatch.setenv(DETERMINISM_ENV_VAR, "1")
    enable_determinism(42, warn_only=False)
    assert seen["warn_only"] is False


def test_strict_still_applies_every_other_switch(monkeypatch):
    """Strict must be additive: it adds warn_only=False, removes nothing."""
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "strict")
    monkeypatch.delenv(CUBLAS_ENV_VAR, raising=False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)
    assert enable_determinism(42) is True
    import os

    assert os.environ[CUBLAS_ENV_VAR] == CUBLAS_REQUIRED_VALUE
    if torch.cuda.is_available():
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cuda.matmul.allow_tf32 is False


def test_flash_attn_determinism_env_is_set_by_enable(monkeypatch):
    """``enable_determinism`` must export FLASH_ATTENTION_DETERMINISTIC=1.

    gr00t_n1_5 forces ``_attn_implementation="flash_attention_2"``, which routes
    attention to the ``flash_attn`` package. That is not a torch operator, so
    ``torch.use_deterministic_algorithms`` cannot reach it and emits no warning;
    transformers reads this env var instead. Measured consequence of leaving it
    at its "0" default even under strict torch determinism: step-1 loss was
    bit-identical while step-1 grad_norm already differed
    (13.856263160705566 vs 13.859310150146484).
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "strict")
    monkeypatch.delenv(FLASH_ATTN_ENV_VAR, raising=False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)

    assert enable_determinism(42) is True
    assert os.environ[FLASH_ATTN_ENV_VAR] == FLASH_ATTN_REQUIRED_VALUE


def test_flash_attn_determinism_env_set_in_non_strict_tier_too(monkeypatch):
    """The non-strict tier must set it as well.

    The var only selects a slower deterministic backward; unlike
    ``warn_only=False`` it can never abort a run, so withholding it from the
    ``=1`` tier would buy no safety and would make that tier's determinism
    claim untrue for any attention model.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(DETERMINISM_ENV_VAR, "1")
    monkeypatch.delenv(FLASH_ATTN_ENV_VAR, raising=False)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)

    assert enable_determinism(42) is True
    assert os.environ[FLASH_ATTN_ENV_VAR] == FLASH_ATTN_REQUIRED_VALUE


def test_flash_attn_determinism_env_untouched_when_gate_off(monkeypatch):
    """A normal training run must keep the fast nondeterministic kernel."""
    monkeypatch.delenv(DETERMINISM_ENV_VAR, raising=False)
    monkeypatch.delenv(FLASH_ATTN_ENV_VAR, raising=False)

    assert enable_determinism(42) is False
    assert FLASH_ATTN_ENV_VAR not in os.environ


def test_flash_attn_env_value_is_exactly_what_transformers_compares(monkeypatch):
    """transformers tests ``os.getenv(...) == "1"`` -- an exact string match.

    So "true"/"yes"/"on" would all read as disabled. Pin the literal.
    """
    assert FLASH_ATTN_REQUIRED_VALUE == "1"
    assert FLASH_ATTN_ENV_VAR == "FLASH_ATTENTION_DETERMINISTIC"
