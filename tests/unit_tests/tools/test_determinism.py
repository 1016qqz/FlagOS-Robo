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

import random

import pytest

from flagos_robo.utils.determinism import (
    CUBLAS_ENV_VAR,
    CUBLAS_REQUIRED_VALUE,
    DETERMINISM_ENV_VAR,
    determinism_enabled,
    enable_determinism,
    make_dataloader_generator,
    seed_worker,
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
