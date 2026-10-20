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
"""Determinism switches shared by the FlagOS-Robo and reference training loops.

Precision alignment compares two runs step by step, so any uncontrolled source
of randomness shows up as a false divergence. Both loops import *this* module
rather than each keeping its own copy, which is what makes "the two sides use
identical settings" a property of the code instead of a review promise.

Off by default. ``FLAGOS_DETERMINISTIC`` gates it, matching how
``precision_dump.dump_dir_from_env`` gates dumping, because a determinism flag
costs real throughput (deterministic kernels are slower) and must not be paid
by ordinary training runs.

The switch list follows the PyTorch reproducibility notes, confirmed against
the installed torch's own docstring:

* ``torch.use_deterministic_algorithms(True)`` raises ``RuntimeError`` on CUDA
  >= 10.2 unless ``CUBLAS_WORKSPACE_CONFIG`` is ``:4096:8`` or ``:16:8``, and
  that variable is read when the cuBLAS handle is created -- setting it after
  CUDA init is too late. We set it here (best effort) and warn if it already
  held some other value, so a stale export cannot silently break the run.
* The same call raises on ops that have no deterministic kernel at all (e.g.
  ``kthvalue`` on CUDA). A hard crash mid-training is the wrong trade for an
  ordinary run, so ``warn_only=True`` stays the default.

``warn_only`` is therefore a two-tier policy, not a cosmetic error-handling
choice. Measured on this model (W4 D2): with ``warn_only=True`` two runs of
byte-identical config were bit-identical at step 1 and then diverged from step 2
on, amplifying 140x by step 3. The sole determinism warning in the whole rank-0
log named its own cause::

    Flash Attention defaults to a non-deterministic algorithm. To explicitly
    enable determinism call torch.use_deterministic_algorithms(True,
    warn_only=False).

So ``warn_only`` also participates in *kernel selection*: a deterministic Flash
Attention backward exists but is chosen only when ``warn_only=False``. Under
``warn_only=True`` the nondeterministic kernel still runs and merely warns --
which is exactly the signature observed (FA is nondeterministic in backward
only, hence a bit-exact forward at step 1 and divergence from step 2).

Hence the gate has three states:

``FLAGOS_DETERMINISTIC`` unset/``0``/``false``/``no``/``off``
    Off. Normal throughput, no determinism cost.
``FLAGOS_DETERMINISTIC=1`` (or any other truthy value)
    On with ``warn_only=True``. Removes controllable nondeterminism without
    ever aborting training. Not sufficient for bit-exactness.
``FLAGOS_DETERMINISTIC=strict``
    On with ``warn_only=False``. Selects deterministic kernels, which is what
    L5 (per-step bit-exact) alignment requires. An op with no deterministic
    implementation now raises instead of warning -- that is the intended
    trade for a comparison run, and the error names the op.
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Mapping

logger = logging.getLogger(__name__)

DETERMINISM_ENV_VAR = "FLAGOS_DETERMINISTIC"
CUBLAS_ENV_VAR = "CUBLAS_WORKSPACE_CONFIG"
# :4096:8 rather than :16:8 -- the larger workspace keeps multi-stream cuBLAS
# usable, where :16:8 restricts it and costs more performance.
CUBLAS_REQUIRED_VALUE = ":4096:8"
_CUBLAS_ACCEPTED_VALUES = frozenset({":4096:8", ":16:8"})
# The one value that additionally turns off ``warn_only``; see module docstring.
DETERMINISM_STRICT_VALUE = "strict"
_OFF_VALUES = frozenset({"0", "false", "no", "off"})
# Flash Attention reached through HuggingFace ``_attn_implementation=
# "flash_attention_2"`` is the ``flash_attn`` package, not a torch operator, so
# ``torch.use_deterministic_algorithms`` has no authority over it -- it neither
# raises nor warns. transformers forwards its own switch:
#   modeling_flash_attention_utils.py
#     flash_kwargs["deterministic"] = (
#         deterministic if deterministic is not None
#         else os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0") == "1")
# The default "0" keeps the atomicAdd dq accumulation in the backward pass,
# which is why strict torch determinism alone still produced a differing
# step-1 grad_norm (13.856263 vs 13.859310) on gr00t_n1_5. Setting this cannot
# abort a run -- it only picks a slower deterministic kernel -- so it is applied
# in both enabled tiers rather than strict alone.
FLASH_ATTN_ENV_VAR = "FLASH_ATTENTION_DETERMINISTIC"
FLASH_ATTN_REQUIRED_VALUE = "1"

__all__ = [
    "CUBLAS_ENV_VAR",
    "CUBLAS_REQUIRED_VALUE",
    "DETERMINISM_ENV_VAR",
    "DETERMINISM_STRICT_VALUE",
    "FLASH_ATTN_ENV_VAR",
    "FLASH_ATTN_REQUIRED_VALUE",
    "determinism_enabled",
    "enable_determinism",
    "make_dataloader_generator",
    "seed_worker",
    "strict_determinism_enabled",
]


def determinism_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether deterministic mode is requested.

    Mirrors ``precision_dump.dump_dir_from_env`` exactly in how it reads the
    environment, so the two gates cannot disagree about what "off" means.
    ``env`` is injectable so tests need not mutate ``os.environ``.
    """
    source = os.environ if env is None else env
    raw = source.get(DETERMINISM_ENV_VAR)
    if raw is None:
        return False
    value = raw.strip()
    if not value:
        return False
    return value.lower() not in _OFF_VALUES


def strict_determinism_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether the *strict* tier is requested (``warn_only=False``).

    Strict is a sub-state of enabled, never a separate axis: anything strict
    accepts, :func:`determinism_enabled` also accepts. Only the exact value
    ``strict`` (case-insensitive, surrounding space tolerated) selects it --
    deliberately not "any value other than 1", so a typo degrades to the safe
    non-aborting tier instead of turning a training run fatal.
    """
    source = os.environ if env is None else env
    raw = source.get(DETERMINISM_ENV_VAR)
    if raw is None:
        return False
    return raw.strip().lower() == DETERMINISM_STRICT_VALUE


def seed_worker(worker_id: int) -> None:  # noqa: ARG001 - DataLoader API
    """Re-seed a DataLoader worker's Python and NumPy RNGs.

    ``torch.manual_seed`` in the parent does not reach these: each worker is a
    fresh process, and while PyTorch gives it a derived torch seed, the global
    ``random`` and ``numpy.random`` state is not derived from it. Augmentations
    built on those two would therefore differ run to run -- and between the two
    sides we are trying to compare.

    ``torch.initial_seed()`` inside the worker already encodes the base seed
    plus the worker id, so deriving from it keeps workers distinct while
    staying reproducible. Masked to 32 bits because that is NumPy's accepted
    range, whereas torch seeds are 64-bit.
    """
    import torch  # local: keeps this module importable without torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        return
    np.random.seed(worker_seed)


def make_dataloader_generator(seed: int):
    """Return a seeded ``torch.Generator`` for ``DataLoader(generator=...)``.

    Without an explicit generator the loader draws its shuffling and worker
    base seeds from the global RNG, whose position depends on how many random
    draws happened earlier in the process. Two runs that differ in any earlier
    random use would then order batches differently even with the same seed.
    """
    import torch  # local: keeps this module importable without torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def enable_determinism(seed: int, *, warn_only: bool | None = None) -> bool:
    """Apply the deterministic switches. Returns whether they were applied.

    No-op unless ``FLAGOS_DETERMINISTIC`` is set, so a normal run keeps the
    faster nondeterministic kernels.

    ``warn_only`` defaults to ``None`` meaning "read the tier from the
    environment": ``strict`` -> ``False``, any other truthy value -> ``True``.
    Callers do not have to know about the tier, so the training loops stay a
    single ``enable_determinism(seed)`` call on both sides and the policy lives
    in one place. Pass an explicit bool only to override (tests do).
    """
    if not determinism_enabled():
        return False

    if warn_only is None:
        warn_only = not strict_determinism_enabled()

    current = os.environ.get(CUBLAS_ENV_VAR)
    if current not in _CUBLAS_ACCEPTED_VALUES:
        if current:
            # Loud, because this is the case that produces a confusing
            # RuntimeError deep inside a matmul rather than at startup.
            logger.warning(
                "%s=%r is not one of %s; overriding with %r. If CUDA is "
                "already initialised this takes effect only next run.",
                CUBLAS_ENV_VAR, current,
                sorted(_CUBLAS_ACCEPTED_VALUES), CUBLAS_REQUIRED_VALUE,
            )
        os.environ[CUBLAS_ENV_VAR] = CUBLAS_REQUIRED_VALUE

    # Set before the model is built: transformers reads this env var at every
    # attention call, so an export that lands after the first forward would
    # silently apply to only part of the run.
    os.environ[FLASH_ATTN_ENV_VAR] = FLASH_ATTN_REQUIRED_VALUE

    import torch

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # TF32 silently reduces matmul mantissa precision; two sides could
        # differ purely by kernel selection, so pin it off for comparisons.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    logger.info(
        "Determinism enabled (seed=%d, warn_only=%s, %s=%s)",
        seed, warn_only, CUBLAS_ENV_VAR, os.environ.get(CUBLAS_ENV_VAR),
    )
    return True
