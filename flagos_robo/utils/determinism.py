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
  alignment aid, so ``warn_only=True`` is the default: controllable
  nondeterminism is removed, the rest is reported instead of fatal.
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

__all__ = [
    "CUBLAS_ENV_VAR",
    "CUBLAS_REQUIRED_VALUE",
    "DETERMINISM_ENV_VAR",
    "determinism_enabled",
    "enable_determinism",
    "make_dataloader_generator",
    "seed_worker",
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
    return value.lower() not in {"0", "false", "no", "off"}


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


def enable_determinism(seed: int, *, warn_only: bool = True) -> bool:
    """Apply the deterministic switches. Returns whether they were applied.

    No-op unless ``FLAGOS_DETERMINISTIC`` is set, so a normal run keeps the
    faster nondeterministic kernels.
    """
    if not determinism_enabled():
        return False

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
