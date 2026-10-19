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
"""Shared helpers for FlagOS-Robo.

Kept import-light on purpose: this package is imported from training
entrypoints, so any heavy import here lands on every training start.
``precision_dump`` is deliberately NOT re-exported -- it touches ``torch`` and
is only needed when precision dumping is switched on. Import it explicitly:

    from flagos_robo.utils.precision_dump import dump_step
"""

__all__: list[str] = []
