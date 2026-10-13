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

# This file makes tests/ a Python package.
#
# It is required: tests/unit_tests/train/utils/test_train_utils.py imports the
# shared helper via the repo-rooted path `tests.unit_tests.test_utilities`.
# Without this file (and tests/unit_tests/__init__.py) that import raises
# ModuleNotFoundError at pytest *collection* time, which aborts the whole run
# with exit code 2 and zero tests executed.
