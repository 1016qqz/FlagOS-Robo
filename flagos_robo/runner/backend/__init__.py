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

# FlagOS-Robo ships only the native training backend.  Upstream FlagScale
# re-exported seven more here (MegatronBackend, LlamaCppBackend,
# NativeCompressBackend, NativeServeBackend, SglangBackend, VerlBackend,
# VllmBackend); those modules are outside this repo's scope, so importing
# them would raise ModuleNotFoundError at package import time.
from flagos_robo.runner.backend.backend_base import BackendBase
from flagos_robo.runner.backend.backend_native_train import NativeTrainBackend

__all__ = ["BackendBase", "NativeTrainBackend"]
