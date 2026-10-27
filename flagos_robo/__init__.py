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

"""FlagOS-Robo: an embodied-AI training & inference development kit.

W1 scope: repository skeleton + hardware platform abstraction layer.
This package must stay importable in a **pure PyTorch** environment with
**no Megatron** in the dependency tree (see ``tools/check_no_megatron.py``).
"""

__all__ = ["__version__"]

_FALLBACK_VERSION = "0.1.0.dev0"


def _detect_version() -> str:
    """Resolve the package version.

    Order:
    1. installed distribution metadata (``importlib.metadata``);
    2. parse ``pyproject.toml`` when running from a source checkout that was
       never installed (keeps ``python -c "import flagos_robo"`` working from
       the repo root, which is how CI smoke-tests it);
    3. a hardcoded fallback.
    """
    try:
        from importlib.metadata import version

        return version("flagos-robo")
    except Exception:
        pass

    try:
        import pathlib

        import tomllib

        pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.is_file():
            with pyproject.open("rb") as handle:
                return tomllib.load(handle)["project"]["version"]
    except Exception:
        pass

    return _FALLBACK_VERSION


__version__ = _detect_version()
