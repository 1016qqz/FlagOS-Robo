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
"""W1 D3 - tests for the no-Megatron acceptance gate.

The gate's value rests entirely on it being able to FAIL. These tests plant
violations in a temp repo and assert detection, and - equally important -
plant the documentation forms that must NOT trip it.

check-no-megatron: allow-self-reference
"""

import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[3] / "tools" / "check_no_megatron.py"
sys.path.insert(0, str(TOOL.parent))

import check_no_megatron as cnm  # noqa: E402


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "flagos_robo").mkdir()
    (tmp_path / "requirements" / "cuda").mkdir(parents=True)
    (tmp_path / "requirements" / "cuda" / "train.txt").write_text("torch>=2.0\n")
    return tmp_path


def _w(repo: Path, rel: str, text: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


class TestL1Imports:
    @pytest.mark.parametrize(
        "src",
        [
            "import megatron\n",
            "import megatron.core\n",
            "from megatron.core import parallel_state\n",
            "from megatron import plugin\n",
            "from megatron.plugin.platform import get_platform\n",
            "import megatron_core\n",
        ],
    )
    def test_detects_import_forms(self, repo, src):
        _w(repo, "flagos_robo/m.py", src)
        assert len(cnm.check_l1_imports(repo)) == 1

    @pytest.mark.parametrize(
        "src",
        [
            "import torch\n",
            "from flagos_robo.platforms import get_platform\n",
            "# import megatron\n",
            '"""mentions megatron.plugin.platform in prose"""\n',
            "import megatronics\n",  # unrelated name that merely starts the same
        ],
    )
    def test_no_false_positive(self, repo, src):
        _w(repo, "flagos_robo/m.py", src)
        assert cnm.check_l1_imports(repo) == []

    def test_pragma_does_not_exempt_imports(self, repo):
        """A file may opt out of the text scan, never out of the import scan."""
        _w(
            repo,
            "flagos_robo/m.py",
            f'"""{cnm.ALLOW_PRAGMA}"""\nfrom megatron.core import x\n',
        )
        assert len(cnm.check_l1_imports(repo)) == 1


class TestL2CodeText:
    def test_detects_dynamic_import_string(self, repo):
        """The blind spot L1 structurally cannot see."""
        _w(repo, "flagos_robo/m.py", 'import importlib\nm = importlib.import_module("megatron.core.foo")\n')
        assert cnm.check_l1_imports(repo) == []
        assert len(cnm.check_l2_code_text(repo)) >= 1

    def test_detects_getattr_style_reference(self, repo):
        _w(repo, "flagos_robo/m.py", 'CLASS_PATH = "megatron.plugin.platform.get_platform"\n')
        assert len(cnm.check_l2_code_text(repo)) >= 1

    def test_comment_mention_is_not_a_violation(self, repo):
        _w(repo, "flagos_robo/m.py", "# megatron_core must never be a dependency\nX = 1\n")
        assert cnm.check_l2_code_text(repo) == []

    def test_module_docstring_mention_is_not_a_violation(self, repo):
        _w(repo, "flagos_robo/m.py", '"""Upstream bound megatron.plugin.platform here."""\nX = 1\n')
        assert cnm.check_l2_code_text(repo) == []

    def test_function_docstring_mention_is_not_a_violation(self, repo):
        _w(
            repo,
            "flagos_robo/m.py",
            "def f():\n    '''explains megatron.plugin.platform drift'''\n    return 1\n",
        )
        assert cnm.check_l2_code_text(repo) == []

    def test_pragma_exempts_text_scan(self, repo):
        _w(repo, "flagos_robo/m.py", f'"""{cnm.ALLOW_PRAGMA}"""\nNAME = "megatron"\n')
        assert cnm.check_l2_code_text(repo) == []


class TestL3DeclaredDeps:
    @pytest.mark.parametrize(
        "line",
        [
            "megatron_core>=0.9.0",
            "megatron-core==0.9.0",
            "Megatron-Core",
            "megatron_core[dev]>=0.9 ; python_version>'3.10'",
            "megatron",
        ],
    )
    def test_detects_requirement_forms(self, repo, line):
        """PEP 503 normalisation: _ . - and case all collapse to one name."""
        _w(repo, "requirements/cuda/extra.txt", line + "\n")
        assert len(cnm.check_l3_declared_deps(repo)) == 1

    @pytest.mark.parametrize(
        "line",
        [
            "# megatron_core is FORBIDDEN - this repo must stay Megatron-free",
            "torch>=2.0  # not megatron_core",
            "-r ./base.txt",
            "transformers>=4.40",
        ],
    )
    def test_comment_and_option_lines_are_not_requirements(self, repo, line):
        """The exact false positive that makes `grep -i megatron` unusable:
        our own requirements file documents megatron_core as banned."""
        _w(repo, "requirements/cuda/extra.txt", line + "\n")
        assert cnm.check_l3_declared_deps(repo) == []

    def test_detects_pyproject_dependency(self, repo):
        _w(
            repo,
            "pyproject.toml",
            '[project]\nname = "x"\nversion = "1"\ndependencies = ["torch", "megatron-core>=0.9"]\n',
        )
        hits = cnm.check_l3_declared_deps(repo)
        assert len(hits) == 1 and "dependencies" in hits[0].detail

    def test_detects_pyproject_optional_dependency(self, repo):
        _w(
            repo,
            "pyproject.toml",
            '[project]\nname = "x"\nversion = "1"\n'
            '[project.optional-dependencies]\ntrain = ["megatron_core"]\n',
        )
        hits = cnm.check_l3_declared_deps(repo)
        assert len(hits) == 1 and "optional-dependencies.train" in hits[0].detail

    def test_clean_pyproject_passes(self, repo):
        _w(
            repo,
            "pyproject.toml",
            '[project]\nname = "x"\nversion = "1"\ndependencies = ["torch", "hydra-core"]\n',
        )
        assert cnm.check_l3_declared_deps(repo) == []


class TestL4Env:
    def test_current_env_is_clean(self):
        """Far-end check: this must hold in whatever env actually runs the tests."""
        assert cnm.check_l4_installed_env() == []

    @staticmethod
    def _plant(root, submodule, dist_name, version):
        pkg = root / "megatron" / submodule
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("")
        di = root / f"{dist_name.replace('-', '_')}-{version}.dist-info"
        di.mkdir(exist_ok=True)
        (di / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n"
        )
        (di / "RECORD").write_text("")

    @staticmethod
    def _scan(root, monkeypatch, allow_energon):
        import importlib

        monkeypatch.syspath_prepend(str(root))
        for mod in [m for m in list(sys.modules) if m.startswith("megatron")]:
            monkeypatch.delitem(sys.modules, mod, raising=False)
        importlib.invalidate_caches()
        return cnm.check_l4_installed_env(allow_energon=allow_energon)

    def test_detects_planted_dist_and_module(self, tmp_path, monkeypatch):
        """Proves L4's clean verdict is a real result, not a no-op."""
        self._plant(tmp_path, "core", "megatron-core", "0.9.0")
        details = " ".join(h.detail for h in self._scan(tmp_path, monkeypatch, False))
        assert "megatron-core" in details
        assert "megatron.core is importable" in details

    def test_bare_namespace_alone_is_not_a_violation(self, tmp_path, monkeypatch):
        """A bare `megatron/` namespace dir with no known submodule is not
        Megatron-LM. Probing the top-level name would false-positive here."""
        (tmp_path / "megatron").mkdir()  # PEP 420: deliberately no __init__.py
        assert self._scan(tmp_path, monkeypatch, False) == []

    def test_energon_flagged_by_default(self, tmp_path, monkeypatch):
        """W1 default is strictest: energon is NOT silently tolerated."""
        self._plant(tmp_path, "energon", "megatron-energon", "7.4.1")
        details = " ".join(h.detail for h in self._scan(tmp_path, monkeypatch, False))
        assert "megatron-energon" in details
        assert "megatron.energon is importable" in details

    def test_energon_allowed_by_flag(self, tmp_path, monkeypatch):
        self._plant(tmp_path, "energon", "megatron-energon", "7.4.1")
        assert self._scan(tmp_path, monkeypatch, True) == []

    def test_exception_cannot_smuggle_core(self, tmp_path, monkeypatch):
        """The narrowness guarantee: --allow-energon must not open the door to
        megatron-core riding along in the same environment."""
        self._plant(tmp_path, "energon", "megatron-energon", "7.4.1")
        self._plant(tmp_path, "core", "megatron-core", "0.9.0")
        details = " ".join(h.detail for h in self._scan(tmp_path, monkeypatch, True))
        assert "megatron-core" in details
        assert "megatron.core is importable" in details
        assert "energon" not in details


class TestL3EnergonException:
    def test_energon_requirement_flagged_by_default(self, repo):
        _w(repo, "requirements/cuda/extra.txt", "megatron-energon>=7.4\n")
        assert len(cnm.check_l3_declared_deps(repo)) == 1

    def test_energon_requirement_allowed_by_flag(self, repo):
        _w(repo, "requirements/cuda/extra.txt", "megatron-energon>=7.4\n")
        assert cnm.check_l3_declared_deps(repo, allow_energon=True) == []

    def test_flag_does_not_whitelist_core(self, repo):
        _w(repo, "requirements/cuda/extra.txt", "megatron-energon>=7.4\nmegatron_core>=0.9\n")
        hits = cnm.check_l3_declared_deps(repo, allow_energon=True)
        assert len(hits) == 1 and "megatron-core" in hits[0].detail


class TestCLI:
    def test_real_repo_passes(self):
        r = subprocess.run(
            [sys.executable, str(TOOL), "--check-env"], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "PASS" in r.stdout

    def test_self_test_passes(self):
        r = subprocess.run([sys.executable, str(TOOL), "--self-test"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "SELF-TEST PASSED" in r.stdout

    def test_exit_code_1_on_violation(self, repo):
        _w(repo, "flagos_robo/bad.py", "from megatron.core import x\n")
        r = subprocess.run(
            [sys.executable, str(TOOL), "--root", str(repo)], capture_output=True, text=True
        )
        assert r.returncode == 1
        assert "FAIL" in r.stdout

    def test_json_output_is_parseable(self, repo):
        _w(repo, "flagos_robo/bad.py", "import megatron\n")
        r = subprocess.run(
            [sys.executable, str(TOOL), "--root", str(repo), "--json"],
            capture_output=True,
            text=True,
        )
        import json

        data = json.loads(r.stdout)
        assert data and data[0]["layer"] == "L1"
