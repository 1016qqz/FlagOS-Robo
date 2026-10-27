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

"""Serve entrypoint must register every servable policy type.

``TrainablePolicy`` / ``PreTrainedConfig`` dispatch is registry-based: a type only
becomes resolvable once its module has been imported, because registration happens in
``__init_subclass__``. ``flagos_robo/models/vla/__init__.py`` deliberately does not
import pi0/pi05 (circular), so each entrypoint registers them itself.

Regression guard for the observed failure when the serve entrypoint skipped them::

    ValueError: Unknown config type 'pi0'. Known types: ['Gr00tN15']
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVE_ENTRY = REPO_ROOT / "flagos_robo" / "serve" / "run_serve_vla.py"
VLA_INIT = REPO_ROOT / "flagos_robo" / "models" / "vla" / "__init__.py"

# Registry keys come from the class name with the suffix stripped: PI0Config -> "PI0",
# Gr00tN15Config -> "Gr00tN15". Lookup is case-insensitive (from_pretrained falls back
# to a lowercase match) so a lerobot checkpoint with "type": "pi0" still resolves --
# but only if the module was imported first.
EXPECTED_MODULES = ("flagos_robo.models.pi0", "flagos_robo.models.pi05")

# Processor steps live in a second, independent registry (ProcessorStepRegistry). The
# GR00T steps are registered by decorators in models/vla/gr00t_n1_5/processor_gr00t.py,
# which vla/gr00t_n1_5/__init__.py does not reach (it only re-exports
# configuration_/modeling_). Serving a GR00T checkpoint without this import dies with::
#
#     KeyError: "Processor step 'groot_pack_inputs' not found in registry"
EXPECTED_PROCESSOR_MODULE = "flagos_robo.models.vla.gr00t_n1_5.processor_gr00t"


def _toplevel_imported_modules(path: Path) -> set[str]:
    """Return every module name imported at module level by *path*."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _function_source(path: Path, qualname: str) -> str:
    """Return the source segment of a (possibly nested) function definition.

    *qualname* is a dotted path of def/class names, e.g. ``"Engine.load_policy"``.
    Raises AssertionError if any component is missing, so a renamed-away target fails
    loudly instead of silently vacuously passing.
    """
    source = path.read_text()
    node: ast.AST = ast.parse(source)
    for part in qualname.split("."):
        children = getattr(node, "body", [])
        match = next(
            (
                child
                for child in children
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and child.name == part
            ),
            None,
        )
        assert match is not None, f"{path.name} has no '{part}' (looking up {qualname})"
        node = match
    return ast.get_source_segment(source, node) or ""


def test_serve_entry_exists():
    assert SERVE_ENTRY.is_file(), f"serve entrypoint missing: {SERVE_ENTRY}"


@pytest.mark.parametrize("module", EXPECTED_MODULES)
def test_serve_entry_registers_pi_policies(module):
    """The serve entry must pull in pi0/pi05 so their types land in the registries."""
    imported = _toplevel_imported_modules(SERVE_ENTRY)
    # Either the package or a submodule of it is enough to trigger registration.
    assert any(name == module or name.startswith(module + ".") for name in imported), (
        f"{SERVE_ENTRY.name} does not import {module}; pi0/pi05 checkpoints will fail to serve"
    )


def test_vla_package_still_does_not_import_pi():
    """Guard the premise: the fix belongs in the entrypoint, not in vla/__init__.py.

    Re-exporting pi0 from ``vla/__init__.py`` is circular (pi0.modeling_pi0 imports
    back into vla). If someone "fixes" it there instead, this test fails and points
    back at the entrypoint-side contract documented above.
    """
    imported = _toplevel_imported_modules(VLA_INIT)
    offenders = [n for n in imported if "pi0" in n or "pi05" in n]
    assert not offenders, f"vla/__init__.py must not import pi0/pi05 (circular): {offenders}"


def test_serve_entry_registers_gr00t_processor_steps():
    """The serve entry must pull in processor_gr00t so its steps land in the registry.

    Mirrors train_gr00t_n1_5.py, which carries the same side-effect import.
    """
    imported = _toplevel_imported_modules(SERVE_ENTRY)
    assert EXPECTED_PROCESSOR_MODULE in imported, (
        f"{SERVE_ENTRY.name} does not import {EXPECTED_PROCESSOR_MODULE}; "
        "GR00T checkpoints will fail with KeyError 'groot_pack_inputs'"
    )


def test_load_policy_overrides_checkpoint_device():
    """``load_policy`` must override the checkpoint's recorded device.

    PI0Policy/PI05Policy.from_pretrained take no ``device`` parameter, so the argument
    passed by base_policy.py lands in ``**kwargs`` and is dropped; the weights are moved
    by ``model.to(config.device)`` inside the policy loader, i.e. the checkpoint's own
    device field wins. pi05_libero_base ships ``"device": "mps"``, which on a CUDA box
    raises "PyTorch is not linked with support for mps devices". The entrypoint
    therefore has to load the config and rewrite ``device`` before handing it over.
    """
    src = _function_source(SERVE_ENTRY, "Policy.load_policy")
    tree = ast.parse(src.strip())

    # (a) an assignment to some `<name>.device`
    device_writes = [
        target
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Attribute) and target.attr == "device"
    ]
    assert device_writes, (
        "load_policy never assigns to <config>.device; a checkpoint recording "
        "device='mps' would still be honoured and crash on a CUDA box"
    )

    # (b) the rewritten config is actually handed to from_pretrained
    config_kwarg_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and any(kw.arg == "config" for kw in node.keywords)
    ]
    assert config_kwarg_calls, (
        "load_policy calls from_pretrained without a config= argument, so the "
        "device override cannot take effect"
    )
