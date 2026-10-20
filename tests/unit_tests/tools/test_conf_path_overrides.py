"""Tests for env-overridable input paths in the example train configs.

Two properties are under test, and they pull in opposite directions:

1. BACKWARD COMPATIBILITY -- with no FLAGOS_* env set, every path must resolve
   to exactly the literal it had before the ``${oc.env:...}`` rewrite. A typo in
   a default would silently repoint training at a non-existent dataset.
2. OVERRIDABILITY -- setting FLAGOS_CHECKPOINT_DIR / FLAGOS_TOKENIZER_PATH /
   FLAGOS_DATA_PATH must actually change the resolved value. This is what
   acceptance item A8-a exercises via tools/check_conf_paths.py.

The configs cannot be loaded standalone: ``train/<name>.yaml`` interpolates
``${experiment.exp_dir}``, which only exists in the outer ``conf/train.yaml``.
So we compose the same way check_conf_paths.py (and Hydra) does.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKER = REPO_ROOT / "tools" / "check_conf_paths.py"

OmegaConf = pytest.importorskip("omegaconf").OmegaConf

# (example name, dotted key, expected default value)
EXPECTED_DEFAULTS = [
    ("pi0", "train.model.checkpoint_dir", "/workspace/models/lerobot/pi0_base"),
    ("pi0", "train.model.tokenizer_path", "/workspace/models/google/paligemma-3b-pt-224"),
    ("pi0", "train.data.data_path", "/workspace/datasets/lerobot/aloha_mobile_cabinet"),
    ("pi0_5", "train.model.checkpoint_dir", "/workspace/models/lerobot/pi05_libero_base"),
    ("pi0_5", "train.model.tokenizer_path", "/workspace/models/google/paligemma-3b-pt-224"),
    (
        "pi0_5",
        "train.data.data_path",
        "/workspace/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
    ),
    ("gr00t_n1_5", "train.model.checkpoint_dir", "/workspace/models/nvidia/GR00T-N1.5-3B"),
    (
        "gr00t_n1_5",
        "train.data.data_path",
        "/workspace/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot/",
    ),
]

ENV_VARS = ("FLAGOS_CHECKPOINT_DIR", "FLAGOS_TOKENIZER_PATH", "FLAGOS_DATA_PATH")

KEY_TO_ENV = {
    "checkpoint_dir": "FLAGOS_CHECKPOINT_DIR",
    "tokenizer_path": "FLAGOS_TOKENIZER_PATH",
    "data_path": "FLAGOS_DATA_PATH",
}


def _load_checker():
    """Import tools/check_conf_paths.py by path (tools/ is not an importable pkg)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_check_conf_paths", CHECKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conf_path(example: str) -> Path:
    return REPO_ROOT / "examples" / example / "conf" / "train.yaml"


def _group_file(example: str) -> Path:
    """Resolve the ``defaults: [{train: <name>}]`` group file for an example."""
    outer_path = _conf_path(example)
    outer = OmegaConf.load(outer_path)
    for entry in outer.get("defaults") or []:
        if isinstance(entry, str):
            continue  # "_self_"
        for group, name in entry.items():
            if str(group) == "train" and name is not None:
                return outer_path.parent / "train" / f"{name}.yaml"
    raise AssertionError(f"{outer_path} has no 'train: <name>' default")


def _compose(example: str):
    """Compose exactly the way the checker (and Hydra) does, so the test and
    the shipped tool cannot drift apart."""
    return _load_checker().load_composed(_conf_path(example))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("example,key,expected", EXPECTED_DEFAULTS)
def test_default_resolves_to_original_hardcoded_path(example, key, expected):
    cfg = _compose(example)
    assert OmegaConf.select(cfg, key) == expected


@pytest.mark.parametrize("example,key,expected", EXPECTED_DEFAULTS)
def test_env_override_changes_resolved_value(example, key, expected, monkeypatch):
    env_var = KEY_TO_ENV[key.rsplit(".", 1)[1]]
    sentinel = f"/tmp/override-sentinel/{example}/{env_var}"
    monkeypatch.setenv(env_var, sentinel)
    cfg = _compose(example)
    resolved = OmegaConf.select(cfg, key)
    assert resolved == sentinel, f"{key} ignored ${env_var} (got {resolved!r})"
    assert resolved != expected


def test_every_input_path_is_env_overridable():
    """No input path may stay hardcoded -- that is the whole point of D4."""
    hardcoded = []
    for example, key, _ in EXPECTED_DEFAULTS:
        raw = OmegaConf.load(_group_file(example))
        leaf = key.split(".", 1)[1]  # drop the "train." prefix
        parent, attr = leaf.rsplit(".", 1)
        node = OmegaConf.select(raw, parent)
        literal = node._get_node(attr)._value()  # pre-resolution text
        if "oc.env:" not in str(literal):
            hardcoded.append((example, key, literal))
    assert hardcoded == []


def test_checker_reports_the_overridden_path_verbatim(tmp_path):
    """A8-a: bad override must surface in the failure text, not be swallowed."""
    bogus = "/nonexistent/path"
    env = dict(os.environ, FLAGOS_TOKENIZER_PATH=bogus)
    proc = subprocess.run(
        [sys.executable, str(CHECKER), "examples/pi0/conf/train.yaml"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert bogus in proc.stdout + proc.stderr
