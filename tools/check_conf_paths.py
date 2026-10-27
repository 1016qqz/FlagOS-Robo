#!/usr/bin/env python3
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
"""Assert that every INPUT path declared in an experiment conf really exists.

Why this gate exists
--------------------
Model / tokenizer / dataset paths in ``examples/*/conf`` are absolute paths that
are valid only on the machine the conf was authored on. A wrong one does not
fail fast: training loads for ~40s, allocates the GPUs, and only then dies deep
inside a HuggingFace loader with a message that does not name the conf key. On a
six-card job that is a minute of wall-clock and a dirty log per attempt.

So the paths are made overridable through ``${oc.env:FLAGOS_*,<default>}`` and
this tool resolves the conf the same way the launcher does, then stats every
input path *before* anything is launched.

Conf composition
----------------
The confs are two-level Hydra: an outer ``train.yaml`` (experiment/runner/envs)
plus a ``defaults: [train: <name>]`` group file in ``conf/train/<name>.yaml``
that lands under the ``train`` key. Loading the group file ALONE raises
``InterpolationKeyError: experiment.exp_dir`` because its
``output_directory: ${experiment.exp_dir}`` points into the outer tree, so this
tool always composes both halves before resolving.

INPUT vs OUTPUT paths
---------------------
Only inputs are required to pre-exist. ``exp_dir`` / ``output_directory`` are
created by the run, so a missing one is not a violation - they are classified
out by key name (see ``OUTPUT_KEYS``), not skipped ad hoc.

Exit codes (same convention as compare_precision.py)
  0 = every input path resolves and exists
  1 = at least one violation (missing / empty / unresolvable)
  2 = usage or conf-parse error (file absent, bad YAML, bad defaults list)

``--self-test`` is a negative control: it proves the checker can actually fail.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
    from omegaconf.errors import OmegaConfBaseException
except Exception as exc:  # pragma: no cover - env without omegaconf
    print(f"check_conf_paths: omegaconf is required ({exc})", file=sys.stderr)
    raise SystemExit(2)


# Keys whose value is an input path that MUST already exist.
INPUT_KEY_RE = re.compile(r"(^|[._])(checkpoint_dir|tokenizer_path|data_path|load)$")

# Keys that name a path the RUN creates. Absence is not a violation.
OUTPUT_KEYS = ("output_directory", "exp_dir")

# Env var that overrides each input key, by leaf key name.
ENV_FOR_KEY = {
    "checkpoint_dir": "FLAGOS_CHECKPOINT_DIR",
    "tokenizer_path": "FLAGOS_TOKENIZER_PATH",
    "data_path": "FLAGOS_DATA_PATH",
}


class Violation:
    """One failing path, carrying enough context to fix the conf without grepping."""

    def __init__(self, dotted: str, value: str, reason: str, env_var: str | None):
        self.dotted = dotted
        self.value = value
        self.reason = reason
        self.env_var = env_var

    def __str__(self) -> str:
        env = f" (override via ${self.env_var})" if self.env_var else ""
        return f"{self.dotted} = {self.value!r}: {self.reason}{env}"

    def as_dict(self) -> dict:
        return {
            "key": self.dotted,
            "value": self.value,
            "reason": self.reason,
            "env_var": self.env_var,
        }


def load_composed(conf_path: Path) -> DictConfig:
    """Compose outer train.yaml with its ``defaults: [train: <name>]`` group file.

    Raises SystemExit(2) on any parse problem - a conf we cannot read is a usage
    error, not a path violation, and the two must not share an exit code.
    """
    if not conf_path.is_file():
        print(f"check_conf_paths: no such conf file: {conf_path}", file=sys.stderr)
        raise SystemExit(2)
    try:
        outer = OmegaConf.load(conf_path)
    except Exception as exc:
        print(f"check_conf_paths: cannot parse {conf_path}: {exc}", file=sys.stderr)
        raise SystemExit(2)
    if not isinstance(outer, DictConfig):
        print(f"check_conf_paths: {conf_path} is not a mapping", file=sys.stderr)
        raise SystemExit(2)

    group_names: list[tuple[str, str]] = []
    for entry in outer.get("defaults") or []:
        if isinstance(entry, str):
            continue  # "_self_"
        if isinstance(entry, DictConfig):
            for group, name in entry.items():
                if name is not None:
                    group_names.append((str(group), str(name)))

    merged = outer
    for group, name in group_names:
        gpath = conf_path.parent / group / f"{name}.yaml"
        if not gpath.is_file():
            print(
                f"check_conf_paths: defaults list names {group}/{name} but {gpath} does not exist",
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            merged = OmegaConf.merge(merged, OmegaConf.create({group: OmegaConf.load(gpath)}))
        except Exception as exc:
            print(f"check_conf_paths: cannot parse {gpath}: {exc}", file=sys.stderr)
            raise SystemExit(2)
    merged.pop("defaults", None)
    return merged


def walk_leaves(node, prefix: str = "") -> Iterator[tuple[str, object]]:
    """Yield (dotted_key, raw_leaf) for every scalar in the tree.

    Resolution is deliberately deferred to the caller so an unresolvable
    interpolation surfaces as a violation on its own key instead of aborting
    the whole walk.
    """
    if isinstance(node, DictConfig):
        for key in node.keys():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            try:
                child = node._get_node(key)
            except Exception:
                continue
            if isinstance(child, (DictConfig, ListConfig)):
                yield from walk_leaves(child, dotted)
            else:
                yield dotted, child
    elif isinstance(node, ListConfig):
        for idx in range(len(node)):
            dotted = f"{prefix}[{idx}]"
            try:
                child = node._get_node(idx)
            except Exception:
                continue
            if isinstance(child, (DictConfig, ListConfig)):
                yield from walk_leaves(child, dotted)
            else:
                yield dotted, child


def is_input_path_key(dotted: str) -> bool:
    leaf = dotted.rsplit(".", 1)[-1]
    if leaf in OUTPUT_KEYS:
        return False
    return bool(INPUT_KEY_RE.search(leaf))


def check_conf(conf_path: Path, cfg: DictConfig) -> tuple[list[Violation], list[tuple[str, str]]]:
    """Return (violations, checked) for one composed conf."""
    violations: list[Violation] = []
    checked: list[tuple[str, str]] = []

    for dotted, _raw in walk_leaves(cfg):
        if not is_input_path_key(dotted):
            continue
        leaf = dotted.rsplit(".", 1)[-1]
        env_var = ENV_FOR_KEY.get(leaf)
        try:
            value = OmegaConf.select(cfg, dotted, throw_on_resolution_failure=True)
        except OmegaConfBaseException as exc:
            violations.append(Violation(dotted, "<unresolved>", f"cannot resolve: {exc}", env_var))
            continue

        # `load: null` means "train from scratch" - a real, valid choice.
        if value is None:
            continue
        text = str(value)
        if not text.strip():
            violations.append(Violation(dotted, text, "resolves to an empty string", env_var))
            continue
        checked.append((dotted, text))
        if not Path(text).exists():
            violations.append(Violation(dotted, text, "path does not exist", env_var))

    return violations, checked


def run_self_test() -> int:
    """Negative control: the checker must FAIL on a conf with a bad input path.

    Builds a throwaway two-level conf in a temp dir so the test exercises the
    real composition + resolution path rather than a stubbed one.
    """
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "train").mkdir()
        (root / "train.yaml").write_text(
            "defaults:\n  - _self_\n  - train: demo\n"
            "experiment:\n  exp_name: demo\n  exp_dir: outputs/${experiment.exp_name}\n"
            "  load: null\n",
            encoding="utf-8",
        )
        good = root / "real_tok"
        good.mkdir()
        (root / "train" / "demo.yaml").write_text(
            "system:\n  checkpoint:\n    output_directory: ${experiment.exp_dir}\n"
            f"model:\n  tokenizer_path: ${{oc.env:FLAGOS_TOKENIZER_PATH,{good}}}\n"
            f"data:\n  data_path: {good}\n",
            encoding="utf-8",
        )
        conf = root / "train.yaml"

        # case 1: all inputs exist -> clean
        os.environ.pop("FLAGOS_TOKENIZER_PATH", None)
        v, checked = check_conf(conf, load_composed(conf))
        c1 = (not v) and len(checked) == 2
        ok &= c1
        print(
            f"  [{'ok' if c1 else 'FAIL'}] clean conf -> 0 violations, checked {len(checked)} path(s)"
        )

        # case 2: env override to a missing path -> violation naming that path
        os.environ["FLAGOS_TOKENIZER_PATH"] = "/nonexistent/path"
        v, _ = check_conf(conf, load_composed(conf))
        c2 = len(v) == 1 and "/nonexistent/path" in str(v[0])
        ok &= c2
        print(f"  [{'ok' if c2 else 'FAIL'}] env override to missing path -> 1 violation naming it")

        # case 3: env override to empty string -> violation (not silently defaulted)
        os.environ["FLAGOS_TOKENIZER_PATH"] = "   "
        v, _ = check_conf(conf, load_composed(conf))
        c3 = len(v) == 1 and "empty" in v[0].reason
        ok &= c3
        print(f"  [{'ok' if c3 else 'FAIL'}] env override to blank -> flagged as empty")
        os.environ.pop("FLAGOS_TOKENIZER_PATH", None)

        # case 4: output-only keys are never required to pre-exist
        v, checked = check_conf(conf, load_composed(conf))
        names = [d for d, _ in checked]
        c4 = not any(n.endswith(OUTPUT_KEYS) for n in names)
        ok &= c4
        print(f"  [{'ok' if c4 else 'FAIL'}] output keys excluded from existence check")

        # case 5: a broken defaults list is exit 2, not a path violation
        (root / "bad.yaml").write_text("defaults:\n  - train: missing_group\n", encoding="utf-8")
        try:
            load_composed(root / "bad.yaml")
            c5 = False
        except SystemExit as exc:
            c5 = exc.code == 2
        ok &= c5
        print(
            f"  [{'ok' if c5 else 'FAIL'}] missing defaults group -> exit 2 (usage, not violation)"
        )

    print(f"\nSELF-TEST {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="check_conf_paths.py",
        description="Assert every input path in an experiment conf exists before launch.",
    )
    ap.add_argument(
        "conf",
        nargs="*",
        type=Path,
        help="one or more outer conf files, e.g. examples/pi0/conf/train.yaml",
    )
    ap.add_argument("--json", action="store_true", help="emit violations as JSON")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="negative control: prove the checker can fail",
    )
    args = ap.parse_args(argv)

    if args.self_test:
        print("check_conf_paths self-test")
        return run_self_test()

    if not args.conf:
        ap.print_usage(sys.stderr)
        print("check_conf_paths: no conf file given", file=sys.stderr)
        return 2

    all_violations: list[Violation] = []
    total_checked = 0
    for conf_path in args.conf:
        cfg = load_composed(conf_path)
        violations, checked = check_conf(conf_path, cfg)
        total_checked += len(checked)
        if not args.json:
            print(f"check_conf_paths: {conf_path}")
            for dotted, value in checked:
                mark = "ok " if Path(value).exists() else "MISSING"
                print(f"  [{mark}] {dotted} = {value}")
        all_violations += violations

    if args.json:
        import json

        print(json.dumps([v.as_dict() for v in all_violations], indent=2))
    else:
        print(f"  checked {total_checked} input path(s) across {len(args.conf)} conf file(s)")
        if all_violations:
            print(f"\nFAIL: {len(all_violations)} unusable path(s)")
            for v in all_violations:
                print("  " + str(v))
        else:
            print("\nPASS: every input path resolves and exists")
    return 1 if all_violations else 0


if __name__ == "__main__":
    sys.exit(main())
