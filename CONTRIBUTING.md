# Contributing to FlagOS-Robo

Thanks for taking the time. This page covers the mechanics: how to set up a
development environment, which checks must pass before a pull request, and the
conventions this repository already follows.

## Development setup

```sh
git clone https://github.com/flagos-ai/FlagOS-Robo.git
cd FlagOS-Robo
pip install -e ".[dev]"
pre-commit install
```

The `dev` extra brings `pytest`, `pre-commit` and `ruff`. Python 3.10 or newer
is required.

Model runtimes (PyTorch, model repositories, datasets) are deliberately not
dependencies of this package. Install them per the guide of the model you are
working on, under [`examples/`](./examples).

## Checks that must pass

Run these three before opening a pull request. They are the same checks CI
runs, so a local pass is the cheapest way to avoid a red build.

```sh
bash tools/run_gates.sh
python3 -m pytest tests/unit_tests -q
pre-commit run --all-files
```

`tools/run_gates.sh` runs the repository's own check scripts. Each one has a
`--self-test` mode that deliberately feeds it a broken input to confirm the
check can actually fail — the script runs those self-tests first, then the real
checks:

| Script | What it checks |
|---|---|
| `check_no_megatron.py` | `megatron_core` has not entered the install-time dependency tree |
| `check_all_dangling.py` | no references to files or modules that do not exist |
| `check_symbol_refs.py` | imported and referenced symbols actually exist at their source |
| `check_dep_drift.py` | declared dependencies have not drifted from what the code imports |
| `check_import_closure.py` | every reachable module is inside the planned module list in `docs/migration_scope.txt` |
| `check_test_coverage.py` | per-module test coverage has not dropped below `tools/test_coverage_baseline.json` |
| `check_conf_paths.py` | paths inside `examples/*/conf` resolve |
| `check_leaf_import.py` | leaf modules stay importable without heavy optional packages |
| `compare_precision.py` | the precision comparison used for migrated code behaves as specified |

## Code style

Formatting and linting are handled by `ruff`, pinned in
[`.pre-commit-config.yaml`](./.pre-commit-config.yaml). Use `pre-commit` rather
than a `ruff` you happen to have on your PATH: a different `ruff` version
reports different findings, and the pinned one is what CI uses.

Line length is 100 (`pyproject.toml`, `[tool.ruff]`).

Every source file carries the Apache-2.0 header used by the existing files.

Where code came from FlagScale, say so in the module docstring and name what
changed — the existing modules do this (see the docstring of
`flagos_robo/cli.py`, which records which upstream sub-commands were kept,
which were dropped, and why). A reader should not have to diff two
repositories to find out whether a file is a copy or a rewrite.

## Tests

New behaviour needs a test under `tests/unit_tests`, mirroring the package
layout (`flagos_robo/runner/...` → `tests/unit_tests/runner/...`). A bug fix
needs a test that fails before the fix. `check_test_coverage.py` enforces the
baseline, so a module added without tests fails the check set.

Tests in `tests/unit_tests` must run on CPU with no accelerator present — that
is the subset CI executes. A test that needs a GPU belongs behind an explicit
skip condition.

## Pull requests

- One topic per pull request.
- The description says what changed, why, and how you verified it — paste the
  output of the checks you ran rather than asserting they pass.
- If something is deliberately left undone, write it down in the description.
  A stated gap is reviewable; a silent one is found later by someone else.
- Note any change in user-visible behaviour in [CHANGELOG.md](./CHANGELOG.md)
  under `## [Unreleased]`.

## Reporting a problem

Open an issue with: what you ran (the full command), what you expected, what
happened (the full error), and your environment — Python version, accelerator,
and the output of `flagos-robo --version`.

## License

Contributions are accepted under the [Apache-2.0](./LICENSE) license of this
repository.
