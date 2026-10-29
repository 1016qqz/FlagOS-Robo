# docs/

Two kinds of document live here. Read the first group if you are trying to use
the repository; the second group is development history.

## User documentation

| Document | What it covers |
|---|---|
| [GETTING_STARTED.md](./GETTING_STARTED.md) | Install the launcher, check the install, the four `flagos-robo` sub-commands, config lookup, action flags, common failures |
| [GETTING_STARTED_cn.md](./GETTING_STARTED_cn.md) | Same content in Chinese |

Per-model guides (environment, model download, dataset layout, exact commands)
are not here — they live next to each model:
[`examples/pi0`](../examples/pi0/README.md),
[`examples/pi0_5`](../examples/pi0_5/README.md),
[`examples/gr00t_n1_5`](../examples/gr00t_n1_5/README.md).

Release contents are in [CHANGELOG.md](../CHANGELOG.md); how to contribute is
in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Development records

These describe how this repository was split out of
[FlagScale](https://github.com/flagos-ai/FlagScale) and how each stage was
checked. They are records of work already done, written in Chinese, and are not
instructions for users. Line counts are as of the 0.1.0 release.

| Document | Lines | What it records |
|---|---|---|
| [MIGRATION_MAP.md](./MIGRATION_MAP.md) | 155 | Which FlagScale module went where. Settled in week 1 and not changed afterwards |
| [migration_scope.txt](./migration_scope.txt) | 125 | The planned module prefix list, read by `tools/check_import_closure.py --planned-from` |
| [MIGRATION_SOP.md](./MIGRATION_SOP.md) | 334 | The per-batch procedure used while moving code in |
| [PRECISION_BASELINE.md](./PRECISION_BASELINE.md) | 270 | How each migrated batch was checked against FlagScale numerically |
| [W1_ACCEPTANCE.md](./W1_ACCEPTANCE.md) | 207 | Week-1 acceptance: the six command groups, run for real, with their output |
| [W2W3_ACCEPTANCE.md](./W2W3_ACCEPTANCE.md) | 352 | Week-2/3 acceptance: the eight command groups A1–A8, run for real |
| [REVIEW_W1_W3.md](./REVIEW_W1_W3.md) | 424 | Code review of the week-1 to week-3 state |
| [REVIEW_W1_W4.md](./REVIEW_W1_W4.md) | 339 | What that review actually changed, and what was deliberately deferred |

## Checks you can run yourself

The scripts behind those acceptance reports are in [`tools/`](../tools) and are
still runnable:

```sh
bash tools/run_gates.sh          # the repository's own check set
python3 -m pytest tests/unit_tests -q
pre-commit run --all-files       # uses the ruff version pinned in .pre-commit-config.yaml
```
