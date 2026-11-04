# Documentation

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

## Development references

- [Precision comparison guide](./PRECISION_BASELINE.md) explains the repository's numerical comparison tool.
- [Migration map](./MIGRATION_MAP.md) and [migration scope](./migration_scope.txt) record the FlagScale baseline used by repository checks.
- [Historical migration, acceptance, and review records](./archive/README.md) are retained for provenance. They describe earlier repository states, not current user instructions.

## Checks you can run yourself

The scripts behind those acceptance reports are in [`tools/`](../tools) and are
still runnable:

```sh
bash tools/run_gates.sh          # the repository's own check set
python3 -m pytest tests/unit_tests -q
pre-commit run --all-files       # uses the ruff version pinned in .pre-commit-config.yaml
```
