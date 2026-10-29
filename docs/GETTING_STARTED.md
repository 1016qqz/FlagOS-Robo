# Getting Started

This page covers the launcher only: how to install `flagos-robo`, how to check
that the install worked, and what the four sub-commands do. Running an actual
model also needs that model's runtime dependencies — those live in the
per-model guides under [`examples/`](../examples).

- [1. Requirements](#1-requirements)
- [2. Install the launcher](#2-install-the-launcher)
- [3. Check the install](#3-check-the-install)
- [4. The four sub-commands](#4-the-four-sub-commands)
- [5. How a config is found](#5-how-a-config-is-found)
- [6. Action flags](#6-action-flags)
- [7. Run a real model](#7-run-a-real-model)
- [8. When something fails](#8-when-something-fails)

## 1. Requirements

| Item | Value | Where it is declared |
|---|---|---|
| Python | >= 3.10 | `pyproject.toml` `requires-python` |
| Runtime dependencies | `hydra-core>=1.3.2`, `omegaconf>=2.3.0`, `typer>=0.9.0` | `pyproject.toml` `[project] dependencies` |
| Build backend | `setuptools.build_meta` | `pyproject.toml` `[build-system]` |

PyTorch is **not** a dependency of the launcher. Each model guide pins its own
PyTorch / model-repository / dataset requirements, because those differ per
model and per accelerator.

## 2. Install the launcher

```sh
git clone https://github.com/flagos-ai/FlagOS-Robo.git
cd FlagOS-Robo
pip install -e .
```

An editable install is the recommended mode for this release: the CLI resolves
example configs relative to your current directory (see
[section 5](#5-how-a-config-is-found)), so working from the repository root is
the normal workflow.

A non-editable install also works:

```sh
pip install .
```

In that case keep a clone of the repository around anyway — `examples/` is not
shipped inside the installed package.

## 3. Check the install

```sh
flagos-robo --version
flagos-robo --help
```

`--help` lists the sub-commands: `run`, `train`, `serve`, `inference`.

You can reach the same entry point without the console script:

```sh
python -c "import flagos_robo; print(flagos_robo.__version__)"
```

## 4. The four sub-commands

### `train <model>`

Train a model.

```sh
flagos-robo train pi0
flagos-robo train pi0 --config ./examples/pi0/conf/train.yaml
flagos-robo train pi0 --stop
```

### `inference <model>`

Run offline inference against a trained checkpoint.

```sh
flagos-robo inference pi0 --config ./examples/pi0/conf/inference.yaml
```

### `serve <model>`

Serve a policy over the websocket protocol.

```sh
flagos-robo serve pi0 --config ./examples/pi0/conf/serve.yaml
flagos-robo serve pi0 --stop
```

### `run --config-path <dir> --config-name <name>`

The low-level form. It takes a config directory and a config file name instead
of a model name, and forwards any trailing `key=value` pairs to Hydra as
overrides.

```sh
flagos-robo run --config-path ./examples/pi0/conf --config-name train --action run
flagos-robo run -p ./examples/pi0/conf -n train -a stop
```

`--action` accepts `run`, `dryrun`, `test`, `stop`, `query`.

## 5. How a config is found

`train` / `serve` / `inference` take a model name. When `--config` is not
given, the config is looked up at:

```
<current directory>/examples/<model>/conf/<task>.yaml
```

So `flagos-robo train pi0`, executed from the repository root, reads
`examples/pi0/conf/train.yaml`. Run the commands from the repository root, or
pass `--config` with an explicit path. The configs that exist in this release:

| Model | train | inference | serve |
|---|---|---|---|
| `pi0` | ✅ | ✅ | ✅ |
| `pi0_5` | ✅ | ✅ | ✅ |
| `gr00t_n1_5` | ✅ | ❌ | ✅ |

If the file is missing, the CLI prints `Error: <path> does not exist` and exits
with status 1 instead of guessing.

## 6. Action flags

`train` / `serve` / `inference` share a set of mutually exclusive flags. They
select the action; without any of them the action is `run`.

| Flag | Meaning | Available on |
|---|---|---|
| (none) | start the task | train, serve, inference |
| `--dryrun` | validate the config only, do not start the task | train, serve, inference |
| `--test` | run with the test path enabled | train, serve, inference |
| `--stop` | stop a running task | train, serve, inference |
| `--query` | report task status | train |

Passing two of them at once is rejected: the CLI prints
`Error: Flags are mutually exclusive: ...` and exits with status 1.

`--dryrun` is the cheapest way to check a config you just edited — it resolves
the Hydra config and reports errors without launching anything.

## 7. Run a real model

The launcher does not install model runtimes. Pick a model and follow its
guide end to end — each one covers its own conda environment, PyTorch version,
model download, dataset layout, and the exact commands:

- [`examples/pi0/README.md`](../examples/pi0/README.md) — training, offline inference, serving
- [`examples/pi0_5/README.md`](../examples/pi0_5/README.md) — training, offline inference, serving
- [`examples/gr00t_n1_5/README.md`](../examples/gr00t_n1_5/README.md) — training, serving

## 8. When something fails

| Symptom | Likely cause |
|---|---|
| `flagos-robo: command not found` | the install did not run, or the environment with the install is not active |
| `Error: <path>/examples/<model>/conf/<task>.yaml does not exist` | not in the repository root, or that task has no config for that model (see the table in [section 5](#5-how-a-config-is-found)) |
| `Config error in: <file>` followed by a Hydra message | a field in the config YAML is missing or unresolvable; the hint line names what to check |
| `Error: Flags are mutually exclusive` | two action flags passed at once (see [section 6](#6-action-flags)) |
| import errors naming torch, a model repository, or a dataset package | model runtime dependencies are not installed — follow that model's guide, not this page |

Evaluation is not part of this release. `flagos-robo eval` does not exist here;
the evaluation flow runs on the
[FlagEval-Robo](https://embodiedverse.flageval.net/home) platform.
