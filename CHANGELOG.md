# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the version
numbers follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0]

First release of FlagOS-Robo as a standalone repository. The embodied-AI (VLA)
training, inference and serving code previously maintained inside
[FlagScale](https://github.com/flagos-ai/FlagScale) now lives here and is
driven by its own `flagos-robo` command line.

### Added

- `flagos-robo` command line with four sub-commands — `run`, `train`,
  `serve`, `inference` — plus `--version` and `--help`
  (`flagos_robo/cli.py`). Model-name invocations resolve their config from
  `<cwd>/examples/<model>/conf/<task>.yaml`; `run` takes an explicit
  `--config-path` / `--config-name` pair and forwards trailing `key=value`
  pairs to Hydra.
- Action flags shared by `train` / `serve` / `inference`: `--dryrun`,
  `--test`, `--stop`, and `--query` on `train`. They are mutually exclusive
  and the CLI rejects combinations instead of picking one.
- Hardware platform abstraction layer (`flagos_robo/platforms`) so that device,
  communication backend and RNG handling are selected per accelerator rather
  than hard-coded to CUDA.
- Task runners for training, offline inference and serving, including
  mid-epoch resume (`flagos_robo/runner`).
- VLA model registry and the shared model utilities the three examples build
  on (`flagos_robo/models/vla`).
- Worked examples, each with its own guide covering environment, model
  download, dataset layout and commands:
  [PI0](./examples/pi0/README.md),
  [PI0.5](./examples/pi0_5/README.md),
  [GR00T-N1.5](./examples/gr00t_n1_5/README.md).
- Data loading for webdataset, Megatron-Energon and lerobot dataset formats.
- Repository check set: `tools/run_gates.sh` plus the individual scripts it
  calls (import-reachability, config-path, dependency-drift, no-megatron,
  symbol-reference, test-coverage, precision comparison).
- Unit test suite under `tests/unit_tests` covering the platform layer, the
  runners, the transformation hooks, the VLA registry and the check scripts.
- Continuous integration workflow running the CPU-only subset of the checks on
  every push and pull request.
- User documentation: [Getting Started](./docs/GETTING_STARTED.md) and its
  [Chinese version](./docs/GETTING_STARTED_cn.md), a [docs index](./docs/README.md),
  and [CONTRIBUTING.md](./CONTRIBUTING.md).

### Changed

- The root `README.md` / `README_cn.md` model table now lists only the models
  this repository ships (PI0, PI0.5, GR00T-N1.5) and links to the guides inside
  this repository instead of to FlagScale.

### Not included

- `flagos-robo eval`. Evaluation runs on the
  [FlagEval-Robo](https://embodiedverse.flageval.net/home) platform; there is
  no eval sub-command in this release.
- Reinforcement learning and model-compression runners, and the
  `install` / `pull` / `test` tooling from FlagScale's CLI. They are outside
  the scope of this split.
- Automatic parallelism tuning (`--tune`): the auto-tuner it depends on was not
  moved into this repository.
- `serve --port` / `--model-path` / `--engine-args`. The serving runner does
  not read command-line engine arguments; serving options belong in the config
  YAML. The flags are omitted rather than accepted and ignored.
- RoboBrain-2.0 / 2.5, RoboBrain-X0, Qwen-GR00T and KERV. These remain in
  [FlagScale](https://github.com/flagos-ai/FlagScale).

[Unreleased]: https://github.com/flagos-ai/FlagOS-Robo/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/flagos-ai/FlagOS-Robo/releases/tag/v0.1.0
