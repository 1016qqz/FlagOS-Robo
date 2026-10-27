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

"""Hydra entry point for FlagOS-Robo train / serve / inference jobs.

Provenance: this module is a *subset rewrite* of
``flagscale/run.py`` (211 lines, FlagScale baseline f8e9a62c), not a verbatim
copy.  Three deliberate differences from upstream, recorded here so a reader
can reconcile the two files:

1. ``VALID_TASKS`` is narrowed to ``{"train", "serve", "inference"}`` (upstream
   also has ``compress`` and ``rl``) and every ``TASK_ACTIONS`` entry drops
   ``auto_tune``.  Reason: ``autotuner_factory`` and the compress/RL runners
   are outside the FlagOS-Robo migration scope, so importing them here would
   fail at module load.  ``handle_auto_tune()`` is removed for the same reason.
2. The ``FLAGSCALE_USE_V1`` switch and the V1 code path are removed.  Upstream
   defaults the switch to ``"1"``, which makes ``get_runner()`` return
   ``flagscale.runner.runner_base.Runner``; that module is not part of this
   repository.  FlagOS-Robo always uses the legacy SSH/cloud runners.
3. ``get_runner()`` offers a cloud runner for ``train`` only.  Upstream also
   has ``CloudServeRunner``; that class was not migrated (it carries the
   PD-disaggregation / Ray code paths that FlagOS-Robo does not use), so a
   ``serve`` task with ``runner.type=cloud`` raises ``NotImplementedError``.
4. ``TASK_ACTIONS["serve"]`` is ``{"run", "dryrun", "test", "stop"}``: upstream
   omits ``dryrun`` but :class:`SSHServeRunner.run` accepts ``dryrun=True``, so
   it is exposed here.  ``query`` is omitted for ``serve`` and ``inference``
   because only the train runner implements a public ``query()``.

Everything else -- cwd restoration, config-error wrapping, the
``experiment.deploy`` compatibility move, the monitoring flags read on
``run``, and ``--config-path`` normalisation -- follows upstream behaviour.
"""

import os
import sys
import warnings

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from flagos_robo.logger import logger
from flagos_robo.runner.runner_inference import SSHInferenceRunner
from flagos_robo.runner.runner_serve import SSHServeRunner
from flagos_robo.runner.runner_train import CloudTrainRunner, SSHTrainRunner

VALID_TASKS = {"train", "serve", "inference"}

# task_type -> SSH runner class
SSH_RUNNER_MAP = {
    "train": SSHTrainRunner,
    "serve": SSHServeRunner,
    "inference": SSHInferenceRunner,
}

# task_type -> allowed actions.  Only the train runner implements a public
# query(); SSHServeRunner has _query_status() and SSHInferenceRunner has no
# status query at all, so "query" is train-only.
TASK_ACTIONS = {
    "train": {"run", "dryrun", "test", "stop", "query"},
    "serve": {"run", "dryrun", "test", "stop"},
    "inference": {"run", "dryrun", "test", "stop"},
}


def check_and_reset_deploy_config(config: DictConfig) -> None:
    if config.experiment.get("deploy", {}):
        OmegaConf.set_struct(config.experiment.runner, False)
        config.experiment.runner.deploy = config.experiment.deploy
        del config.experiment.deploy
        warnings.warn(
            "'config.experiment.deploy' has been moved to 'config.experiment.runner.deploy'. "
            "Support for the old location will be removed in a future release."
        )
        OmegaConf.set_struct(config.experiment.runner, True)


def validate_task(task_type: str, action: str) -> None:
    if task_type not in VALID_TASKS:
        raise ValueError(f"Invalid task_type '{task_type}', must be one of {sorted(VALID_TASKS)}")

    allowed_actions = TASK_ACTIONS[task_type]
    if action not in allowed_actions:
        raise ValueError(
            f"Action '{action}' is not allowed for task_type '{task_type}'. "
            f"Allowed actions: {sorted(allowed_actions)}"
        )


def get_runner(config: DictConfig, task_type: str):
    runner_type = config.experiment.runner.get("type", "ssh")

    if runner_type == "cloud":
        if task_type == "train":
            return CloudTrainRunner(config)
        raise NotImplementedError(
            f"Task type '{task_type}' is not supported by the cloud runner; use 'ssh'"
        )
    if runner_type == "ssh":
        return SSH_RUNNER_MAP[task_type](config)
    raise NotImplementedError(
        f"Runner type '{runner_type}' is not supported by FlagOS-Robo; use 'ssh' or 'cloud'"
    )


def execute_action(runner, action: str, task_type: str, config: DictConfig) -> None:
    if action == "run":
        if task_type == "train":
            # The monitoring flags are train-only: SSHServeRunner.run and
            # SSHInferenceRunner.run take (background, dryrun) and would raise
            # TypeError on these keywords.
            enable_monitoring = config.experiment.runner.get("enable_monitoring", False)
            enable_gpu_health_check = config.experiment.runner.get("enable_gpu_health_check", False)
            runner.run(
                enable_monitoring=enable_monitoring, enable_gpu_health_check=enable_gpu_health_check
            )
            if enable_monitoring:
                logger.info("Monitor service will be started automatically when training begins.")
        else:
            runner.run()
    elif action == "dryrun":
        runner.run(dryrun=True)
    elif action == "test":
        # Serve tasks are long-running daemons validated by an external client,
        # so they must start in the background even under `test`.  Train and
        # inference run in the foreground so output streams directly.
        if task_type == "serve":
            runner.run()
        else:
            runner.run(background=False)
    elif action == "stop":
        runner.stop()
    elif action == "query":
        runner.query()
    else:
        raise ValueError(f"Unknown action '{action}'")


@hydra.main(version_base=None, config_name="config")
def main(config: DictConfig) -> None:
    # Restore the user's original working directory.
    # Hydra defaults to hydra.job.chdir=True, which changes cwd to a run-specific
    # output dir (e.g. outputs/2024-01-01/12-00-00/).  This breaks all relative
    # paths in configs (tokenizer files, data paths, hostfiles, entrypoints, etc.).
    # FlagOS-Robo manages output directories explicitly (exp_dir, log_dir, etc.),
    # so the cwd change provides no benefit.  Restoring cwd here is equivalent to
    # hydra.job.chdir=False but works for all invocation paths (CLI, direct python,
    # tests) without requiring Hydra config file changes.
    os.chdir(hydra.utils.get_original_cwd())

    try:
        _main(config)
    except OmegaConfBaseException as e:
        logger.error(f"Config validation error: {e}")
        try:
            config_yaml = OmegaConf.to_yaml(config, resolve=False)
        except Exception:
            config_yaml = repr(config)
        logger.error(f"Config content:\n{config_yaml}")
        raise SystemExit(
            f"Config error: {e}\nHint: Check your YAML config files for missing or invalid fields."
        ) from e


def _main(config: DictConfig) -> None:
    """Core logic, separated from main() for error wrapping."""
    check_and_reset_deploy_config(config)

    task_type = config.experiment.task.get("type", None)
    action = config.action
    validate_task(task_type, action)

    runner = get_runner(config, task_type)
    execute_action(runner, action, task_type, config)


def resolve_config_path_in_argv():
    """Convert relative --config-path in sys.argv to an absolute path.

    Hydra interprets a relative config_path as relative to the calling
    module's package directory, not the current working directory.
    Converting to an absolute path ensures it is treated as a filesystem path.
    """
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--config-path="):
            path = arg[len("--config-path=") :]
            if not os.path.isabs(path):
                sys.argv[i] = f"--config-path={os.path.abspath(path)}"
        elif arg == "--config-path" and i + 1 < len(sys.argv):
            path = sys.argv[i + 1]
            if not os.path.isabs(path):
                sys.argv[i + 1] = os.path.abspath(path)


if __name__ == "__main__":
    resolve_config_path_in_argv()
    main()
