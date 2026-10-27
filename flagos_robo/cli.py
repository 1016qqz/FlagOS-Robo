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

"""``flagos-robo`` command line entry point.

Provenance: a *subset rewrite* of ``flagscale/cli.py`` (528 lines, FlagScale
baseline f8e9a62c), not a verbatim copy.  Upstream exposes nine sub-commands
(run / train / serve / inference / rl / compress / install / test / pull) plus
an ``eval`` sub-app.  FlagOS-Robo keeps ``run``, ``train``, ``serve`` and
``inference``; the RL and compress runners plus the install/pull/test/eval
tooling are outside this repository's migration scope.

The ``--tune`` flag is dropped: ``auto_tune`` needs
``flagscale.runner.autotuner_factory``, which was not migrated.  Upstream's
``serve`` also takes ``--port`` / ``--model-path`` / ``--engine-args``, which
are forwarded as ``experiment.runner.cli_args`` overrides; FlagOS-Robo's
serve runner does not read ``cli_args`` (the inference-engine argument
plumbing was not migrated), so those flags are omitted rather than silently
ignored.  Serve options belong in the config YAML.

Helper shapes (``resolve_config``, ``_handle_config_error``, ``run_task``,
``get_action``) follow upstream so that behaviour -- including the
``<cwd>/examples/<model>/conf/<task>.yaml`` lookup and the argv rewrite that
bridges typer to Hydra -- stays recognisable.
"""

import sys

from enum import Enum
from pathlib import Path
from typing import List, Optional

import typer

from flagos_robo import __version__ as FLAGOS_ROBO_VERSION

app = typer.Typer(
    name="flagos-robo",
    help="FlagOS-Robo CLI - embodied-AI (VLA) training toolkit.",
    add_completion=False,
)


# ============================================================================
# Helper Functions
# ============================================================================


def resolve_config(
    model_name: str, yaml_path: Optional[Path], task: str = "train"
) -> "tuple[str, str]":
    """Resolve config path and name from ``model_name`` or ``yaml_path``.

    When no explicit ``yaml_path`` is given, the config is looked up at
    ``<cwd>/examples/<model>/conf/<task>.yaml``.  The user is expected to run
    CLI commands from the repo root -- ``examples/`` is not part of the
    installed package.
    """
    if yaml_path:
        yaml_path = yaml_path.resolve()
        if not yaml_path.exists():
            typer.echo(f"Error: {yaml_path} does not exist", err=True)
            raise typer.Exit(1)
        return str(yaml_path.parent), yaml_path.stem

    yaml_path = Path.cwd() / "examples" / model_name / "conf" / f"{task}.yaml"
    if not yaml_path.exists():
        typer.echo(f"Error: {yaml_path} does not exist", err=True)
        raise typer.Exit(1)
    return str(yaml_path.parent), yaml_path.stem


def _handle_config_error(error: Exception, config_path: str, config_name: str) -> None:
    """Format config errors with detailed context and exit."""
    config_file = f"{config_path}/{config_name}.yaml"
    typer.secho(f"Config error in: {config_file}", fg="red", err=True)
    typer.secho(f"Error: {error}", fg="red", err=True)
    typer.secho(
        "Hint: Check that the config file exists, all defaults are resolvable, "
        "and required fields are present.",
        fg="yellow",
        err=True,
    )
    raise typer.Exit(1)


def run_task(
    config_path: str, config_name: str, action: str, extra_args: Optional[list] = None
) -> None:
    """Execute a task via :mod:`flagos_robo.run`.

    The task type is determined from the config at runtime by Hydra, not
    passed explicitly here.
    """
    from flagos_robo.run import main as run_main

    args = [
        "run.py",
        f"--config-path={config_path}",
        f"--config-name={config_name}",
        f"action={action}",
    ]
    if extra_args:
        args.extend(extra_args)
    sys.argv = args
    try:
        run_main()
    except SystemExit as e:
        if e.code and e.code != 0:
            raise
    except Exception as e:
        _handle_config_error(e, config_path, config_name)


def get_action(stop: bool, dryrun: bool, test: bool, query: bool) -> str:
    """Determine action from flags (mutually exclusive)."""
    flags = [("stop", stop), ("dryrun", dryrun), ("test", test), ("query", query)]
    set_flags = [name for name, value in flags if value]

    if len(set_flags) > 1:
        typer.echo(f"Error: Flags are mutually exclusive: --{', --'.join(set_flags)}", err=True)
        raise typer.Exit(1)

    if set_flags:
        return set_flags[0]
    return "run"  # default


# ============================================================================
# Run Command: flagos-robo run --config-path <path> --config-name <name> [options]
# This replaces the old `python run.py` interface
# ============================================================================


class Action(str, Enum):
    run = "run"
    dryrun = "dryrun"
    test = "test"
    stop = "stop"
    query = "query"


@app.command("run")
def run_cmd(
    config_path: Path = typer.Option(..., "--config-path", "-p", help="Path to config directory"),
    config_name: str = typer.Option(
        ..., "--config-name", "-n", help="Config file name (without .yaml)"
    ),
    action: Action = typer.Option(Action.run, "--action", "-a", help="Action to perform"),
    overrides: Optional[List[str]] = typer.Argument(
        None, help="Additional Hydra overrides (e.g., key=value)"
    ),
):
    """Run a task with an explicit config path and name (replaces python run.py).

    Example:
        flagos-robo run --config-path ./examples/pi0/conf --config-name train --action run
        flagos-robo run -p ./examples/pi0/conf -n train -a stop
    """
    config_path = config_path.resolve()
    if not config_path.exists():
        typer.echo(f"Error: Config path does not exist: {config_path}", err=True)
        raise typer.Exit(1)

    config_file = config_path / f"{config_name}.yaml"
    if not config_file.exists():
        typer.echo(f"Error: Config file does not exist: {config_file}", err=True)
        raise typer.Exit(1)

    typer.echo(
        f"Running: flagos-robo --config-path={config_path} "
        f"--config-name={config_name} action={action.value}"
    )
    run_task(
        str(config_path),
        config_name,
        action.value,
        extra_args=list(overrides) if overrides else None,
    )


# ============================================================================
# Task Command: flagos-robo train [--flags] <model>
# ============================================================================


@app.command()
def train(
    model: str = typer.Argument(..., help="Model name (e.g., pi0, pi0_5, gr00t_n1_5)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop training"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Validate config only"),
    test: bool = typer.Option(False, "--test", help="Run with test"),
    query: bool = typer.Option(False, "--query", help="Query status"),
):
    """Train a model."""
    action = get_action(stop, dryrun, test, query)
    cfg_path, cfg_name = resolve_config(model, config, "train")
    typer.echo(f"Train {model} [{action}]")
    typer.echo(f"config_path: {cfg_path}")
    typer.echo(f"config_name: {cfg_name}")
    run_task(cfg_path, cfg_name, action)


@app.command()
def serve(
    model: str = typer.Argument(..., help="Model name (e.g., pi0, pi0_5, gr00t_n1_5)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop serving"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Validate config only"),
    test: bool = typer.Option(False, "--test", help="Test serving"),
):
    """Serve a policy over the websocket protocol."""
    action = get_action(stop, dryrun, test, False)
    cfg_path, cfg_name = resolve_config(model, config, "serve")
    typer.echo(f"Serve {model} [{action}]")
    typer.echo(f"config_path: {cfg_path}")
    typer.echo(f"config_name: {cfg_name}")
    run_task(cfg_path, cfg_name, action)


@app.command()
def inference(
    model: str = typer.Argument(..., help="Model name (e.g., pi0, pi0_5)"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Config YAML path"),
    stop: bool = typer.Option(False, "--stop", help="Stop inference"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Validate config only"),
    test: bool = typer.Option(False, "--test", help="Run with test"),
):
    """Run offline inference against a trained checkpoint."""
    action = get_action(stop, dryrun, test, False)
    cfg_path, cfg_name = resolve_config(model, config, "inference")
    typer.echo(f"Inference {model} [{action}]")
    typer.echo(f"config_path: {cfg_path}")
    typer.echo(f"config_name: {cfg_name}")
    run_task(cfg_path, cfg_name, action)


# ============================================================================
# Version
# ============================================================================


def version_callback(value: bool):
    if value:
        typer.echo(f"flagos-robo version {FLAGOS_ROBO_VERSION}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True, help="Show version"
    ),
):
    """FlagOS-Robo CLI - embodied-AI (VLA) training toolkit."""
    pass


# Entry point
def flagos_robo():
    app()


if __name__ == "__main__":
    flagos_robo()
