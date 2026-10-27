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

"""Task-type dispatch in :mod:`flagos_robo.run`.

Covers the W5-B4 change that opened ``serve`` and ``inference`` alongside
``train``: the action tables, runner selection per task type, and the
per-task keyword split in ``execute_action`` (the serve/inference runners
take only ``background``/``dryrun``, so the train-only monitoring keywords
must never reach them).
"""

import inspect

import pytest
from omegaconf import OmegaConf

from flagos_robo.run import (
    SSH_RUNNER_MAP,
    TASK_ACTIONS,
    VALID_TASKS,
    check_and_reset_deploy_config,
    execute_action,
    get_runner,
    validate_task,
)
from flagos_robo.runner.runner_inference import SSHInferenceRunner
from flagos_robo.runner.runner_serve import SSHServeRunner
from flagos_robo.runner.runner_train import CloudTrainRunner, SSHTrainRunner

TASKS = {"train", "serve", "inference"}


def make_config(runner_type="ssh", **runner_extra):
    return OmegaConf.create(
        {"experiment": {"task": {"type": "train"}, "runner": {"type": runner_type, **runner_extra}}}
    )


class TestTaskTables:
    def test_valid_tasks(self):
        assert VALID_TASKS == TASKS

    def test_task_actions_cover_valid_tasks(self):
        assert set(TASK_ACTIONS) == VALID_TASKS

    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_base_actions_present(self, task):
        assert {"run", "dryrun", "test", "stop"} <= TASK_ACTIONS[task]

    def test_query_is_train_only(self):
        # Only the train runner exposes a public query(); SSHServeRunner has
        # _query_status() and SSHInferenceRunner has none.
        assert "query" in TASK_ACTIONS["train"]
        assert not hasattr(SSHServeRunner, "query")
        assert not hasattr(SSHInferenceRunner, "query")
        for task in ("serve", "inference"):
            assert "query" not in TASK_ACTIONS[task]

    def test_auto_tune_not_exposed(self):
        # autotuner_factory was not migrated.
        for actions in TASK_ACTIONS.values():
            assert "auto_tune" not in actions

    def test_runner_map(self):
        assert SSH_RUNNER_MAP == {
            "train": SSHTrainRunner,
            "serve": SSHServeRunner,
            "inference": SSHInferenceRunner,
        }


class TestValidateTask:
    @pytest.mark.parametrize(
        "task,action",
        [
            ("train", "run"),
            ("train", "query"),
            ("serve", "run"),
            ("serve", "dryrun"),
            ("serve", "stop"),
            ("inference", "test"),
        ],
    )
    def test_accepted(self, task, action):
        validate_task(task, action)

    @pytest.mark.parametrize("task", ["rl", "compress", "", None])
    def test_unknown_task_rejected(self, task):
        with pytest.raises(ValueError, match="Invalid task_type"):
            validate_task(task, "run")

    @pytest.mark.parametrize(
        "task,action", [("serve", "query"), ("inference", "query"), ("serve", "auto_tune")]
    )
    def test_disallowed_action_rejected(self, task, action):
        with pytest.raises(ValueError, match="is not allowed for task_type"):
            validate_task(task, action)


class TestGetRunner:
    @pytest.mark.parametrize(
        "task,cls",
        [
            ("train", SSHTrainRunner),
            ("serve", SSHServeRunner),
            ("inference", SSHInferenceRunner),
        ],
    )
    def test_ssh_dispatches_on_task_type(self, task, cls, monkeypatch):
        seen = {}

        def fake_init(self, config, **kwargs):
            seen["cls"] = type(self)

        monkeypatch.setattr(cls, "__init__", fake_init)
        config = make_config("ssh")
        config.experiment.task.type = task
        get_runner(config, task)
        assert seen["cls"] is cls

    def test_cloud_train_supported(self, monkeypatch):
        monkeypatch.setattr(CloudTrainRunner, "__init__", lambda self, config: None)
        runner = get_runner(make_config("cloud"), "train")
        assert isinstance(runner, CloudTrainRunner)

    @pytest.mark.parametrize("task", ["serve", "inference"])
    def test_cloud_rejects_non_train(self, task):
        # CloudServeRunner was not migrated.
        with pytest.raises(NotImplementedError, match="cloud runner"):
            get_runner(make_config("cloud"), task)

    def test_unknown_runner_type(self):
        with pytest.raises(NotImplementedError, match="is not supported by FlagOS-Robo"):
            get_runner(make_config("slurm"), "train")

    def test_default_runner_type_is_ssh(self, monkeypatch):
        monkeypatch.setattr(SSHTrainRunner, "__init__", lambda self, config, **kw: None)
        config = OmegaConf.create({"experiment": {"task": {"type": "train"}, "runner": {}}})
        assert isinstance(get_runner(config, "train"), SSHTrainRunner)


class RecordingRunner:
    """Stands in for a runner; records how execute_action called it."""

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(("run", kwargs))

    def stop(self):
        self.calls.append(("stop", {}))

    def query(self):
        self.calls.append(("query", {}))


class TestExecuteAction:
    def _config(self, **runner_extra):
        return make_config("ssh", **runner_extra)

    def test_train_run_passes_monitoring_flags(self):
        runner = RecordingRunner()
        config = self._config(enable_monitoring=True, enable_gpu_health_check=True)
        execute_action(runner, "run", "train", config)
        assert runner.calls == [
            ("run", {"enable_monitoring": True, "enable_gpu_health_check": True})
        ]

    def test_train_run_defaults_monitoring_off(self):
        runner = RecordingRunner()
        execute_action(runner, "run", "train", self._config())
        assert runner.calls == [
            ("run", {"enable_monitoring": False, "enable_gpu_health_check": False})
        ]

    @pytest.mark.parametrize("task", ["serve", "inference"])
    def test_non_train_run_gets_no_monitoring_flags(self, task):
        # The real serve/inference runners would raise TypeError on these.
        runner = RecordingRunner()
        config = self._config(enable_monitoring=True, enable_gpu_health_check=True)
        execute_action(runner, "run", task, config)
        assert runner.calls == [("run", {})]

    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_dryrun(self, task):
        runner = RecordingRunner()
        execute_action(runner, "dryrun", task, self._config())
        assert runner.calls == [("run", {"dryrun": True})]

    def test_test_action_serve_runs_in_background(self):
        runner = RecordingRunner()
        execute_action(runner, "test", "serve", self._config())
        assert runner.calls == [("run", {})]

    @pytest.mark.parametrize("task", ["train", "inference"])
    def test_test_action_foreground_for_others(self, task):
        runner = RecordingRunner()
        execute_action(runner, "test", task, self._config())
        assert runner.calls == [("run", {"background": False})]

    @pytest.mark.parametrize("task", sorted(TASKS))
    def test_stop(self, task):
        runner = RecordingRunner()
        execute_action(runner, "stop", task, self._config())
        assert runner.calls == [("stop", {})]

    def test_unknown_action(self):
        with pytest.raises(ValueError, match="Unknown action"):
            execute_action(RecordingRunner(), "frobnicate", "train", self._config())


class TestRunnerSignatures:
    """Guards the assumption behind the per-task keyword split."""

    @pytest.mark.parametrize("cls", [SSHServeRunner, SSHInferenceRunner])
    def test_serve_inference_run_takes_only_background_dryrun(self, cls):
        params = inspect.signature(cls.run).parameters
        assert set(params) - {"self"} == {"background", "dryrun"}
        assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
            f"{cls.__name__}.run grew **kwargs; the keyword split in execute_action can be relaxed"
        )

    def test_train_run_accepts_monitoring_keywords(self):
        params = inspect.signature(SSHTrainRunner.run).parameters
        assert "enable_monitoring" in params
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


class TestDeployCompat:
    def test_deploy_moved_under_runner(self):
        config = OmegaConf.create(
            {"experiment": {"runner": {"type": "ssh"}, "deploy": {"use_fs_serve": False}}}
        )
        with pytest.warns(UserWarning, match="has been moved"):
            check_and_reset_deploy_config(config)
        assert "deploy" not in config.experiment
        assert config.experiment.runner.deploy == {"use_fs_serve": False}

    def test_no_deploy_is_noop(self):
        config = make_config("ssh")
        check_and_reset_deploy_config(config)
        assert "deploy" not in config.experiment
