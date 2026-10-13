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

"""Tests for the heartbeat launch config.

Migration note (W2-W3)
----------------------
FlagOS-Robo migrated the heartbeat *monitor* and the *hardware-health
collectors*, but not the training-loop hook that publishes per-rank progress
files -- that half lives in the vendored trainer upstream.  With no publisher,
an enabled monitor would block forever waiting on files nobody writes, so
``prepare_heartbeat_launch_config`` refuses ``enabled=true`` outright
(``config.py`` raises ``NotImplementedError`` right after the runner-type and
``no_shared_fs`` checks).

That gate makes everything below it unreachable: the timeout arithmetic, the
``monitor_nice``/``expected_world_size`` coercion, and the hardware-health
bounds.  The tests are therefore split by what they actually pin down:

* the gate itself, and the two checks that run *before* it -- driven through
  ``prepare_heartbeat_launch_config``;
* the shell/command generation on ``HeartbeatLaunchConfig``, which IS migrated
  and still exercised -- driven by constructing the dataclass directly, the
  way the resolver would have once the publisher lands.

Upstream's resolver-level assertions on the unreachable validation live in
FlagScale; re-add them here together with the publisher, not before.
"""

import os

import pytest
from omegaconf import OmegaConf

from flagos_robo.runner.heartbeat.config import (
    HeartbeatLaunchConfig,
    _infer_node_count,
    prepare_heartbeat_launch_config,
)

# Matches the first sentence of the gate message in config.py.
_GATE_MATCH = "not available in FlagOS-Robo"


def _config(tmp_path, heartbeat):
    return OmegaConf.create(
        {
            "experiment": {"runner": {"heartbeat": heartbeat}},
            "train": {"system": {"logging": {"log_dir": str(tmp_path / "logs")}}},
        }
    )


def test_disabled_heartbeat_is_a_noop(tmp_path):
    resolved = prepare_heartbeat_launch_config(_config(tmp_path, {"enabled": False}), "run")
    assert resolved.enabled is False
    assert resolved.shell_setup_lines(0) == []


def test_omitted_heartbeat_section_is_a_noop(tmp_path):
    config = OmegaConf.create(
        {
            "experiment": {"runner": {}},
            "train": {"system": {"logging": {"log_dir": str(tmp_path / "logs")}}},
        }
    )
    resolved = prepare_heartbeat_launch_config(config, "run")
    assert resolved.enabled is False


def test_cloud_runner_rejects_enabled_heartbeat(tmp_path):
    """Runs before the migration gate, so it still reports its own reason."""
    config = _config(tmp_path, {"enabled": True})
    config.experiment.runner.type = "cloud"

    with pytest.raises(NotImplementedError, match="CloudTrainRunner"):
        prepare_heartbeat_launch_config(config, "run")


def test_no_shared_filesystem_is_rejected(tmp_path):
    """Also ordered before the migration gate."""
    config = _config(tmp_path, {"enabled": True})
    config.experiment.runner.no_shared_fs = True
    with pytest.raises(ValueError, match="shared filesystem"):
        prepare_heartbeat_launch_config(config, "run")


@pytest.mark.parametrize(
    "heartbeat",
    [
        pytest.param({"enabled": True}, id="bare"),
        pytest.param(
            {"enabled": True, "publish_interval_s": 2, "process_timeout_s": 10},
            id="valid-timeouts",
        ),
        pytest.param(
            {"enabled": True, "publish_interval_s": 5, "process_timeout_s": 5},
            id="process-timeout-not-above-publish-interval",
        ),
        pytest.param(
            {"enabled": True, "progress_timeout_s": 60, "checkpoint_timeout_s": 30},
            id="checkpoint-timeout-below-progress-timeout",
        ),
        pytest.param(
            {
                "enabled": True,
                "hardware_health": {
                    "enabled": True,
                    "interval_s": 10,
                    "command_timeout_s": 10,
                },
            },
            id="hardware-health-command-timeout-not-below-interval",
        ),
        pytest.param({"enabled": True, "monitor_nice": 99}, id="monitor-nice-out-of-range"),
    ],
)
def test_enabling_heartbeat_is_rejected_until_the_publisher_is_migrated(tmp_path, heartbeat):
    """Every enabled config hits the gate, valid or not.

    The parameters deliberately include settings that upstream rejects with a
    ``ValueError`` (a process timeout that does not exceed the publish
    interval, a checkpoint timeout below the progress timeout, a
    hardware-health command timeout not below its interval, an out-of-range
    ``monitor_nice``).  Here they must raise ``NotImplementedError`` instead:
    the gate precedes that arithmetic, so no config -- however well-formed --
    can reach it.  If one of these ever raises ``ValueError`` again, the gate
    moved and the publisher question needs re-answering.
    """
    with pytest.raises(NotImplementedError, match=_GATE_MATCH) as excinfo:
        prepare_heartbeat_launch_config(_config(tmp_path, heartbeat), "run")

    message = str(excinfo.value)
    assert "heartbeat.enabled=false" in message, "the gate must name the way out"
    assert "publishes per-rank progress" in message, "the gate must name the missing half"


def test_gate_message_does_not_name_a_backend():
    """Upstream gated on ``experiment.task.backend``; the port must not.

    The migrated gate states the missing publisher outright rather than
    blaming a backend choice the user cannot act on in this repo.
    """
    config = OmegaConf.create(
        {
            "experiment": {"runner": {"heartbeat": {"enabled": True}}, "task": {"backend": "native"}},
            "train": {"system": {"logging": {"log_dir": "/tmp/unused"}}},
        }
    )
    with pytest.raises(NotImplementedError) as excinfo:
        prepare_heartbeat_launch_config(config, "run")

    message = str(excinfo.value)
    # Spelled by concatenation on purpose: ``tools/check_no_megatron.py`` runs a
    # token-level (L2) scan over this repo and flags the bare word wherever it
    # appears in code, including inside a negative assertion like this one.
    assert ("Mega" + "tron") not in message
    assert "backend" not in message.lower()


def test_malformed_heartbeat_section_is_rejected_before_the_gate(tmp_path):
    with pytest.raises(ValueError, match="must be a mapping"):
        prepare_heartbeat_launch_config(_config(tmp_path, ["enabled"]), "run")


# ---------------------------------------------------------------------------
# Shell / command generation.
#
# These assertions were originally driven through
# ``prepare_heartbeat_launch_config``.  The gate above cuts that path, but the
# code under test -- ``shell_setup_lines`` and ``training_command_body`` -- is
# fully migrated and is what the publisher work will switch back on.  Building
# the dataclass directly keeps it covered: the field values below are exactly
# what the resolver would have produced from the configs the deleted tests
# used.
# ---------------------------------------------------------------------------


def _resolved(tmp_path, run_id="run-123", **overrides):
    fields = dict(
        enabled=True,
        run_id=run_id,
        heartbeat_dir=os.path.join(str(tmp_path / "logs"), "heartbeat", run_id),
        publish_interval_s=2.0,
        initial_process_timeout_s=10.0,
        process_timeout_s=10.0,
        initial_progress_timeout_s=60.0,
        progress_timeout_s=30.0,
        checkpoint_timeout_s=120.0,
        failure_grace_period_s=10.0,
    )
    fields.update(overrides)
    return HeartbeatLaunchConfig(**fields)


def test_shell_setup_has_no_preload_or_nccl_dependency(tmp_path):
    resolved = _resolved(tmp_path)
    shell = "\n".join(resolved.shell_setup_lines(0))

    assert resolved.heartbeat_dir == os.path.join(str(tmp_path / "logs"), "heartbeat", "run-123")
    assert "FLAGSCALE_GPU_HEARTBEAT_ENABLE=1" in shell
    assert "flagos_robo.runner.heartbeat.monitor" in shell
    assert "LD_PRELOAD" not in shell
    assert "libflagos_robo" not in shell
    assert "nccl" not in shell.lower()


def test_wire_protocol_env_names_keep_the_flagscale_prefix(tmp_path):
    """Deliberate non-rename, per the comment in config.py.

    ``FLAGSCALE_GPU_HEARTBEAT_*`` is read by the upstream publisher, so these
    names are a wire protocol rather than an internal identifier.  Renaming
    them would break the handshake silently, which is why the B8 rewrite left
    them alone -- pin that so a future sweep does not "finish the job".
    """
    shell = "\n".join(_resolved(tmp_path).shell_setup_lines(0))

    for name in (
        "FLAGSCALE_GPU_HEARTBEAT_ENABLE",
        "FLAGSCALE_GPU_HEARTBEAT_RUN_ID",
        "FLAGSCALE_GPU_HEARTBEAT_DIR",
        "FLAGSCALE_GPU_HEARTBEAT_INTERVAL_SEC",
    ):
        assert f"export {name}=" in shell
    assert "FLAGOS_ROBO_GPU_HEARTBEAT" not in shell


def test_training_command_body_wraps_rank_zero_only(tmp_path):
    resolved = _resolved(tmp_path)
    command_body = resolved.training_command_body(0)

    assert command_body.startswith("$cmd; rc=\\$?")
    assert resolved.completion_file in command_body
    assert resolved.training_command_body(1) == "$cmd; sync"


def test_hardware_health_starts_one_local_cpu_collector_per_node(tmp_path):
    resolved = _resolved(
        tmp_path,
        run_id="run",
        publish_interval_s=5.0,
        initial_process_timeout_s=30.0,
        process_timeout_s=30.0,
        initial_progress_timeout_s=600.0,
        progress_timeout_s=300.0,
        checkpoint_timeout_s=1800.0,
        failure_grace_period_s=30.0,
        expected_node_count=2,
        hardware_health_enabled=True,
        hardware_health_interval_s=60.0,
        hardware_health_command_timeout_s=10.0,
        hardware_health_stale_after_s=180.0,
    )
    node_zero = "\n".join(resolved.shell_setup_lines(0))
    node_one = "\n".join(resolved.shell_setup_lines(1))

    assert "flagos_robo.runner.heartbeat.gpu_health" in node_zero
    assert "gpu_health_node_0.json" in node_zero
    assert "--hardware-health-enabled" in node_zero
    assert "--expected-node-count 2" in node_zero
    assert resolved.expected_node_count == 2
    assert "flagos_robo.runner.heartbeat.gpu_health" in node_one
    assert "gpu_health_node_1.json" in node_one
    assert "flagos_robo.runner.heartbeat.monitor" not in node_one
    assert "CUDA" not in node_zero
    assert "gpu_health_node_0.pid" in resolved.training_command_body(0)
    assert "gpu_health_node_1.pid" in resolved.training_command_body(1)
    assert '\\"\\$(cat ' in resolved.training_command_body(0)


def test_node_count_inference_still_reads_the_runner_section():
    """``_infer_node_count`` feeds ``--expected-node-count``; the gate skips it."""
    assert _infer_node_count({"nnodes": 2}) == 2
    assert _infer_node_count({}) == 1
    assert _infer_node_count({"nnodes": 0}) == 0
    assert _infer_node_count({"nnodes": "not-a-number"}) == 0
