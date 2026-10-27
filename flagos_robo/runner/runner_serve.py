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

import contextlib
import os
import shlex
import signal
import subprocess

import psutil
from omegaconf import DictConfig, OmegaConf

from flagos_robo.runner.runner_base_legacy import JobStatus, RunnerBase
from flagos_robo.runner.utils import (
    get_free_port,
    get_node0_log_file,
    get_nproc_per_node,
    get_pkg_dir,
    logger,
    parse_hostfile,
    resolve_path,
    run_local_command,
    setup_exp_dir,
    setup_logging_dirs,
    start_tail_log,
)


def _get_args_serve(config: DictConfig):
    # see the following link for more details
    # https://github.com/facebookresearch/hydra/discussions/2750
    config_dict = OmegaConf.to_container(config, resolve=True)

    # step2: restructuring the config
    # config_dict = config_dict["serve"]
    config_dict["logging"].pop("log_dir")
    config_dict["logging"].pop("scripts_dir")
    config_dict["logging"].pop("pids_dir")
    if not config_dict.get("logging"):
        config_dict.pop("logging")

    # step3: dict -> yaml
    logging_config = config.logging
    new_config = OmegaConf.create(config_dict)
    new_conf_file = os.path.join(logging_config.scripts_dir, "serve.yaml")

    # step4: write the new yaml file to `outputs_dir/serve_logs/scripts/serve.yaml`
    with open(new_conf_file, "w") as f:
        OmegaConf.save(config=new_config, f=f.name, resolve=True)

    args = []
    args.append(f"--config-path={new_conf_file}")

    return args


def _update_config_serve(config: DictConfig):
    exp_dir = setup_exp_dir(config)

    OmegaConf.set_struct(config, False)

    if config.get("logging", None) is None:
        config.logging = DictConfig({})

    setup_logging_dirs(config.logging, exp_dir, log_subdir="serve_logs")

    os.makedirs(config.logging.scripts_dir, exist_ok=True)
    OmegaConf.set_struct(config, True)


def _generate_run_script_serve(config, host, node_rank, cmd, background=False):
    logging_config = config.logging

    no_shared_fs = config.experiment.runner.get("no_shared_fs", False)
    if no_shared_fs:
        host_output_file = os.path.join(logging_config.log_dir, "host.output")
    else:
        host_output_file = os.path.join(logging_config.log_dir, f"host_{node_rank}_{host}.output")
    host_run_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_run.sh"
    )
    host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    pkg_dir = get_pkg_dir()
    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        before_start_cmd = cmds_config.get("before_start", "")
    else:
        before_start_cmd = ""
    cmd += f" --log-dir={logging_config.log_dir}"
    envs = config.experiment.get("envs", {})
    with open(host_run_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -x\n")
        f.write("\n")
        f.write(f"{before_start_cmd}\n")
        f.write("\n")

        f.write('if [ -z "$PYTHONPATH" ]; then\n')
        f.write(f"    export PYTHONPATH={pkg_dir}\n")
        f.write("else\n")
        f.write(f'    export PYTHONPATH="$PYTHONPATH:{pkg_dir}"\n')
        f.write("fi\n")
        f.write("\n")
        envs_str = " && ".join(
            f"export {key}={value}" for key, value in envs.items() if key != "nodes_envs"
        )
        f.write(f"{envs_str}\n")

        # Note: config key device_type is specified for single node serving in neither gpu or cpu.
        device_type = None
        nproc_per_node = None
        if config.experiment.get("runner", None) and config.experiment.runner.get(
            "device_type", None
        ):
            device_type = config.experiment.runner.get("device_type", None)
            nproc_per_node = config.experiment.runner.get("nproc_per_node", None)
            if nproc_per_node is None:
                raise ValueError(
                    f"nproc_per_node must be specified when device_type {device_type} is specified."
                )
        node_cmd = None

        if before_start_cmd:
            node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
        if node_cmd:
            f.write(f"{node_cmd}\n")

        f.write(f"mkdir -p {logging_config.log_dir}\n")
        f.write(f"mkdir -p {logging_config.pids_dir}\n")
        f.write("\n")
        f.write(f"cd {pkg_dir}\n")
        f.write("\n")
        f.write(f'cmd="{cmd}"\n')
        f.write("\n")
        # TODO: need a option to control whether to append or overwrite the output file
        # Now, it always appends to the output file
        f.write("echo '=========== launch task ==========='\n")
        if background:
            f.write(
                f'nohup bash -c "$cmd; sync" >> {host_output_file} 2>&1 & echo $! > {host_pid_file}\n'
            )
        else:
            f.write("set -o pipefail\n")
            f.write(f'bash -c "$cmd; sync" 2>&1 | tee -a {host_output_file}\n')
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_run_script_file, 0o755)

    return host_run_script_file


def _generate_stop_script(config, host, node_rank):
    logging_config = config.logging

    host_stop_script_file = os.path.join(
        logging_config.scripts_dir, f"host_{node_rank}_{host}_stop.sh"
    )

    os.makedirs(logging_config.scripts_dir, exist_ok=True)

    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        after_stop = cmds_config.get("after_stop", "")
    else:
        after_stop = ""

    cmds_config = config.experiment.get("cmds", None)
    if cmds_config:
        before_start_cmd = cmds_config.get("before_start", "")
    else:
        before_start_cmd = ""

    envs = config.experiment.get("envs", {})
    with open(host_stop_script_file, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("set -x\n")
        f.write("\n")
        f.write(f"{before_start_cmd}\n")
        f.write("\n")
        envs_str = " && ".join(f"export {key}={value}" for key, value in envs.items())
        f.write(f"{envs_str}\n")

        node_cmd = None
        if before_start_cmd:
            node_cmd = f"{before_start_cmd} && {node_cmd}" if node_cmd else before_start_cmd
        if node_cmd:
            f.write(f"{node_cmd}\n")
        f.write("pkill -f 'run_serve_vla'\n")
        f.write("pkill -f multiprocessing\n")
        f.write("\n")
        f.write(f"{after_stop}\n")
        f.flush()
        os.fsync(f.fileno())
    os.chmod(host_stop_script_file, 0o755)

    return host_stop_script_file


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


class SSHServeRunner(RunnerBase):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.task_type = getattr(self.config.experiment.task, "type", None)
        assert self.task_type == "serve", f"Unsupported task type: {self.task_type}"
        self.deploy_config = self.config.experiment.get("runner", {}).get("deploy", {})

        self._prepare()
        self.host = None

    def _prepare(self):
        _update_config_serve(self.config)
        self.user_args = _get_args_serve(self.config)
        self.user_envs = self.config.experiment.get("envs", {})
        entrypoint = self.config.experiment.task.get("entrypoint", None)
        if isinstance(entrypoint, str) and entrypoint.endswith(".py"):
            self.user_script = entrypoint
        else:
            raise ValueError(
                f"Invalid config entrypoint: {entrypoint}, must be a python file path."
            )
        hostfile_path = self.config.experiment.runner.get("hostfile", None)
        self.resources = None
        if hostfile_path:
            hostfile_path = resolve_path(
                hostfile_path, "experiment.runner.hostfile", raise_missing=True
            )
            self.resources = parse_hostfile(hostfile_path)
            for key, value in self.resources.items():
                if not value.get("type", None):
                    logger.warning(
                        f"The hostfile key type is not set for host {key}, using gpu by default"
                    )
                    self.resources[key]["type"] = "gpu"
            if self.resources:
                OmegaConf.set_struct(self.config, False)
                self.config["nodes"] = list(self.resources.items())
                OmegaConf.set_struct(self.config, True)

        logger.info("\n************** configuration **************")
        logger.info(f"\n{OmegaConf.to_yaml(self.config)}")

    def _run_each(
        self,
        host,
        master_addr,
        master_port,
        nnodes,
        node_rank,
        nproc_per_node,
        background=True,
        dryrun=False,
    ):
        export_cmd = []
        for k, v in self.user_envs.items():
            if k != "nodes_envs":
                export_cmd += [f"{k}={v}"]

        cmd = shlex.join([*export_cmd, "python", self.user_script, *self.user_args])

        host_run_script_file = _generate_run_script_serve(
            self.config, host, node_rank, cmd, background=background
        )

        run_local_command(f"bash {host_run_script_file}", dryrun)

    def run(self, background=True, dryrun=False):
        num_visible_devices = None
        visible_devices = self.user_envs.get("CUDA_VISIBLE_DEVICES", None)
        if visible_devices is not None and isinstance(visible_devices, str):
            visible_devices = visible_devices.split(",")
            num_visible_devices = len(visible_devices)

        runner_config = self.config.experiment.runner

        # In background mode, tail node 0's log file on the login node console.
        # In foreground mode, tee already streams stdout directly.
        _tail_stop = None
        if not dryrun and background:
            logging_config = self.config.logging
            no_shared_fs = self.config.experiment.runner.get("no_shared_fs", False)
            log_file = get_node0_log_file(logging_config, no_shared_fs)
            _, _tail_stop = start_tail_log(log_file)

        try:
            # If hostfile is not provided, run the job on localhost
            nproc_from_args = runner_config.get("nproc_per_node", None)
            nproc_per_node = get_nproc_per_node(None, nproc_from_args, num_visible_devices)
            available_addr = runner_config.get("master_addr", "localhost")
            available_port = runner_config.get("master_port", get_free_port())
            self._run_each(
                "localhost",
                available_addr,
                available_port,
                1,
                0,
                nproc_per_node,
                background=background,
                dryrun=dryrun,
            )
            self.host = available_addr
        finally:
            if _tail_stop:
                _tail_stop.set()

    def _stop_each(self, host, node_rank):
        logging_config = self.config.logging
        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")
        with open(host_pid_file, "r") as f:
            pid = f.readlines()[0]
            pid = int(pid.strip())
        kill_process_tree(pid)

        host_stop_script_file = _generate_stop_script(self.config, host, node_rank)
        logging_config = self.config.logging
        cmd = f"bash {host_stop_script_file}"
        logger.info(f"Run the local command: {cmd}")
        subprocess.run(
            cmd, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )

    def stop(self):
        self._stop_each("localhost", 0)
        return

    def _generate_query_script(self, host, node_rank):
        """Genetrate the query script for each host."""
        logging_config = self.config.logging

        host_query_script_file = os.path.join(
            logging_config.scripts_dir, f"host_{node_rank}_{host}_query.sh"
        )

        host_pid_file = os.path.join(logging_config.pids_dir, f"host_{node_rank}_{host}.pid")
        os.makedirs(logging_config.scripts_dir, exist_ok=True)

        with open(host_query_script_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("if [ -f " + host_pid_file + " ]; then\n")
            f.write("    pid=$(cat " + host_pid_file + ")\n")
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("else\n")
            # TODO: This is a temporary fix. We need to find a better way to query the job.
            f.write(
                "    pid=$(ps aux | grep -E 'run_serve_vla' | grep -v grep | head -n 1 | awk '{print $2}')\n"
            )
            f.write("    ps -p $pid -o state --no-headers\n")
            f.write("fi\n")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(host_query_script_file, 0o755)

        return host_query_script_file

    def _query_each(self, host, node_rank):
        "Query each node status."
        host_query_script_file = self._generate_query_script(host, node_rank)
        result = ""
        try:
            result = run_local_command(f"bash {host_query_script_file}", query=True)
        except Exception as e:
            logger.error(f"Failed to query job status on {host}: {e}")
        result = result.stdout.rstrip() if result else ""
        return result

    def _query_status(self):
        "Query Job status."
        results = []
        result = self._query_each("localhost", 0)
        results.append(result)
        if all((status != "" and status != "Z") for status in results):
            job_status = JobStatus.RUNNING
        elif all((status == "" or status == "Z") for status in results):
            job_status = JobStatus.COMPLETED_OR_IDLE
        else:
            job_status = JobStatus.TRANSITIONAL
        return job_status
