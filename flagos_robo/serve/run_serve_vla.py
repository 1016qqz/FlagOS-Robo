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

# Mainly adopted from:
# https://github.com/starVLA/starVLA/blob/3f7feefbc5fc25890ad3a7d262b8a0aea1339aa7/deployment/model_server/server_policy.py

import argparse
import time

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

import flagos_robo.serve.processor  # noqa: F401 — registers serve-specific processor steps
from flagos_robo.logger import logger
from flagos_robo.platforms import get_platform  # noqa: F401 must be before model imports

# TrainablePolicy / PreTrainedConfig dispatch is registry-based: a type is only
# resolvable after its module has been imported, because registration happens in
# ``__init_subclass__``. flagos_robo/models/vla/__init__.py deliberately does NOT
# import pi0/pi05 (pi0.modeling_pi0 imports back into vla, so re-exporting them from
# there is circular -- see the note in that file), which means every entrypoint has to
# register them itself. train_pi.py and inference_pi.py already import the concrete
# classes; this entrypoint only ever touches the base class, so it needs the two
# side-effect imports below. Without them, serving a pi0 checkpoint dies in
# PreTrainedConfig.from_pretrained with:
#   ValueError: Unknown config type 'pi0'. Known types: ['Gr00tN15']
import flagos_robo.models.pi0  # noqa: E402,F401 — registers PI0Config / PI0Policy
import flagos_robo.models.pi05  # noqa: E402,F401 — registers PI05Config / PI05Policy

# Same registry story one level down, for processor steps rather than policies: the
# GR00T-specific steps (groot_pack_inputs / groot_eagle_encode / groot_eagle_collate /
# groot_action_unpack_unnormalize) are registered by the @ProcessorStepRegistry.register
# decorators in models/vla/gr00t_n1_5/processor_gr00t.py, and that module is not reached
# via vla/gr00t_n1_5/__init__.py (which only re-exports configuration_/modeling_).
# train_gr00t_n1_5.py carries this same side-effect import; without it here, serving a
# GR00T checkpoint dies in PolicyProcessorPipeline.from_pretrained with:
#   KeyError: "Processor step 'groot_pack_inputs' not found in registry"
import flagos_robo.models.vla.gr00t_n1_5.processor_gr00t  # noqa: E402,F401
from flagos_robo.models.utils.constants import ACTION, OBS_IMAGES, OBS_STATE  # noqa: E402
from flagos_robo.models.vla import TrainablePolicy  # noqa: E402
from flagos_robo.models.vla.pretrained_config import PreTrainedConfig  # noqa: E402
from flagos_robo.serve.processor.image_layout_processor import ImageLayoutProcessorStep
from flagos_robo.serve.processor.image_resize_processor import ImageResizeProcessorStep
from flagos_robo.serve.processor.numpy_to_tensor_processor import NumpyToTensorProcessorStep
from flagos_robo.serve.websocket_policy_server import WebsocketPolicyServer
from flagos_robo.train.processor import PolicyProcessorPipeline, ProcessorStepRegistry
from flagos_robo.train.processor.pipeline import get_device_override

# TODO: (yupu) to constant.py?
TASK_KEY = "task"


def _default_serve_preprocessor(model_variant: str) -> PolicyProcessorPipeline | None:
    """Return the default serve preprocessor for a given model variant, or None if not needed."""
    variant = model_variant.lower()
    if variant == "pi0":
        return _default_pi0_serve_preprocessor()
    if variant == "pi0.5":
        return _default_pi05_serve_preprocessor()
    if variant == "qwengr00t":
        return _default_qwen_gr00t_serve_preprocessor()
    return None


def _pi_serve_steps() -> list:
    """Serve preprocessor steps shared by pi0 and pi0.5.

    ``_preprocess_images`` in both models expects images as [B, C, H, W] float32 [0, 1]
    ``torch.Tensor`` (it reads ``img.device``), and the policy preprocessor steps that
    follow gate on ``isinstance(x, torch.Tensor)``. Client observations arrive as HWC
    uint8 numpy arrays, so the layout step converts them and the tensor step moves the
    whole observation (images AND state) onto torch.

    Order matters: ``ImageLayoutProcessorStep`` gates on ``isinstance(img, np.ndarray)``,
    so it has to run BEFORE the numpy->tensor conversion or it would skip every image.
    Neither model needs an explicit resize step -- ``resize_with_pad_torch`` inside
    ``_preprocess_images`` handles the target resolution.
    """
    return [
        ImageLayoutProcessorStep(
            src_layout="hwc",
            dst_layout="chw",
            add_batch_dim=True,
            to_float=True,
        ),
        NumpyToTensorProcessorStep(dtype="float32"),
    ]


def _default_pi0_serve_preprocessor() -> PolicyProcessorPipeline:
    """Default serve preprocessor for pi0."""
    return PolicyProcessorPipeline(steps=_pi_serve_steps())


def _default_pi05_serve_preprocessor() -> PolicyProcessorPipeline:
    """Default serve preprocessor for pi0.5."""
    return PolicyProcessorPipeline(steps=_pi_serve_steps())


def _default_qwen_gr00t_serve_preprocessor() -> PolicyProcessorPipeline:
    """Default serve preprocessor for QwenGr00t.

    Resizes images to the training resolution (224x224). The Qwen VL processor
    handles further preprocessing (normalization, patching) internally.
    """
    return PolicyProcessorPipeline(steps=[ImageResizeProcessorStep(image_size=[224, 224])])


def validate_batch(batch: dict) -> list[str]:
    """Validate a batch against the internal data contract.

    Expected canonical keys (after rename_map has been applied)::

        - ``observation.images.*``: ``np.ndarray``, HWC uint8, ndim == 3
        - ``observation.state``: ``np.ndarray`` or ``list``
        - ``task``: ``str``

    Returns a list of warning/error messages. Empty list means the batch is valid.
    """
    errors: list[str] = []

    if TASK_KEY not in batch:
        errors.append(f"Missing required key '{TASK_KEY}'")
    elif not isinstance(batch[TASK_KEY], str):
        errors.append(f"'{TASK_KEY}' must be str, got {type(batch[TASK_KEY]).__name__}")

    if OBS_STATE not in batch:
        errors.append(f"Missing required key '{OBS_STATE}'")
    elif not isinstance(batch[OBS_STATE], (np.ndarray, list)):
        errors.append(
            f"'{OBS_STATE}' must be np.ndarray or list, got {type(batch[OBS_STATE]).__name__}"
        )

    image_keys = [k for k in batch if k.startswith(OBS_IMAGES)]
    if not image_keys:
        errors.append(f"No image keys found (expected keys starting with '{OBS_IMAGES}')")

    for key in image_keys:
        img = batch[key]
        if not isinstance(img, np.ndarray):
            errors.append(f"'{key}' must be np.ndarray, got {type(img).__name__}")
            continue
        if img.ndim != 3:
            errors.append(f"'{key}' must be HWC (ndim=3), got ndim={img.ndim} shape={img.shape}")
        if img.dtype != np.uint8:
            errors.append(f"'{key}' expected dtype=uint8, got {img.dtype}")

    return errors


def debug_print(batch):
    for k, v in batch.items():
        if hasattr(v, "shape"):
            logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")
        else:
            logger.info(f"  {k}: type={type(v).__name__} value={repr(v)[:120]}")


class Policy:
    """VLA policy server wrapping a TrainablePolicy with pre/post-processing.

    Loads a pretrained policy model and its processor pipelines from a checkpoint.
    Optionally applies serve-time preprocessing (e.g., image resize) and observation
    key renaming before the saved model preprocessor.

    Args:
        config: Serve config containing an ``engine_args`` section.
    """

    def __init__(self, config: DictConfig | ListConfig) -> None:
        self.config_engine = config["engine_args"]

        self.host: str = self.config_engine.get("host", "0.0.0.0")
        self.port: int = self.config_engine.get("port", 5000)
        self.model: TrainablePolicy | None = None
        self.preprocessor: PolicyProcessorPipeline | None = None
        self.postprocessor: PolicyProcessorPipeline | None = None
        self.serve_preprocessor: PolicyProcessorPipeline | None = None
        self.rename_map: dict[str, str] | None = None

        self.load_policy()

    def load_policy(self) -> None:
        """Load the policy model and all processor pipelines from the checkpoint."""
        t_s = time.perf_counter()
        pretrained_dir: str = self.config_engine.model
        runtime_device = getattr(self.config_engine, "device", None) or "cpu"
        # ``device=`` alone is NOT enough: PI0Policy/PI05Policy.from_pretrained have no
        # ``device`` parameter, so base_policy.py's device argument lands in **kwargs and is
        # silently dropped. What actually moves the weights is ``model.to(config.device)``
        # inside the policy's own loader -- i.e. the *checkpoint's* device field wins.
        # pi05_libero_base ships ``"device": "mps"``, which on a CUDA box raises
        # "PyTorch is not linked with support for mps devices" (wrapped as "Could not load
        # PI05 pretrained weights"). So load the config first and let the runtime device
        # override whatever the checkpoint recorded -- same two-step inference_pi.py:82-83 does.
        config = PreTrainedConfig.from_pretrained(pretrained_dir)
        config.device = runtime_device
        self.model = TrainablePolicy.from_pretrained(
            pretrained_dir, device=runtime_device, config=config
        )
        self._load_processors(pretrained_dir)
        # TODO: (yupu): model.to(dtype)?
        logger.info(f"Policy model loading latency: {time.perf_counter() - t_s:.2f}s")

    def _load_processors(self, pretrained_dir: str) -> None:
        """Load pre/post-processors from the checkpoint and build the serve preprocessor."""
        self.rename_map = self.config_engine.get("rename_map")

        runtime_device = getattr(self.config_engine, "device", None) or "cpu"
        overrides = dict(get_device_override(runtime_device))
        # The checkpoint's policy_preprocessor.json stores ``tokenizer_name`` as a HF
        # Hub repo id (e.g. "google/paligemma-3b-pt-224"), which is a gated repo: loading
        # it would hit the Hub and fail with 401 in an offline/token-less environment.
        # The serve config carries a local tokenizer directory, so redirect the step to
        # it -- same override inference_pi.py:150 applies. Without this the
        # ``engine_args.tokenizer`` field is silently dead config.
        tokenizer_path = self.config_engine.get("tokenizer")
        if tokenizer_path:
            overrides["tokenizer_processor"] = {"tokenizer_name": str(tokenizer_path)}
        self.preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_preprocessor.json",
            overrides=overrides,
        )
        self.postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_dir,
            config_filename="policy_postprocessor.json",
        )
        self.serve_preprocessor = self._build_serve_preprocessor()

    def _build_serve_preprocessor(self) -> PolicyProcessorPipeline | None:
        """Build a serve-time preprocessor pipeline from the ``serve_preprocessor`` config.

        Returns ``None`` if no serve preprocessor is configured. Steps are resolved
        from the ``ProcessorStepRegistry`` by their ``registry_name``.
        """
        serve_preproc_cfg = self.config_engine.get("serve_preprocessor")
        if not serve_preproc_cfg or not serve_preproc_cfg.get("steps"):
            # TODO: (yupu) No way to explicitly opt out of the default — need a flag like
            # ``serve_preprocessor: {disabled: true}`` for cases where no steps are wanted.
            return _default_serve_preprocessor(self.config_engine.get("model_variant", ""))
        steps = []
        for step_cfg in serve_preproc_cfg.steps:
            step_cls = ProcessorStepRegistry.get(step_cfg.registry_name)
            config = OmegaConf.to_container(step_cfg.get("config", {}), resolve=True)
            steps.append(step_cls(**config))
        return PolicyProcessorPipeline(steps=steps)

    def inference(self, batch: dict) -> dict:
        """Run inference on a single batch.


        Args:
            batch: Raw observation dict from the client. Image values are typically
                ``np.ndarray`` in HWC uint8 format.

        Returns:
            Dict with ``action`` key containing a numpy array of shape
            ``[B, T, action_dim]``.
        """
        t_req = time.perf_counter()
        logger.info("Start to inference")
        logger.info(f"Raw batch keys: {list(batch.keys())}")

        if self.rename_map:
            batch = {self.rename_map.get(k, k): v for k, v in batch.items()}
            logger.info(f"After rename keys: {list(batch.keys())}")

        errors = validate_batch(batch)
        if errors:
            for err in errors:
                # TODO: (yupu) Response with error status?
                logger.warning(f"Batch validation: {err}")

        debug_print(batch)
        # for k, v in batch.items():
        #     if hasattr(v, "shape"):
        #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")
        #     else:
        #         logger.info(f"  {k}: type={type(v).__name__} value={repr(v)[:120]}")

        if self.serve_preprocessor:
            batch = self.serve_preprocessor(batch)
            logger.info("After serve_preprocessor:")
            debug_print(batch)
            # for k, v in batch.items():
            #     if hasattr(v, "shape"):
            #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")

        batch = self.preprocessor(batch)
        logger.info("After preprocessor:")
        debug_print(batch)
        # for k, v in batch.items():
        #     if hasattr(v, "shape"):
        #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype}")

        t_infer = time.perf_counter()
        with torch.no_grad():
            action = self.model.predict_action(batch)
        predict_s = time.perf_counter() - t_infer

        logger.info(f"Raw action keys: {list(action.keys())}")
        debug_print(action)
        # for k, v in action.items():
        #     if hasattr(v, "shape"):
        #         logger.info(f"  {k}: shape={v.shape} dtype={v.dtype} first_step={v[0,0,:7]}")

        action = self.postprocessor(action)

        # Convert to numpy for msgpack serialization; squeeze batch dim [1,T,D] → [T,D]
        action[ACTION] = action[ACTION].squeeze(0).detach().cpu().numpy()
        # TODO: (yupu): rename_map for output key
        action["actions"] = action[ACTION]
        logger.info(f"Final action shape: {action[ACTION].shape}, first_step={action[ACTION][0]}")

        # Per-request serve metric. The repo convention for a robo serve entrypoint is a
        # one-line latency record per request (see FlagScale
        # flagscale/serve/run_serve_robobrain_x0.5_libero.py:102 "Infer latency"); the VLA
        # entrypoint only timed model loading. ``predict_s`` isolates the policy forward,
        # ``total_s`` covers pre/post-processing too, so a slow round can be attributed
        # without a profiler. serve/metric/serve_metric.py is deliberately NOT used here:
        # calculate_metrics() computes LLM batch-benchmark stats (ttft/tpot/itl over a
        # request list) and does not describe a single VLA action-chunk request.
        total_s = time.perf_counter() - t_req
        logger.info(
            f"Infer latency: total={total_s:.3f}s predict={predict_s:.3f}s "
            f"action_shape={tuple(action[ACTION].shape)}"
        )

        return action


def parse_config() -> DictConfig | ListConfig:
    """Parse the configuration file"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path", type=str, required=True, help="Path to the configuration YAML file"
    )
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the log")
    args = parser.parse_args()
    config = OmegaConf.load(args.config_path)
    return config


def main(config: DictConfig | ListConfig) -> None:
    """Start the websocket policy server."""
    policy = Policy(config)
    # start websocket server
    server = WebsocketPolicyServer(
        policy=policy,
        host=policy.host,
        port=policy.port,
        metadata={"env": "simpler_env"},
    )
    logger.info(f"Server running at {policy.host}:{policy.port}...")
    server.serve_forever()


if __name__ == "__main__":
    parsed_cfg = parse_config()
    if isinstance(parsed_cfg, ListConfig):
        main(parsed_cfg[0])
    else:
        main(parsed_cfg["serve"][0])
