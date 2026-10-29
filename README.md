[中文](./README_cn.md)

## FlagOS-Robo Overview

🤖 FlagOS-Robo is built upon the unified and open-source AI system software stack, [FlagOS](https://flagos.io), which supports various AI chips.
It serves as an integrated training and inference framework for AI models used in robots🤖 , so-called Embodied Intelligence.
It can be deployed across diverse scenarios, ranging from edge to cloud.
Being portable across various chip models, it enables efficient training, inference, and deployment
for both Vision Language Models (VLMs) and Vision Language Action (VLA) models.
Here, VLMs usually act as the brain🧠 for task planning, while VLA models act as the cerebellum to output actions for robot control🦾.

FlagOS-Robo supports the full lifecycle of embodied intelligence models,
including **data loading** from diverse formats (webdataset, Megatron-Energon and lerobot dataset), **supervised fine-tuning** (SFT),
**inference deployment**, and integrated **testing and evaluation** via the [FlagEval-Robo](https://embodiedverse.flageval.net/home) platform. Users can easily reproduce the full end-to-end pipeline in their own environment by downloading and running the provided examples.

FlagOS-Robo has been deeply integrated into **BAAI’s Embodied Intelligence platform** [RoboXStudio](https://ei2data.baai.ac.cn/home), which provides one-stop services including real-robot data collection, data annotation, supervised fine-tuning of VLA models, and evaluation. Users without a local setup can directly access RoboXStudio and run experiments without any installation.

FlagOS-Robo provides a powerful computational foundation and systematic support for cutting-edge researches
and industrial applications in embodied intelligence, accelerating innovations and real-world deployments
of intelligent agents.

## Feature Highlights

- FlagOS-Robo is the entrypoint for users: a single `flagos-robo` command line covers training, offline inference, and online serving for the models shipped in [`examples/`](./examples) — currently PI0, PI0.5, and GR00T-N1.5.
- FlagOS-Robo supports [RoboOS](https://github.com/FlagOpen/RoboOS)-based cross-embodiment collaboration,
  ensuring compatibility with different data formats, efficient edge-cloud coordination,
  and real-machine evaluation.

## Installation

```sh
git clone https://github.com/flagos-ai/FlagOS-Robo.git
cd FlagOS-Robo
pip install -e .
flagos-robo --help
```

The command above installs the launcher only. Each model needs its own runtime
dependencies (PyTorch, the model repository, datasets) — follow the
`Installation` section of that model's guide below.

## Quick Start🚀

| Models | Type | Checkpoint | Train | Inference | Serve | Evaluate |
|--------------|--------|--------|--------|-------------------|----------------------|---------------------------|
| PI0 | VLA | [Huggingface](https://huggingface.co/lerobot/pi0_base) | ✅︎  [Guide](./examples/pi0/README.md#training) | ✅︎  [Guide](./examples/pi0/README.md#inference) | ✅ [Guide](./examples/pi0/README.md#serving) | ❌ |
| PI0.5 | VLA | [Huggingface](https://huggingface.co/lerobot/pi05_libero_base) | ✅︎  [Guide](./examples/pi0_5/README.md#training) | ✅ [Guide](./examples/pi0_5/README.md#inference) | ✅   [Guide](./examples/pi0_5/README.md#serving) | ❌ |
| GR00T-N1.5 | VLA | [Huggingface](https://huggingface.co/nvidia/GR00T-N1.5-3B) | ✅︎  [Guide](./examples/gr00t_n1_5/README.md#training) | ❌ | ✅   [Guide](./examples/gr00t_n1_5/README.md#serving) | ❌ |

Models listed in earlier releases (RoboBrain-2.0 / 2.5, RoboBrain-X0,
Qwen-GR00T, KERV) are not part of this repository; they remain in
[FlagScale](https://github.com/flagos-ai/FlagScale).

## Documentation

- [Getting Started](./docs/GETTING_STARTED.md) — install, verify, and the four
  `flagos-robo` subcommands.
- [Changelog](./CHANGELOG.md) — what each release contains.
- [docs/](./docs/README.md) — migration records, acceptance reports, precision
  baselines (development history, not user guides).
