<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

[English](./README.md)

## 介绍

FlagOS-Robo 基于开源开放的多芯片 AI 软件栈 [FlagOS](https://flagos.io) 构建，是一个面向具身智能的训练与推理一体化框架。
它支持从端到云的多场景部署，兼容多种芯片，能够同时实现大脑模型（VLM）与小脑模型（VLA）的高效协同训练与推理。
FlagOS-Robo 打通从数据采集到真机与评测平台（[FlagEval](https://github.com/FlagOpen/FlagEval/)）的全流程链路，
已深度集成于智源具身智能一站式平台。
通过这一完整生态，FlagOS-Robo 将为具身智能的前沿研究与产业应用提供强大的算力底座与系统化支撑，加速智能体技术的创新与落地。

## 核心特性

- 通过 FlagOS 软件栈实现具身智能大、小脑模型的跨芯片高效训练与推理，既支持不同云侧服务器芯片，又支持不同端侧模组。
- 实现大脑模型（VLM）和小脑模型（VLA）从训练与推理部署全流程所需功能，并提供用户简单易用接口，可自适应调优和一键部署。
- 支持 [RoboOS](https://github.com/FlagOpen/RoboOS) 跨本体协作，实现不同数据格式兼容、高效端云协同、真机评测等功能。


## 安装

```sh
git clone https://github.com/flagos-ai/FlagOS-Robo.git
cd FlagOS-Robo
pip install -e .
flagos-robo --help
```

上面只装了命令行入口。每个模型还有自己的运行依赖（PyTorch、模型仓库、数据集），
按下表对应模型文档里的 `Installation` 一节装。

## 快速上手🚀

| 模型 | 类型 | 模型权重 | 训练 | 离线推理 | 在线推理 | 评估 |
|--------------|--------|--------|--------|-------------------|----------------------|---------------------------|
| PI0 | VLA | [Huggingface](https://huggingface.co/lerobot/pi0_base) | ✅︎  [文档](./examples/pi0/README.md#training) | ✅︎  [文档](./examples/pi0/README.md#inference) | ✅ [文档](./examples/pi0/README.md#serving) | ❌ |
| PI0.5 | VLA | [Huggingface](https://huggingface.co/lerobot/pi05_libero_base) | ✅︎  [文档](./examples/pi0_5/README.md#training) | ✅ [文档](./examples/pi0_5/README.md#inference) | ✅   [文档](./examples/pi0_5/README.md#serving) | ❌ |
| GR00T-N1.5 | VLA | [Huggingface](https://huggingface.co/nvidia/GR00T-N1.5-3B) | ✅︎  [文档](./examples/gr00t_n1_5/README.md#training) | ❌ | ✅   [文档](./examples/gr00t_n1_5/README.md#serving) | ❌ |

早期版本列过的 RoboBrain-2.0 / 2.5、RoboBrain-X0、Qwen-GR00T、KERV 不在本仓库里，
它们留在 [FlagScale](https://github.com/flagos-ai/FlagScale)。

## 文档

- [快速上手](./docs/GETTING_STARTED_cn.md) —— 怎么装、怎么确认装好了、`flagos-robo` 四个子命令各是干什么的。
- [变更记录](./CHANGELOG.md) —— 每个版本包含什么。
- [docs/](./docs/README.md) —— 迁移记录、验收报告、精度基线（开发过程记录，不是用户指南）。
