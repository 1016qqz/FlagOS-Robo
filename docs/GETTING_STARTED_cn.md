# 快速上手

本页只讲启动器：怎么安装 `flagos-robo`、怎么确认装好了、四个子命令各自做什么。
真正跑起一个模型还需要该模型自己的运行依赖，那部分在
[`examples/`](../examples) 下各模型的文档里。

- [1. 环境要求](#1-环境要求)
- [2. 安装启动器](#2-安装启动器)
- [3. 确认装好了](#3-确认装好了)
- [4. 四个子命令](#4-四个子命令)
- [5. 配置文件是怎么找到的](#5-配置文件是怎么找到的)
- [6. 动作开关](#6-动作开关)
- [7. 跑一个真实模型](#7-跑一个真实模型)
- [8. 出问题时怎么查](#8-出问题时怎么查)

## 1. 环境要求

| 项 | 取值 | 在哪里声明 |
|---|---|---|
| Python | >= 3.10 | `pyproject.toml` 的 `requires-python` |
| 运行依赖 | `hydra-core>=1.3.2`、`omegaconf>=2.3.0`、`typer>=0.9.0` | `pyproject.toml` 的 `[project] dependencies` |
| 构建后端 | `setuptools.build_meta` | `pyproject.toml` 的 `[build-system]` |

PyTorch **不是**启动器的依赖。各模型文档各自写明自己需要的 PyTorch 版本、
模型仓库和数据集，因为这些按模型和按加速卡都不一样。

## 2. 安装启动器

```sh
git clone https://github.com/flagos-ai/FlagOS-Robo.git
cd FlagOS-Robo
pip install -e .
```

本版本推荐用可编辑安装：命令行是相对当前目录去找示例配置的
（见[第 5 节](#5-配置文件是怎么找到的)），所以在仓库根目录下工作是常规用法。

非可编辑安装也可以：

```sh
pip install .
```

这种情况下仓库仍然要留着 —— `examples/` 不随安装包一起装进去。

## 3. 确认装好了

```sh
flagos-robo --version
flagos-robo --help
```

`--help` 会列出四个子命令：`run`、`train`、`serve`、`inference`。

不走命令行脚本也能到同一个入口：

```sh
python -c "import flagos_robo; print(flagos_robo.__version__)"
```

## 4. 四个子命令

### `train <model>`

训练一个模型。

```sh
flagos-robo train pi0
flagos-robo train pi0 --config ./examples/pi0/conf/train.yaml
flagos-robo train pi0 --stop
```

### `inference <model>`

拿训练好的权重做离线推理。

```sh
flagos-robo inference pi0 --config ./examples/pi0/conf/inference.yaml
```

### `serve <model>`

用 websocket 协议把策略服务起来。

```sh
flagos-robo serve pi0 --config ./examples/pi0/conf/serve.yaml
flagos-robo serve pi0 --stop
```

### `run --config-path <dir> --config-name <name>`

底层形式。它收的是配置目录加配置文件名，而不是模型名；命令行末尾多余的
`key=value` 会作为 Hydra 覆盖项传下去。

```sh
flagos-robo run --config-path ./examples/pi0/conf --config-name train --action run
flagos-robo run -p ./examples/pi0/conf -n train -a stop
```

`--action` 可取 `run`、`dryrun`、`test`、`stop`、`query`。

## 5. 配置文件是怎么找到的

`train` / `serve` / `inference` 收的是模型名。不给 `--config` 时，配置按这个
位置去找：

```
<当前目录>/examples/<model>/conf/<task>.yaml
```

所以在仓库根目录下执行 `flagos-robo train pi0`，读的是
`examples/pi0/conf/train.yaml`。命令要在仓库根目录下跑，或者用 `--config`
给出明确路径。本版本实际存在的配置：

| 模型 | train | inference | serve |
|---|---|---|---|
| `pi0` | ✅ | ✅ | ✅ |
| `pi0_5` | ✅ | ✅ | ✅ |
| `gr00t_n1_5` | ✅ | ❌ | ✅ |

文件不存在时，命令行会打印 `Error: <path> does not exist` 并以状态码 1 退出，
不会去猜。

## 6. 动作开关

`train` / `serve` / `inference` 共用一组互斥开关，用来选动作；一个都不给时
动作是 `run`。

| 开关 | 含义 | 哪些子命令有 |
|---|---|---|
| （不给） | 启动任务 | train、serve、inference |
| `--dryrun` | 只校验配置，不启动任务 | train、serve、inference |
| `--test` | 带测试路径跑 | train、serve、inference |
| `--stop` | 停掉正在跑的任务 | train、serve、inference |
| `--query` | 查任务状态 | train |

同时给两个会被拒：命令行打印
`Error: Flags are mutually exclusive: ...` 并以状态码 1 退出。

刚改完配置想确认有没有写错，`--dryrun` 是最省的办法 —— 它会把 Hydra 配置解析
一遍并报错，但不启动任何任务。

## 7. 跑一个真实模型

启动器不负责装模型运行环境。选一个模型，照它自己的文档从头走一遍 —— 每份
文档都写了自己的 conda 环境、PyTorch 版本、模型下载、数据集目录结构和具体命令：

- [`examples/pi0/README.md`](../examples/pi0/README.md) —— 训练、离线推理、服务
- [`examples/pi0_5/README.md`](../examples/pi0_5/README.md) —— 训练、离线推理、服务
- [`examples/gr00t_n1_5/README.md`](../examples/gr00t_n1_5/README.md) —— 训练、服务

## 8. 出问题时怎么查

| 现象 | 大概率原因 |
|---|---|
| `flagos-robo: command not found` | 没执行安装，或者当前激活的环境不是装了它的那个 |
| `Error: <path>/examples/<model>/conf/<task>.yaml does not exist` | 不在仓库根目录下，或者该模型没有这个任务的配置（见[第 5 节](#5-配置文件是怎么找到的)的表） |
| `Config error in: <file>` 后面跟一段 Hydra 报错 | 配置 YAML 里有字段缺失或解析不出来；紧跟的提示行写了该查什么 |
| `Error: Flags are mutually exclusive` | 一次给了两个动作开关（见[第 6 节](#6-动作开关)） |
| 报 torch、模型仓库或数据集包的 import 错误 | 模型运行依赖没装 —— 照该模型自己的文档装，不是看本页 |

评测不在本版本范围内。这里没有 `flagos-robo eval`；评测流程跑在
[FlagEval-Robo](https://embodiedverse.flageval.net/home) 平台上。
