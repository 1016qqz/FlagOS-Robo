# FlagOS-Robo 迁移映射表（MIGRATION_MAP）

> 本文件是 FlagOS-Robo 从 FlagScale 独立建仓的**迁移契约**。W1 定稿，W2 起不得擅自变更；
> 如需变更，必须在本文件记录变更原因与日期。

## 一、基线钉死（W1 D1.5 / 风险 R1）

| 项 | 值 |
|---|---|
| 上游仓库 | `FlagScale`（github.com/flagos-ai/FlagScale） |
| **baseline commit** | **`f8e9a62c499d526a6ffa47f533bcf342330b7326`**（短号 `f8e9a62c`） |
| baseline commit 标题 | `[Fix] fix flagcx non hetero init (#1303)` |
| 钉死时间 | W1 D1 |

**约定**：W2–W8 的所有代码迁移、行号引用、精度对齐均以此 commit 为准。
上游在此之后的变更**不自动跟进**，而是在本文件 §五「上游差异记录」中单独登记，
由负责人决定是否回合，避免「迁移过程中上游漂移」导致对齐结论失效。

校验命令（在 FlagScale 工作副本中执行）：

```bash
git rev-parse HEAD   # 必须等于 f8e9a62c499d526a6ffa47f533bcf342330b7326
```

## 二、包命名映射契约

| FlagScale | FlagOS-Robo | 迁移批次 | 备注 |
|---|---|---|---|
| `flagscale.platforms` | `flagos_robo.platforms` | **W1** | 直接对应，8 文件 |
| `flagscale.logger` | `flagos_robo.logger` | **W1** | 被 5 处引用 |
| `flagscale.models.utils` | `flagos_robo.models.utils` | **W1** | `constants` 被引 30 次（最热节点） |
| `flagscale.models.configs` | `flagos_robo.models.configs` | **W1** | `types` 被引 25 次 |
| `flagscale.models.pi0` | `flagos_robo.models.pi0` | W2–W3 | 4 文件 / 1674 LOC |
| `flagscale.models.pi05` | `flagos_robo.models.pi05` | W2–W3 | 6 文件 / 2053 LOC |
| `flagscale.models.vla` | `flagos_robo.models.vla` | W2–W3 | 27 文件中迁 21；`qwen_gr00t/`(4) + `vlm/`(2) 留在 FlagScale |
| `flagscale.train.datasets` | `flagos_robo.data` | W2–W3 | **改名**，对齐架构图「数据层」；12 文件 / 6482 LOC |
| `flagscale.train.processor` | `flagos_robo.processor` | W2–W3 | **提级**，processor 非训练专属；15 文件 / 4524 LOC |
| `flagscale.train.utils` | `flagos_robo.train.utils` | W2–W3 | 8 文件 / 1640 LOC |
| `flagscale.train.train_pi` | `flagos_robo.train.train_pi` | W2–W3 | 767 LOC |
| `flagscale.train.train_gr00t_n1_5` | `flagos_robo.train.train_gr00t_n1_5` | W2–W3 | 671 LOC |
| `flagscale.train.train_config` | `flagos_robo.train.train_config` | W2–W3 | 336 LOC |
| `flagscale.transformations` | `flagos_robo.transformations` | W2–W3 | 6 文件 |
| `flagscale.inference.inference_pi` | `flagos_robo.inference.inference_pi` | W2–W3 | 211 LOC，隐藏范围（见 §四 B2） |
| `flagscale.serve.run_serve_vla` | `flagos_robo.serve.run_serve_vla` | W2–W3 | 293 LOC，隐藏范围 |
| `flagscale.serve.processor.*` | `flagos_robo.serve.processor.*` | W2–W3 | 18+102+69 LOC |
| `flagscale.runner.runner_train` | `flagos_robo.runner.runner_train` | W2–W3 | 仅 native train 链路，非全部 63 文件 |
| `flagscale.runner.backend_native_train` | `flagos_robo.runner.backend_native_train` | W2–W3 | — |
| `flagscale.runner.utils.logger` | `flagos_robo.logger` | W2–W3 | **合并**，`transformations` 内 2 处引用 |

### 特殊改写

| 原引用 | 改写为 | 位置 | 批次 |
|---|---|---|---|
| `megatron.plugin.platform.get_platform` | `flagos_robo.platforms.get_platform` | `train/processor/device_processor.py:29` | W2 |
| `megatron.plugin.platform.get_platform` | `flagos_robo.platforms.get_platform` | `train/utils/random_utils.py:25` | W2 |
| `megatron.energon.DefaultTaskEncoder` | —（文件不迁移，自动消解） | `models/vla/qwen_gr00t/task_encoder_qwen_gr00t.py:22` | 不迁 |
| `flagscale.inference.*` | 降级为 `try/except ImportError` 可选依赖 | `transformations/` 内 3 处 | W2–W3 |

## 三、W1 修复的 API 缺口

`flagscale/platforms/` 的 `PlatformBase` 原有 9 个方法，**缺少 `current_device()`**，
而 `train_pi.py:113` 调用 `get_platform().current_device()`（用于 `FSDP(device_id=...)`）
—— 说明该入口实际绑定的是 `megatron.plugin.platform` 那一套，两套平台层已 API 漂移。

W1 在 `flagos_robo/platforms/` 中**一次性收口**：`PlatformBase` 新增抽象方法
`current_device()`，CUDA / NPU / MUSA / TXDA 四个后端全部实现。
否则 W2 迁入训练入口后会直接 `AttributeError`。

| 后端 | `current_device()` 实现 |
|---|---|
| cuda | `torch.cuda.current_device()` |
| npu | `torch.npu.current_device()` |
| musa | `torch.musa.current_device()` |
| txda | `torch.cuda.current_device()`（与其 `set_device`/`device_count` 保持一致，txda 复用 cuda runtime） |

## 四、迁移范围校验 SOP（三管齐下）

AST 静态 import 闭包有三类结构性盲区，**每类都已在本仓实测命中**，故范围校验必须三管齐下：

| # | 盲区 | 实测命中 | 处置 |
|---|---|---|---|
| B1 | **动态 `importlib` 类路径**（运行期解析，改包名后 import 期不报错，跑到那一步才 `ImportError`） | `train/processor/pipeline.py:866` 的 `importlib.import_module(module_path)`，`module_path` 来自 config 的 `step_entry["class"]` 全类路径字符串；同类还有 `models/kerv/entrypoint.py:180/236`、`kerv/ops/__init__.py:46` | 迁移集 `.py` 内 `"flagscale.*"` 字符串字面量实测 0 处 → 风险全在外部 YAML，归入 B3 一并改写 |
| B2 | **非训练入口的同级消费者**（只用 3 个 train 入口做 seed 会漏） | `examples/pi0*/conf/` 还声明了 `flagscale/inference/inference_pi.py`(211) 与 `flagscale/serve/run_serve_vla.py`(293)，并额外拉入 serve 侧 4 模块（`websocket_policy_server.py` 110 + `msgpack_numpy.py` 74 + `serve/processor/` 18+102+69） | **~877 LOC 隐藏范围**。W1 不迁，但已在包结构中预留 `flagos_robo/inference/`、`flagos_robo/serve/{,processor/}` 占位，避免 W2 才改包结构 |
| B3 | **YAML / README 的路径与 CLI 字符串**（grep `.py` 查不到） | `examples/{pi0,pi0_5,gr00t_n1_5}/` 共 **34 处** `flagscale` 文本：`conf/*.yaml` 的 `entrypoint: flagscale/train/train_pi.py`、README 的 `flagscale train pi0` | 随包重命名同步改写；验收加 `grep -rn "flagscale" examples/` 必须为空 |

**固化为三步校验**（见 `tools/check_import_closure.py`）：

1. AST 闭包定静态边界 —— seed **必须枚举 `examples/*/conf/*.yaml` 里所有 `entrypoint`**，不能只取 train；
2. `grep -rnE "importlib|__import__|import_module"` + `grep -rnE "[\"']flagscale\."` 定动态边界；
3. `grep -rn flagscale examples/ *.md` 定文本边界。

并且闭包脚本必须做**负对照**（`--self-test`：剔除一个已知必需模块后 outside 必须由 0 变正），
证明「outside=0」是真结论而非脚本空转。

## 五、上游差异记录

| 日期 | 上游 commit | 影响模块 | 是否回合 | 说明 |
|---|---|---|---|---|
| — | — | — | — | 暂无（baseline 钉死于 W1 D1） |
