# FlagOS-Robo W1–W3 代码 Review（框架工程视角）

Review 对象：`dev` 分支 HEAD `e966889`，全仓 163 个 tracked `.py` / 41637 LOC，
其中包主体 `flagos_robo/` 为 128 个模块 / 31755 LOC（下文覆盖率等指标均以此为分母）。
Review 方式：静态分析（AST + grep）为主，配合真构建 / 真安装 / 真 import / 四门禁复跑。
本机无 torch，全量 pytest 不可执行 —— 相关限制在文末「证据边界」如实标注。

本次 review 自研了 5 个审计工具，落盘在 `robo_file/_ops/review_*.py`（纯 stdlib，无 torch 可跑）。
每个工具都先做注入测试（人造已知阳性 + 已知阴性）确认不误报、不漏报，才用于下结论。

---

## 结论摘要

| 级别 | 数量 | 内容 |
|---|---|---|
| **P0 错误** | 1 | `heartbeat/monitor.py` 缺模块，import 必然失败 |
| **P1 错误** | 2 | `__all__` 悬挂名；`get_addr()` 环回地址过滤恒失效 |
| **P2 可优化** | 5 | 测试覆盖盲区、日志双通道、裸 except、依赖声明空洞、门禁盲点加固 |
| **P3 建议** | 4 | `__init__.py` 形态统一、冗余 pin 清理、TODO 归档、`MagicMock/` 测试污染 |
| 已验证无问题 | 4 | packaging 完整性、命名残留、硬编码路径、被裁符号悬挂引用 |

最值得注意的一点：**四门禁全绿（G1/G1'/G2/G3 全部 RC=0，self-test 21/21）却漏掉了 P0 错误**。
这不是门禁实现 bug，而是「静态闭包 + 动态引用白名单」双通道设计的结构性盲点，详见 P2-5。
对迁移类项目，这个盲点比单个缺陷本身更值得修。

> **修复状态（后续补记）：** P0-1、P1-1、P1-2、P2-3、P2-5 已修复落地，
> 下文各节描述的是**修复前**的代码状态，请勿当作现存缺陷。
> 逐项状态、实测证据与三处「照原建议改会改错」的偏离说明见「修复优先级」一节。
> 修复过程中另有**两项首轮 review 未发现的真缺陷**被反向检验抓出并一并修掉
> （CI 负对照的硬编码工具列表对新门禁盲视；G6 会扫进 vendored 的 `_ci/`
> 上游克隆因而在干净代码上报红），见「已修项的落地记录」第 4、5 项。
> 仍未修：P2-1、P2-2、P2-4、P3-*，以及 **P2-5 的根治方案（闭包门禁 fixpoint 迭代）**。

---

## P0 — 必须修

### P0-1 `runner/heartbeat/monitor.py` 引用不存在的模块

**位置**：`flagos_robo/runner/heartbeat/monitor.py:29`

```python
from .health_reader import ...   # health_reader.py 不在仓里
```

**证据**：

- 目录与 git 双向确认：`flagos_robo/runner/heartbeat/` 下无 `health_reader.py`，
  `git ls-files` 亦无；而 `monitor.py` 本身是 tracked 文件。
- 真 import 复现（先装齐 omegaconf 排除依赖干扰）：
  `ModuleNotFoundError: No module named 'flagos_robo.runner.heartbeat.health_reader'`
- 上游存在该文件：与本仓同级的 `../FlagScale/flagscale/runner/heartbeat/health_reader.py`，
  292 行，import 仅 `json` / `dataclasses` / `typing` —— 纯 stdlib，补搬不引入新依赖。
- 全仓 128 个叶子模块逐个真 import（`review_leaf_import.py`）：
  `MISSING-INTERNAL = 1`（即本项）、`OTHER = 0`，其余 101 个失败全是本机缺第三方包。
  → 这是**唯一一处**同类漏网，不是系统性问题。

**影响**：`enable_monitoring` 开启时心跳监控进程起不来。
由于 `monitor` 是经 `heartbeat/config.py` 的字符串模块路径动态加载，
失败点会远离真实根因，排查成本高。

**修复**：从上游补搬 `health_reader.py`。纯 stdlib、无新增依赖、不扩大迁移面。

该修复已在 `/tmp` 副本上实测（未改动本仓），双方法交叉验证：

| 检查 | 补搬前 | 补搬后 |
|---|---|---|
| `review_symbol_audit.py` UNRESOLVABLE 内部模块 | 4（含 health_reader） | 3（仅剩 3 项 `inference/` 按需注册，属设计意图） |
| `review_leaf_import.py` MISSING-INTERNAL | 1 | **0** |
| `review_leaf_import.py` OTHER | 0 | 0 |

补搬的文件自身 import 仅 `json` / `dataclasses` / `typing`（另有 `TYPE_CHECKING` 下的
`pathlib`），且全文 0 处 `flagscale` 引用 —— 即可直接放入，无需任何改名或二次补搬。

---

## P1 — 应当修

### P1-1 `models/vla/__init__.py` 的 `__all__` 留有悬挂名

**位置**：`flagos_robo/models/vla/__init__.py:36-39`（guarded import）与 `:42-54`（`__all__`）

```python
try:
    from flagos_robo.models.pi0.modeling_pi0 import PI0Policy
    from flagos_robo.models.pi05.modeling_pi05 import PI05Policy
except ImportError:
    pass

__all__ = [..., "PI0Policy", "PI05Policy"]   # 走 except 分支时这两个名字无绑定
```

同文件 `:29-34` 的注释**明确警告过这个反模式** ——
"a name left in `__all__` with nothing bound to it makes `from ... import *` raise AttributeError"。
当初删 `qwen_gr00t`/`vlm` 时正确地把那几个名字从 `__all__` 一并删了，
紧接着新增的 PI0/PI05 guarded import 又踩了同一个坑。

**证据**：本机无 torch 无法直接 import 真模块（两次 stub 注入均失败），
改用 AST 从真实文件算出「仅在 try 内绑定」的名字集合，再忠实重放 `import *`：

```
__all__ 声明      : 11 项
仅在 try 内绑定   : ['PI05Policy', 'PI0Policy']
RESULT: AttributeError -> module has no attribute 'PI0Policy'
```

全仓扩展扫描（`review_all_dangling.py`）：163 个 `.py`、18 个定义 `__all__` 的文件，
**悬挂仅此 1 处**。

**影响**：仅在依赖不全时暴露（CPU-only 容器、文档构建、裁剪安装、CI lint）。
正常训练环境装齐依赖后 try 成功，不触发 —— 所以 pytest 与四门禁都没抓到。

**修复**：在 `except ImportError` 分支里把两个名字从 `__all__` 移除，
与同文件处理 `qwen_gr00t`/`vlm` 的既有做法保持一致。

### P1-2 `get_addr()` 的 127.0.0.1 过滤恒失效

**位置**：`flagos_robo/runner/utils.py:316` 与 `:323`

```python
if not ip.startswith == "127.0.0.1":   # 比较 bound method 与 str
    return ip
```

**证据**（python3 实测，`ip = '127.0.0.1'`）：

```
ip.startswith == '127.0.0.1'      -> False
not ip.startswith == '127.0.0.1'  -> True     ← 永真，过滤完全失效
not ip.startswith('127.0.0.1')    -> False    ← 期望值
```

**归属**：**继承自上游**，非迁移引入 ——
`FlagScale/flagscale/runner/utils.py` 的 316/323 两行内容与行号完全一致。
同文件 `get_host_ip()`（`:299-308`）是正确实现，有 `except → "127.0.0.1"` 兜底。

**影响**：只有环回路由的机器（容器缺省网络、CI runner、离线机）会把 `127.0.0.1`
当成可对外通告的地址交给分布式启动器 → 跨节点 rendezvous 连不上。
平时"碰巧能用"是因为对 `8.8.8.8` 的 UDP connect 通常选到真实出口网卡，掩盖了失效的过滤。

**修复**：两行改为 `if not ip.startswith("127.0.0.1"):`；
顺带把 `:318`/`:325` 的裸 `except:` 收窄为 `except OSError:`。
因属上游缺陷，建议同时向 FlagScale 上游提 PR，避免下次同步被覆盖回来。

---

## P2 — 可优化

### P2-1 测试覆盖盲区集中在「新写 / 子集重写」代码上

静态引用覆盖（`review_test_coverage.py`，`tests/` 下 28 个 `.py` = 26 个 `test_*.py` + 2 个
`__init__.py`，对 `flagos_robo/` 的 128 个源模块）：

```
被测试引用     : 41/128 模块   7624/31755 LOC = 24%
零引用         : 87 个模块
按子树         : runner 64%(11/16) | train 盲区 11895 LOC | models 3%(8610 LOC 盲区)
                 cli.py 0% | run.py 0%
```

关键是**区分两类零覆盖**，风险完全不同：

| 类型 | 代表 | 上游背书 | 风险 |
|---|---|---|---|
| 逐字节搬运 | `models/`、`train/datasets/` | 有（cmp 过） | 低 |
| 子集重写 | `cli.py` 238 行 vs 上游 528 行 | 无 | **高** |
| 子集重写 | `run.py` 174 行 vs 上游 211 行 | 无 | **高** |
| 顶层编排 | `runner_train.py` 953 行 | 部分 | 中 |

`cli.py` / `run.py` 同属 commit `c1e4078`，都是在上游基础上**裁剪重写**（分别保留 45% / 82% 行数），
既无逐字节 cmp 背书、又 0% 测试覆盖 —— 这是真盲区。
重写比搬运更容易出错，而搬运代码反而有 cmp 保护，覆盖优先级应当倒过来。

旁证：`heartbeat/` 有 `test_config.py` 和 `test_gpu_health.py`，**独缺 `test_monitor.py`**，
而坏掉的正是 `monitor.py`。覆盖盲区与真实缺陷位置重合，不是巧合。

**建议**：优先给 `cli.py`（参数解析 / 子命令分派）和 `run.py` 补测试，
而不是追求整体覆盖率数字。

### P2-2 日志双通道并存

仓内有统一 logger（`flagos_robo/logger.py`，14 处 `from flagos_robo.logger import` ），
但 `flagos_robo/` 下仍有 91 处行首 `print(`（含行内调用共 93 处）。
最典型的是 `runner/utils.py`：`:36` 已 `import` 统一 logger，同文件却仍有 23 处 `print`。

归属区分：上游同文件 `print` 数也是 23 → 完全继承，非迁移引入。
而新写的 `cli.py` / `run.py` 的 `print` 数为 **0** → 新代码是干净的。

**建议**：不必全量替换（搬运代码改动会增加后续同步成本）；
但新增代码应保持零 `print`，并把这条写进贡献规范。

### P2-3 异常处理过宽

```
裸 except          : 3 处（runner/utils.py:318 / :325 / :658）
except Exception: pass : 2 处
```

三处裸 `except` 全部继承上游，且都在网络地址探测路径上 ——
与 P1-2 是同一段代码，建议随 P1-2 一并收窄为 `except OSError:`。

### P2-4 非 NVIDIA 平台的依赖声明是空洞

4 项真实 import 未在任何 requirements 中声明：
`torch_npu` / `torch_musa` / `torch_txda` / `flag_gems`，全部位于 `platforms/`。

根因：`requirements/ascend/base.txt` 与 `requirements/musa/base.txt`
目前仅含 `-r ../common.txt` 一行，注释标明 phase 3 待填。
G3 门禁把这 4 项列为 intentionally unpinned，**设计是自洽的**，不算当前缺陷。

**建议**：phase 3 接入对应硬件时必须补齐，否则 `platforms/platform_*.py` 的
降级路径会静默吞掉"依赖根本没装"这种情况，表现为莫名回退到 CPU 路径。

### P2-5 门禁盲点加固（本次 review 的最高价值项）

四门禁全绿却漏掉 P0-1，根因已读码定位（`tools/check_import_closure.py`）：

- **C1**（静态 BFS）：只在 planned set 内展开依赖；
- **C2**（动态引用）：只校验「字符串模块路径是否落在 planned set 内」，
  **不把命中的模块回灌成新的 BFS seed**。

于是 `monitor.py` 作为「人工补入的动态引用项」进了 planned set，
但它自己的 `from .health_reader import ...` 从未被任何通道走到。
`docs/migration_scope.txt:86-91` 的注释写着
"pulling it in closes the set rather than widening it" ——
该断言对 `monitor_launcher` 成立、对 `monitor` 不成立，而**没有任何自动检查验证这个断言**。

这是闭包算法的结构性盲点，不是实现 bug。三个加固方案（任一可独立落地）：

1. **把 `review_leaf_import.py` 升为门禁 G5**：逐个真 import 全部叶子模块，
   断言 MISSING-INTERNAL = 0。最直接，且本次已证明它能抓到 P0-1。
2. **改 C2 做 fixpoint 迭代**：动态命中的模块回灌为 seed，重跑 BFS 直到不动点。
   从根上消除盲点，但要改既有门禁逻辑。
3. **把 `review_all_dangling.py` 升为门禁 G6**：覆盖 `__all__` 悬挂类（P1-1 那一类）。

方案 1 成本最低、收益最直接，建议先做。

---

## P3 — 建议

**P3-1 `train/datasets/` 缺 `__init__.py`，与其余 27 个子包形态不一致。**
源仓同样缺，是忠实搬运；作为 PEP420 隐式命名空间包工作正常（已真构建 + 真安装验证）。
补一个空 `__init__.py` 可消除形态差异，代价是偏离"逐字节搬运"。本条是取舍，不是缺陷。

**P3-2 11 项声明但全仓未使用的 pin 可瘦身。**
原始差集 13 项，`wandb`（命中 3 处 yaml 环境变量）和 `pyyaml`
（命中 `tools/compare_precision.py:285`）已洗白，剩 11 项 `.py` 外引用数为 0。
清理前建议确认不是为下游 extra 预留。

**P3-3 47 处 TODO/FIXME/XXX 宜归档。**
重点一条：`backend_native_train.py:164` 的 "temporary fix ... stop the job" ——
临时措施会直接终止作业，应有 issue 跟踪而不是只留注释。

**P3-4 `tests/unit_tests/runner/elastic/` 会在仓根创建 `MagicMock/` 目录，且未被 `.gitignore` 覆盖。**
本条是修复阶段附带发现，不在首轮 review 范围内（首轮未跑 elastic 子目录）。
现象：跑过 `pytest tests/unit_tests/runner` 之后仓根出现
`MagicMock/mock.train.system.logging.log_dir/<id>/monitor/host_0_localhost_diagnostic.txt`。
成因：某处把 `MagicMock` 对象直接当路径用（`str(mock)` 成了目录名），说明该测试的
`log_dir` 没有被真正 patch 成 tmp_path。定位方法是逐文件二分，确认 `CREATED_BY`
是 `elastic/` 而非 `heartbeat/` 或本次新增的测试。
`git check-ignore -v MagicMock` 返回 NOT_IGNORED → 会被误 `git add .` 提交进仓。
两种修法：（a）测试 fixture 用 `tmp_path` 真实目录替代 MagicMock 的 `log_dir` 属性（治本）；
（b）`.gitignore` 加 `MagicMock/`（只防误提交，污染仍在）。
本次未修，仅在复验时 `rm -rf MagicMock` 清理了工作树。

---

## 已验证无问题（反向结论同样是交付物）

这 4 项是本次 review 原本怀疑、实测后推翻的，记录下来避免重复排查：

**(a) packaging 完整性 —— 无错误。**
初判「`packages.find` 漏了 `train/datasets` 会导致 wheel 缺文件」，被真构建推翻：

```
wheel 内 .py = 128  ==  git ls-files 内 .py = 128   差值 0
真 venv pip install 后 find_spec 四级全 OK（train.datasets 的 ns=True）
entry point flagos-robo 落在 venv/bin/
__pycache__ / .pyc 入库 = 0
```

机制：`setuptools.config.expand.find_packages` 默认 `namespaces=True`，
走 `PEP420PackageFinder`，不要求 `__init__.py`。

**(b) 命名残留 —— 干净。**
`flagscale` 在 `flagos_robo/**.py` 中出现 25 处（含 yaml/txt 等非 py 文件共 56 处），
**全部是溯源注释 / docstring**，可执行 import 与字符串引用为 **0**。

**(c) 硬编码绝对路径 —— 源码 0 处。**

**(d) 被裁符号的悬挂引用 —— 0 处。**
`review_symbol_audit.py` 解析全仓 328 个内部 `from ... import ...` 站点，
DANGLING = 0 → `backend/__init__.py` 裁掉 7 个类的操作很干净。
工具做过注入测试（人造 3 个悬挂符号 + 1 个不存在模块，3/3 全抓，真符号零误报）。

---

## 修复优先级

| 顺序 | 项 | 工作量 | 理由 | 状态 |
|---|---|---|---|---|
| 1 | P0-1 补搬 `health_reader.py` | 小 | 唯一会导致运行期失败的缺陷 | **已修** |
| 2 | P2-5 方案 1（G5 门禁） | 小 | 防止同类缺陷再次漏网，应与 P0-1 同批 | **已修（G5 + G6）** |
| 3 | P1-1 `__all__` 移除悬挂名 | 极小 | 一行改动 | **已修（改法有变）** |
| 4 | P1-2 + P2-3 `startswith` 与裸 except | 极小 | 同段代码，一并处理 + 上游 PR | **已修（改法有变）** |
| 5 | P2-1 给 `cli.py` / `run.py` 补测试 | 中 | 无上游背书的新代码 | 未修 |
| 6 | P2-4 phase 3 依赖补齐 | — | 随硬件接入 | 未修（按设计待硬件） |
| 7 | P2-2 / P3-* | 小 | 可随手改，不阻塞 | 未修 |

前 4 项合计工作量很小，建议一个 PR 解决。

### 已修项的落地记录（与报告原建议的偏离）

本节让读者不把上面四项当成现存缺陷，同时记录两处「照报告原建议改会改错」的地方。

**1 — P0-1 补搬 `health_reader.py`（无偏离）**
`docs/migration_scope.txt` 先加条目，再逐字节 `cp`（SHA256 与上游相同
`971b83e8…6122dc`），顺带补回 B8 批次当初以「需 health_reader，不在 scope」
为由排除的两个回归测试。实测 MISSING-INTERNAL 1→0、`heartbeat/` 全目录 32 passed。
负对照：移走该文件 → pytest RC 0→2、G5 RC 0→1；放回即恢复。

**2 — P1-1 `__all__` 悬挂名（偏离：整块删除，不是只摘 `__all__` 条目）**
报告原建议只从 `__all__` 摘掉两个名字、保留 `try/except`。探查后改为整块删除：
那个守卫的真因是**循环依赖**（`pi0/modeling_pi0.py:44 → vla.base_policy`），
不是可选依赖容错 —— `base_policy.py:23` 已硬依赖 `safetensors.torch` 且无守卫，
实测 import 炸在 `:23`，根本走不到第 36 行的 `try`。即那是死代码。
与 W2–W3 切 `qwen_gr00t` / `vlm` 的既有先例（`66e1353`）同构处理。

**3 — P1-2 环回过滤（偏离：用 `ipaddress`，不是字面前缀匹配）**
报告原建议改成 `.startswith("127.0.0.1")`。写回归测试时发现这个改法**抓不到**
Debian 系的 `127.0.1.1`（`gethostbyname(getfqdn())` 的典型返回），而它同属
`127.0.0.0/8`。故改用 stdlib `ipaddress.ip_address(ip).is_loopback`，整段环回
加 IPv6 `::1` 一并覆盖。真值表实测：`127.0.0.1` / `127.0.1.1` /
`127.255.255.254` / `::1` → True；`10.1.2.3` / `fe80::1` / 非法输入 → False。
P2-3 同段三处裸 `except:` 一并收窄为 `except OSError:`（代码层裸 except 3→0）。
偏离上游之处已在 `docs/MIGRATION_MAP.md` §五 留痕（L1/L2/L3）。

**4 — P2-5 门禁加固（范围比原建议大：G5 + G6 + 一条 CI 纪律的泛化）**
除 G5/G6 进 `tools/` 并接入 `gates.sh`（十门禁 RC 全 0）与 `ci.yml` 三个 job 外，
还修掉一个被负对照抓出来的既有缺陷：`test_negative_control_steps_are_present`
原先把工具名**硬编码**成三项，导致新加的 G5/G6 的 `--self-test` step 可以被
整段删掉而测试仍然全绿。改为从 workflow 自身推导 ——「凡 CI 信任其做真扫描的
`tools/*.py`，必须同时带 `--self-test` 调用」。
**教训：新 step 通过现有纪律测试 ≠ 现有纪律测试覆盖新 step；必须负对照「删掉新 step 测试是否报错」。**

**5 — G6 的剪枝集漏了 `_ci/`（修复过程中反向检验新发现，不在首轮 review 内）**
G6 刚进仓时在**干净代码**上报红 `RC=1`，点名的却是上游文件
`_ci/FlagScale/flagscale/models/vla/__init__.py` —— 也就是 P1-1 在上游的同构缺陷。
成因：本节 §2.2 的闭包检查要求把上游克隆到 `_ci/FlagScale` 当 `--source-root`，
而 G6 扫的是整棵仓库根，`os.walk` 剪枝集里没有 `_ci`。实测 `_ci` 存在时扫 170 个
`.py`（本仓真实 169）、`RC=1`；加入 `_PRUNE` 后扫 169、`CLEAN`、`RC=0`。
G5 不受影响是因为它从 `repo/<PKG>` 起 `rglob`，天然进不到 `_ci` —— 即
**扫描起点决定一道遍历型门禁是否需要剪枝**，同类工具不能照抄彼此的剪枝集。
配套加了**双向** self-test（`_ci/` 下种阳性必须 `RC=0`，换成未剪枝的
`vendor_not_pruned/` 下同样本必须 `RC=1`；两侧各另种一个干净顶层 `.py`
保证 `scanned>0`，不退化成零扫描 `RC=2`）：只测剪枝侧不算证明，因为
「剪枝生效」与「扫描器整体失灵」的输出完全一样。4+1 → 7 case。
同时把 `_ci/` 加进 `.gitignore`（此前 `git check-ignore` 返回 NOT_IGNORED，
一次 `git add .` 就会把别人的仓库提交进来）。
**教训：遍历型门禁的剪枝集是正确性的一部分，不是性能优化。在干净代码上误报红比漏报更糟 ——
它训练使用者忽略这道门禁。**

**仍未修的两项值得单独点出：**
- P2-4 phase 3 依赖声明 —— 按设计等硬件接入，不是遗漏。
- P2-5 的**根治方案（方案 2）仍未做**。G5/G6 是事后网，只能在缺陷落地后抓到；
  盲点本体仍在 `check_import_closure.py`：C2 动态引用扫描只检查字符串模块路径
  是否落在 planned set 内，**不把命中模块回灌成新 seed**，所以 `monitor.py` 自己的
  `from .health_reader import` 从未被任何通道走到。根治要把 C2 命中模块加入 C1
  的 seed 做 fixpoint 迭代。

---

## 证据边界

如实声明本次 review 的能力范围，避免结论被过度解读：

- **本机无 torch，全量 pytest 不可执行。** 18 个 collect error 已按缺失包归因：
  torch 9 / aiohttp 5 / safetensors 2 / pydantic 1 / numpy 1 —— **全为环境性，零内部错误**，
  与 `review_leaf_import.py` 的结论（MISSING-EXTERNAL 101 / MISSING-INTERNAL 1 / OTHER 0）互相印证。
  历史记录的「258 passed」来自 H800 容器，本次未复现。
- **覆盖率是静态引用覆盖，不是运行时行覆盖率。** 24% 是上界估计 ——
  被测试文件 import 的模块未必每行都被执行，真实行覆盖率只会更低。
- **`review_dep_audit.py` 不识别 `importlib.util.find_spec` 守卫**，
  曾对 `torchcodec` 报假阳性（`video_utils.py:38/176` 实为 find_spec 条件守卫 + 函数级惰性 import）。
  该工具只认 `try/except`。
- **`review_symbol_audit.py` 第一版有 off-by-one**（相对 import level=1 对非 `__init__` 模块
  应解析到父包），曾报 38 个假阳性，已修正并做注入测试后才采用其结论。
- 四门禁状态为本次实跑复测：`G1_RC=0 G1P_RC=0 G2_RC=0 G3_RC=0`，
  `check_import_closure.py --self-test` 输出 `SELF-TEST PASSED: 21/21 checks`。

## 审计工具清单

全部在 `robo_file/_ops/`，纯 stdlib，可在无 torch 环境运行：

| 工具 | 作用 | 退出码语义 |
|---|---|---|
| `review_dep_audit.py` | 依赖声明四项对账 | 恒 0（只报告） |
| `review_symbol_audit.py` | 全仓内部符号解析，查悬挂引用 | 恒 0 |
| `review_leaf_import.py` | 逐个真 import 叶子模块，区分内部/外部缺失 | MISSING-INTERNAL 非空 = 1，否则 0（`:101`）；**OTHER 不计入** |
| `review_test_coverage.py` | 静态引用覆盖映射 | 恒 0 |
| `review_all_dangling.py` | 全仓 `__all__` 悬挂名扫描 | 1 = 有悬挂 |

**勘误（本表初版）：** 初版把 `review_leaf_import.py` 写成「恒 0」，不成立。
`:101` 是 `return 0 if not missing_internal else 1` —— 有内部缺失时确实返回 1，
故它本就可以直接当门禁用。真正的缺陷是另一回事：OTHER 类（SyntaxError、
import 期崩溃）只打印不进退出码，所以一个语法错误能从这道门禁底下走过去。

建议把 `review_leaf_import.py` 与 `review_all_dangling.py` 纳入 CI（见 P2-5）。
**已落地** —— 两者重写进仓为 `tools/check_leaf_import.py` / `tools/check_all_dangling.py`，
退出码语义同时收紧：

| 门禁 | 工具 | 退出码语义（落地后） |
|---|---|---|
| G5 | `tools/check_leaf_import.py` | `len(MISSING-INTERNAL) + len(OTHER)`，扫到 0 个模块 → 2 |
| G6 | `tools/check_all_dangling.py` | 1 = 有悬挂，扫到 0 个文件 → 2 |

两处「零扫描 → RC=2」不是防御性冗余，而是针对实际踩到的假通过：
`review_all_dangling.py --self-test` 原本 RC=0，看着是绿的，实际是
`--self-test` 被 `len(argv)==2` 的路径参数吃掉 → 扫 0 个文件 → 打印 CLEAN。
**识别方法：任何检查器跑 self-test，先看它报告的扫描计数；计数为 0 还说 CLEAN 就是假通过。**
新工具改用 argparse 根治，并各带 5 个真 self-test case（已知阳性必须被抓到、
已知阴性必须不报）。
