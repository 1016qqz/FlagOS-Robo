# FlagOS-Robo 迁移 SOP

> 适用范围：W2-W3 把 FlagScale 的机器人训练/推理/服务代码搬进 FlagOS-Robo。
> 基线：FlagScale @ `f8e9a62c499d526a6ffa47f533bcf342330b7326`（见 `MIGRATION_MAP.md`）
> 本文档描述**流程**；范围边界在 `migration_scope.txt`，包名对照在 `MIGRATION_MAP.md`。

## 0. 为什么需要 SOP：三类实测盲区

W1 阶段对真实 FlagScale 做过完整扫描，得到一条结论：**纯靠读代码或 grep import 判定迁移范围一定会漏**。
已实测的三类盲区，每一类都对应工具里的一个检查通道：

| 盲区 | 实测证据 | 为何 grep/AST 看不见 |
|---|---|---|
| B1 传递 import | `flagscale/serve/__init__.py:15` 的 `from .utils import load_args, task_config` 把 `serve.utils`(54 LOC) 和 `serve.metric`(145 LOC) 强行拉进闭包 | 看得见，但要**传递闭包**才知道会被拉进来；人读 3 层就放弃了 |
| B2 字符串模块路径 | `flagscale/train/processor/pipeline.py:866` 用 `importlib.import_module()` 解析 YAML 里的全类路径 | 目标模块名在**运行期**才成形，源码里没有 import 语句 |
| B3 配置文本引用 | `examples/*/conf/*.yaml` 的 `entrypoint:` 字段是 launcher 读取的脚本路径 | 根本不在 `.py` 文件里，扫 Python 永远扫不到 |

B3 曾让我自己栽过一次：只用 3 个 `train_*.py` 做起点，漏掉了 `inference_pi` 和 `run_serve_vla`
两条入口，连带漏掉约 877 LOC。所以 SOP 的第一步不是"列模块"，而是"**让工具去发现入口**"。

## 1. 工具：tools/check_import_closure.py

纯 stdlib，不需要装任何依赖，也不需要 FlagScale checkout 就能自检。三个通道一一对应上表：

- **C1 静态闭包** — 从入口做 BFS 传递 import 闭包（对应 B1）
- **C2 动态引用** — 扫 `importlib.import_module` / `__import__` 及字符串字面量模块路径（对应 B2）
- **C3 文本引用** — 扫 `yaml|yml|md|sh|toml` 里的模块/脚本路径（对应 B3）

### 标准命令 G1（训练入口侧）

```bash
cd FlagOS-Robo
python3 tools/check_import_closure.py \
  --source-root ../FlagScale --package flagscale \
  --planned-from docs/migration_scope.txt \
  --discover-seeds \
  --examples-only pi0 --examples-only pi0_5 --examples-only gr00t_n1_5 \
  --cut flagscale.models.vla.qwen_gr00t --cut flagscale.models.vla.vlm
```

W2 开工时的实测基线（必须可复现）：

```
seeds   : 4
    + flagscale.inference.inference_pi
    + flagscale.serve.run_serve_vla
    + flagscale.train.train_gr00t_n1_5
    + flagscale.train.train_pi
C1 static closure : reached 80 module(s)   OUTSIDE planned set: 0
                                           declared CUT: 5 module(s) via 2 prefix(es)
C2 dynamic refs   : 171 total              outside: 0   opaque: 1
C3 text refs      : 8 total                outside: 0
PASS: migration scope is closed            RC=0
```

### 标准命令 G1'（启动器侧，必须手写两个 seed）

```bash
python3 tools/check_import_closure.py \
  --source-root ../FlagScale --package flagscale \
  --planned-from docs/migration_scope.txt \
  --seed flagscale.runner.runner_train \
  --seed flagscale.runner.backend.backend_native_train \
  --examples-only pi0 --examples-only pi0_5 --examples-only gr00t_n1_5
```

`--discover-seeds` 只读 `examples/*/conf/*.yaml` 的 `entrypoint:` 字段，而 runner 是**被 CLI
调用的**、不出现在任何 entrypoint 里，自动发现永远找不到它。所以启动器那条 import 链只能手写
seed，而且必须给两个：单给 `runner_train` 会漏掉 `backend.backend_native_train` 底下的
`elastic/` 与 `heartbeat/`。实测基线 `reached 12 / OUTSIDE 0`（11 个模块 4426 LOC + 包根）。

退出码：`0` 闭合 / `1` 有泄漏 / `2` 用法或范围定义本身有问题。

### 三条必须理解的语义

**`--examples-only` 同时收窄 seed 发现与 C3 文本扫描。** FlagScale 全仓有 28 个 `entrypoint:`
（含 LLM / compress / megatron GPT），不限定的话闭包会吞下整个 monorepo。两个通道必须对齐，
否则 C1 扫 3 个套件而 C3 扫全树，会报 65 条无关项。

**范围文件有两种条目形式，区别很要紧：**

| 写法 | 含义 |
|---|---|
| `pkg.sub` | 子树——匹配 `pkg.sub` 及其后代，按 dotted 边界匹配（`a.b` 永不匹配 `a.bc`） |
| `=pkg` | **精确**——只匹配该模块本身 |

包根（`flagscale`、`flagscale.train` …）**必须写 `=` 形式**：它们的 `__init__.py` 在 import
链上所以必须迁，但不能顺带把兄弟子包（`train.megatron`、`compress`）全拖进来。

**`--cut` 用来声明「够得到但故意不迁」，它和范围文件解决的不是同一个问题。**

这一条是 W2 开工第一天实测撞出来的。按设计 `models/vla/` 下的 `qwen_gr00t/` 与 `vlm/` 不迁，
于是范围文件里只列要迁的 7 个子包。结果扫描报 `OUTSIDE: 5` —— 因为
`flagscale/models/vla/__init__.py:28-32` 那段 `try: from .qwen_gr00t ... from .vlm ...`
在**源树里是真实存在的**，闭包必然走到它们。改范围文件只改变「算不算计划内」，改不了「够不够得到」。

**推论：对源树跑的扫描，不可能靠调范围文件让一个有意切断的模块降到 0 违例。** 违例本来就有两类，
必须分开记账：

| 类别 | 含义 | 处置 |
|---|---|---|
| 真漏 | 没想到会用到 | 纳入范围或改写掉 |
| 有意切断 | 设计上决定不迁，源树里那条 import 会在改写时删掉 | `--cut <前缀>` 声明 |

`--cut` 的输出是**独立一行 `declared CUT`**，不混进 OUTSIDE，所以「切了几个」永远看得见。
为了不让它退化成单纯的屏蔽名单（那就是换个马甲重犯「0 由构造保证」的错），工具上了两道约束：

1. **每个 `--cut` 前缀必须在本次扫描里真的被够到**，否则说明条目已过期 → exit 2 拒跑。
   和 vacuity guard 同一个思路：声明必须对应现实。
   ```
   error: --cut prefix(es) never reached by the closure (stale declaration):
     - flagscale.models.vla.nonexistent
   ```
2. **切断只有在迁移后的树里才算证实。** 代码落地后对 `--source-root . --package flagos_robo`
   跑同一个工具：`qwen_gr00t` 在那棵树里根本不存在，任何残留 import 都会以 unresolved 报出来。
   这是切断的终局证据，写进 D5 验收。

## 2. 反同义反复：为什么 "OUTSIDE: 0" 本身不算结论

第一次真扫描时工具报了 `OUTSIDE: 0 / C2 outside: 0 / C3 outside: 0`，看着像范围已闭合。
实际是**同义反复**：范围文件里写了裸 `flagscale` 子树条目，于是"任何模块都算计划内"，
0 是由构造保证的，工具报成功但什么都没判定。

现在工具会**直接拒跑**这种范围定义（exit 2）：

```
error: planned set would make this scan vacuous:
  - 'flagscale' is a subtree entry for the package root: it makes every module
    'planned' and OUTSIDE is 0 by construction. Use '=flagscale' instead.
```

**通用教训**：任何输出"违例 = 0"的检查工具，必须同时过三种自证，缺一不可。

1. **负对照** — 剔除一个已知必需项，0 必须变正。
   ```bash
   grep -v '^flagscale.models.pi0$' docs/migration_scope.txt > /tmp/scope_minus.txt
   # 用 /tmp/scope_minus.txt 重跑 → 预期 OUTSIDE: 2，RC=1
   ```
2. **反同义反复** — 范围集合本身不能宽到让 0 由构造成立（即上面的 vacuity guard）。
3. **反向对账** — 范围文件里的条目也要拿去和现实核对，见下一节。

W1 只做了第 1 条，在合成 fixture 上 13/13 全过，仍然在真实数据上栽在第 2 条上。
**合成自检通过 ≠ 在真实数据上有判别力。**

### 2.1 单向检查的结构性盲区

闭包扫描只问一个方向的问题：**我够到的每个模块，是否都在计划集内？** 反方向它根本不问。
于是一个「命名了不存在模块」的计划条目**永远不会被够到**，也就永远不会被拿去比较，
扫描照样报 PASS。

这不是假设。`flagscale.runner.backend_native_train` 这条路径在范围文件里躺了整个 W1 ——
真路径中间还有一层 `.backend.`。三道绿灯全程没看见它，因为 G1 的 reached 集里 runner
模块数就是 0（runner 不在任何 entrypoint 上），这条死路径对 G1 完全不可见。

现在工具加了反方向检查：拿计划集逐条去文件系统核对，不存在就 exit 2。

```
error: planned entry(ies) name a module that does not exist in ../FlagScale (typo or stale path):
  - flagscale.runner.backend_native_train
  The closure direction cannot catch this: an entry that is never reached is
  never compared, so the scan would still report PASS.
```

工具里现在有三道这样的拒跑检查，它们是同一条教训的三个实例 —— **声明必须拿去和现实对账**：

| 检查 | 防的是 |
|---|---|
| vacuity guard | 范围集合宽到让 0 由构造成立 |
| stale `--cut` guard | 切断声明已过期，却继续吸收同名的新泄漏 |
| dead-entry guard | 范围条目拼错或过时，单向检查看不见 |

## 3. 每批迁移的七步流程

一次搬一个"批次"（一个子系统，如 `platforms`、`models/pi0`、`train/datasets`），不要一次搬全部。

### Step 1 — 更新范围文件
把本批模块前缀写入 `docs/migration_scope.txt`。包根用 `=` 形式。

### Step 2 — 跑闭包检查，**先看它报什么**
```bash
python3 tools/check_import_closure.py --source-root ../FlagScale --package flagscale \
  --planned-from docs/migration_scope.txt --discover-seeds \
  --examples-only pi0 --examples-only pi0_5 --examples-only gr00t_n1_5
```
报出来的 OUTSIDE 项就是你漏掉的依赖。**不要手动猜，让工具告诉你。**
W1 的 `serve.utils` / `serve.metric` 就是这么发现的——读代码没读出来。

### Step 3 — 逐条裁决 OUTSIDE 项
每一条只有三种合法处置，必须在 `MIGRATION_MAP.md` 留痕：
- **纳入**：确实是运行期必需 → 加进 `migration_scope.txt`，回到 Step 2；
- **改写**：FlagScale 特有而 FlagOS-Robo 有等价物 → 记入 MIGRATION_MAP 改写表；
- **切断**：功能可删（如 megatron 专属分支）→ 记录删除理由。

### Step 4 — 逐字节搬运，不要边搬边改
用脚本 `cp` 原样搬，**迁移和改写分两次 commit**。混在一起会让精度对不齐时无法二分定位。

### Step 5 — 改写 import 与耦合点
按 `MIGRATION_MAP.md` 的改写表执行。四处已知 Megatron 耦合点见该文档"特殊改写表"。

### Step 6 — 纯净性门
```bash
python tools/check_no_megatron.py --check-env   # 必须 RC=0
```
一个文件如果**必须**在可执行代码里出现 megatron 名字（如检查器自身、
断言 megatron 被排除的测试），在文件内声明 pragma：

```
check-no-megatron: allow-self-reference
```

这是**标记**而非硬编码文件名清单，所以豁免在使用现场可见，不会悄悄扩大成
"tools/ 整个目录不扫"这种真盲区。

### Step 7 — 单测 + 容器内实跑
```bash
# 本机改完 → 同步 → 在真实 8xH800 容器内跑
../robo_file/_ops/sync.sh
../robo_file/_ops/rexec.sh 'python -m pytest tests/unit_tests/ -q'
```
**必须在容器里跑**，不是本机。本机没有 GPU、没有 pytest，跑过不代表消费者环境跑过。

## 4. 已知必须人工复核的点

C2 会把无法静态判定目标的调用点单列为 `opaque`，**不计入违例但必须人工看**。
当前全仓只有 1 处：

```
flagscale/train/processor/pipeline.py:866  import_module(<runtime value>)
```

这里解析的是 YAML 配置里的全类路径字符串。迁移后必须**实跑**一次数据处理
pipeline 验证，光靠静态检查无法确认这条路径没断。

## 5. 范围变更的纪律

`docs/migration_scope.txt` 是迁移边界的**唯一真源**，`MIGRATION_MAP.md` 是包名对照的唯一真源。
两者必须同步修改。改范围时：

1. 先改 `migration_scope.txt`；
2. 重跑 Step 2 命令确认 RC=0；
3. 重跑负对照确认 0 仍然非同义反复；
4. 在容器内重跑 `pytest tests/unit_tests/`（其中 `TestRealScopeFileIsUsable`
   会校验范围文件可解析、包根是 `=` 形式、4 个入口在范围内、megatron 训练树在范围外）。

任何绕过工具"凭经验加一个模块"的改动，都会在 W2-W3 运行期以 ImportError 的形式回来。
