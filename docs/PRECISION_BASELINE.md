# FlagOS-Robo 精度对齐基线方案

> 适用范围：W2-W3 每批代码从 FlagScale 迁入后的精度验收。
> 参考基线：FlagScale @ `f8e9a62c499d526a6ffa47f533bcf342330b7326`，同一台 8xH800，同一容器。
> 工具：`tools/compare_precision.py`

## 0. 本次迁移属于哪种对齐场景

标准方法论有三种场景（A 框架迁移 / B 自身迭代 / C 硬件迁移）。
FlagOS-Robo 是 **A 的一个特例，且比一般的 A 更严**：

| 维度 | 一般的框架迁移 | FlagOS-Robo 迁移 |
|---|---|---|
| 硬件 | 相同 | 相同（同一台 H800，同一容器） |
| 算子实现 | **不同**（不同框架各自实现） | **相同**（代码逐字节搬运） |
| 模型结构代码 | 重新实现 | 逐字节搬运 |
| 变化的是什么 | 整个计算路径 | **只有 import 路径和包名** |

结论：**没有任何合法理由产生数值差异。**
一般的框架迁移要容忍 FA/TE 带来的 BF16 量级偏差，而这里如果出现任何非零差异，
那就是迁移把代码改坏了（漏搬一个文件、改写 import 时顺手改了逻辑、落了某个 config 默认值）。

所以本方案的验收标准是：**Level 1-4 必须 bit-exact（`atol=0, rtol=0`），Level 5 严格模式。**
这不是把标准定得过高，而是因为**我们有权利要求 bit-exact**——这是逐字节迁移给的红利，
放弃它就等于放弃了唯一能把"迁移错误"和"数值噪声"区分开的手段。

**唯一例外**：若某批次确实改写了实现（不只是 import），该批次必须在
`MIGRATION_MAP.md` 显式登记"实现已改写"，并降级到 Level 5 relaxed 模式 + 说明理由。
未登记的差异一律判为缺陷。

## 1. 六级递进（不得跳级）

Level 1-4 成本极低且能排掉绝大多数问题，跳级会让 Level 5 的 loss 偏差无法归因。

| Level | 对齐对象 | 判定 | 工具子命令 |
|---|---|---|---|
| L1 | 模型结构 | 参数名 1:1 映射，元素总数完全相同 | `params` |
| L2 | 超参 | 语义不同项全部消除；batch 等式成立 | `config` |
| L3 | 数据管道 | 前 N step 的 `input_ids` bit-exact | `tensors` |
| L4 | 权重初始化 | 全部参数 bit-exact（`atol=0,rtol=0`） | `tensors --strict` |
| L5 | loss 曲线 | 逐 step bit-exact（本场景）或 relaxed | `loss` |
| L6 | 前反向逐层 | 定位首个发散层/算子 | `tensors --per-layer` |

### L2 的 batch 等式
```
global_batch_size = micro_batch_size x DP x gradient_accumulation_steps
```
两侧必须各自成立**且结果相等**。DP 由 `world_size / (TP x PP)` 推出，改并行度时最容易算错。

### L2 差异三分类（照抄标准方法论，必须逐项归类而非一眼扫过）
- **数学等价**（ZeRO-1 vs AllReduce）→ 只需记录；
- **数值不同**（FA vs native attention）→ 一般场景可接受，**但本场景不该出现**，出现即说明改写了实现；
- **语义不同**（eps 1e-8 vs 1e-18）→ 必须修。

## 2. 确定性开关（L3-L5 严格模式的前提）

严格对齐前必须关掉所有随机性，否则 bit-exact 根本不可能成立，
而你会误以为是迁移出了问题。`tools/compare_precision.py --emit-determinism-snippet`
会打印下面这段，直接贴进两侧训练脚本入口：

```python
import os, torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # 必须在 CUDA 初始化前设置
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
```

同时必须 `attention_dropout=0`、`hidden_dropout=0`。

注意 `CUBLAS_WORKSPACE_CONFIG` 要在 CUDA 初始化**之前**设置——写在 import torch 之后
再设置是无效的，这是个静默失效点。

## 3. 发散模式诊断表

Level 5 出现偏差时，**先看形状再查代码**。偏差的形状直接指向病因：

| 偏差形状 | 病因 | 动作 |
|---|---|---|
| step 0 就不等 | 初始化没对齐 | 回 L4 |
| 线性缓增 | 微小数值差累积 | 本场景不可接受 → L6 |
| 指数增长 | 计算有 bug | L6 |
| 一条平一条降 | 梯度断了/算子坏了 | L6，优先查 backward |
| 突刺后回落 | 数值不稳定 | 查 loss scaling / grad clip |
| 周期性大误差 | 数据顺序或 LR schedule 不一致 | 回 L2-L3 |

## 4. 插桩原则：改训练循环，不写独立脚本

**不要**另写一个"对齐脚本"去模拟训练过程——它模拟不了真实的 dataloader 顺序、
不了 DP rank 切分、不了 grad accumulation 边界，对出来的"齐"是假的。

正确做法是在两侧真实训练循环里落 dump 点，`compare_precision.py` 只负责**比对落盘产物**：

```python
# 在 train step 内，两侧都插同样的代码
from flagos_robo.utils.precision_dump import dump_step   # W2 交付
dump_step(out_dir, step, input_ids=input_ids, loss=loss, model=model)
```

每侧产出一棵目录树，然后离线比对（不占 GPU）：

```bash
python3 tools/compare_precision.py tensors \
    --ref  /public-flash/zhangqq/align/ref_flagscale \
    --tgt  /public-flash/zhangqq/align/tgt_flagos_robo \
    --strict
```

**DEBUG-first**：dump 点要在启动**之前**插好。等跑挂了再回去加插桩，等于白跑一轮
——在 8 卡上这代价是以小时计的。

## 5. 实验目录纪律

每个 level 一个独立目录，**绝不覆盖**已有实验：

```
/public-flash/zhangqq/align/
├── exp001_L1_structure/
│   ├── README.md          # 做了什么、为什么、配置、结论（结论必须写明 PASS/FAIL）
│   ├── ref_params.txt
│   └── tgt_params.txt
├── exp002_L2_config/
├── exp003_L3_data/
├── exp004_L4_init/
└── exp005_L5_loss/
```

`README.md` 里**必须有显式结论**。没有结论的实验目录等于没做这个实验——
下一个人（或三周后的你）无法判断那堆 tensor 是对齐过的还是跑废的。

## 6. 工具用法

`tools/compare_precision.py` 纯 stdlib + 可选 torch（torch 只在读 `.pt` 时需要；
读 `.json`/`.txt` 产物时不需要，所以能在本机离线跑）。

```bash
# L1 结构：比参数表，输出 1:1 映射与元素总数
python3 tools/compare_precision.py params --ref ref_params.txt --tgt tgt_params.txt

# L2 超参：比 config，自动校验 batch 等式，按三分类归类差异
python3 tools/compare_precision.py config --ref ref_cfg.yaml --tgt tgt_cfg.yaml

# L3/L4 张量：bit-exact 比对
python3 tools/compare_precision.py tensors --ref ref_dir --tgt tgt_dir --strict

# L5 loss：逐 step 比对 + 自动判定发散形状
python3 tools/compare_precision.py loss --ref ref_loss.json --tgt tgt_loss.json --strict

# 确定性代码片段
python3 tools/compare_precision.py --emit-determinism-snippet

# 自检（负对照：证明工具能报错，不是恒返回 PASS）
python3 tools/compare_precision.py --self-test
```

退出码：`0` 对齐 / `1` 有偏差 / `2` 用法或输入问题。

### 为什么工具必须带 --self-test

一个"比对工具"最危险的失效模式是**恒返回 PASS**：读文件失败当成空、
两边都空当成相等、形状不同跳过不比。这种工具会给你一路绿灯，
直到 W3 末尾模型不收敛才发现从头就没对过。

`--self-test` 在临时目录造合成产物并**逐级植入已知偏差**，断言每一级都能捕获：
- L1 植入一个多出来的参数 → 必须报映射缺失
- L2 植入 eps 1e-8 vs 1e-18 → 必须归类为"语义不同"而非"数学等价"
- L3/L4 植入最后一位 bit 翻转 → strict 模式必须报 FAIL
- L5 植入各种发散形状 → 必须正确归类（step0/线性/指数/平坦）
- **假阳性对照**：完全相同的输入必须报 PASS

W1 阶段在闭包工具上栽过一次同类问题：合成 fixture 13/13 全过，
真实数据上却因为范围集合写得太宽而让"违例=0"由构造成立（同义反复）。
所以**负对照和反同义反复两种自证都要有**，单靠"自检通过"不足以信任一个输出 0 的工具。

---

## 7. 汇总统计量不能证明 bit-exact（硬规则）

这一节不是设计偏好，是本仓实测抓到的一个**漏报缺陷**的结论，W2-W3 任何人改
`tools/compare_precision.py` 都必须守住。

### 7.1 缺陷现场

工具最初对 `.pt` 张量的处理是：读入后先归约成 5 个汇总统计量
（`sum / mean / std / max / norm`），再比较这 5 个数。看起来很合理，
直到对真 torch 张量做 1-ULP 负对照：

```python
torch.manual_seed(0)
a = torch.randn(4, 8)
b = a.clone()
b[0, 0] = torch.nextafter(b[0, 0], torch.tensor(float("inf")))   # 动 1 个 ULP

torch.equal(a, b)                      # False —— 两个张量确实不同
a.sum().item() == b.sum().item()       # True  ←← float32 求和把这 1 ULP 完全吸收
a.mean().item() == b.mean().item()     # True
a.norm().item() == b.norm().item()     # True
```

于是 `--strict`（声称 bit-exact，atol=0 rtol=0）对一个**已知不同**的张量
打印 PASS、RC=0。这是最坏的一类缺陷：它不是报错，而是在你以为自己在守门时
把门开着。

### 7.1.1 吸收是输入相关的，这让缺陷更坏而不是更轻

上面写 `manual_seed(0)` 不是装饰——**换个 seed 结论就可能反过来**。
容器实测（torch 2.7.1+cu128，每形状 300 seeds，`randn` 元素 0 加 1 ULP）：

| shape | `sum()` 结果不同 | 即被吸收 |
|-------|-----------------|---------|
| (4, 8) | 37/300 (12%) | 88% |
| (32, 32) | 5/300 (2%) | 98% |
| (256, 256) | 2/300 (1%) | 99% |

所以统计量比较不是「弱一点」，而是**它漏掉什么取决于数据**：同一份代码改动，
今天的 batch 上报 PASS，明天的 batch 上报 FAIL。一个检出能力随输入漂移的门禁，
比一个始终不检出的门禁更危险，因为它会给出虚假的历史绿灯。

直觉在这里完全不可靠：我两次手工构造「必然吸收」的例子都猜错了——
`ones(4,8)` 加 1 ULP 后 `sum()` **相同**（我以为会不同），
「一个 1e8 主导元素 + 其余 1.0」的构造 `sum()` 也相同。
张量越大吸收率越高（更多同量级元素参与求和），但始终不是 100%。

推论，对 W2-W3 有约束力：**不要用随机输入去断言任何数值「显然成立」的性质**。
要么构造可证明的情形，要么先测量频率、再按实测挑 seed 把两个方向都钉死
（`tests/unit_tests/tools/test_raw_tensor_bitexact.py` 的
`_ABSORBING_SEED` / `_SURVIVING_SEED` 就是这么来的），
并额外断言逐元素路径**对所有 seed 都成立**——无输入相关性才是工具该依赖的性质。

### 7.2 根因（可推广）

> 任何声称 bit-exact 的比较器，若中间经过**有损归约**
> （求和、均值、范数、低精度转换、量化、排序去重），其 strict 模式就是假的。

有损归约的定义域压缩天然允许不同输入映射到同一输出。汇总统计量用来
**定位**发散（L5 的发散形状分类就靠它），但在原理上无法用来**判定** bit-exactness。

### 7.3 现行修法（两层，勿退化）

1. **逐元素**：两侧都拿到 raw tensor 时，strict 走 `torch.equal`，
   relaxed 走逐元素 max-rel；发现不同时报
   `N/total element(s) differ, max|delta|=...`，让结论可定位而不是只有一个 bool。
2. **字节摘要**：每个张量算 sha256，且把 `shape|dtype` 一并混进摘要
   （否则 `(4,8)` 与 `reshape(8,4)` 的 payload 字节完全相同会碰撞）。
   strict 比摘要，relaxed **不**比 —— relaxed 的本意就是容许字节不同。

### 7.4 字节提取 API：必须用 `view(torch.uint8)`

| 写法 | fp32 | **bf16** | fp16 | int64 | bool |
|---|---|---|---|---|---|
| `t.numpy().tobytes()` | OK | **TypeError: Got unsupported ScalarType BFloat16** | OK | OK | OK |
| `t.contiguous().flatten().view(torch.uint8).numpy().tobytes()` | OK | OK | OK | OK | OK |

bf16 恰好是本工作负载的训练 dtype，所以 numpy 路线在**最需要它的那个 dtype 上
直接抛异常**。`view(torch.uint8)` 先把张量重解释为字节再出 numpy，绕开了
numpy 不认识 bf16 的问题。注意必须先 `contiguous()`：转置视图与原张量共享
storage，不规范化会让逻辑上相同的张量摘要不同。

### 7.5 给 W2-W3 的纪律

- 改 `compare_tensor_dir` 后，**必须**重跑 1-ULP 负对照
  （`tests/unit_tests/tools/test_raw_tensor_bitexact.py`，21 passed）。
  这组测试里 `TestSummaryStatsCannotProveBitExactness` 显式断言
  "sum 看不见 1 ULP" —— 它是整个修法的前提，被写成断言而不是注释。
- `--self-test` 的 24/24 **不覆盖**这个缺陷：自检 fixture 走的是 `.json`
  统计量转储，`.pt` 分支要真 torch 才会执行。两者都要跑。
- 新增任何"为了省内存先归约再比"的优化前，先问：它会不会让 strict 退化成
  7.1 的样子。
