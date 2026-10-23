# W1–W4 Code Review 落地记录

本文记录一轮 code review 结论的实际落地：改了什么、为什么改、怎么验证的、哪些没改以及为什么。
所有数字都是本轮在本机实测得到的，命令和日志路径随文给出，便于复核。

文中引用的 `/tmp/w4fix/*.log` 是当时跑出来的日志，属于一次性产物，读者机器上不存在；
写出来是为了交代每个数字的出处，复核请按文中给出的命令重跑。
仓内可直接复跑的是 `pytest tests/unit_tests/` 和 `tools/check_*.py --self-test`。

- 基线 commit：`7978caf`
- 代码改动落地后：`0f69f62`
- 代码变更量：11 files, 2766 insertions(+), 88 deletions(-)（`git diff --shortstat 7978caf..0f69f62`，不含本文）
- 本文自身是第 6 个 commit（docs-only），计入后为 12 files / 3104 insertions

## 0. 先说清楚：哪些结论没有在 GPU 上实跑

本轮全部工作在一台 macOS 机器上完成，**没有 GPU，也没有安装 torch / hydra / safetensors /
huggingface_hub**。这不是疏忽，而是必须写明的边界，因为它决定了下面每条证据的强度。

受影响最大的是 checkpoint resume 这一块（第 4、5 节）。它走的是 FSDP2 的
`get_model_state_dict` / `get_optimizer_state_dict` / `set_state_dict` 路径，这些是
**集合通信（collective）**，所有 rank 必须一起调用，少一个就挂住。本机跑不了，所以：

- 这些改动的正确性证据是 **AST 结构断言 + 与已验证入口的同形对照 + 变异负对照**，
  不是真实多卡跑通。
- 具体说：测试用 `ast` 解析源码，断言「`get_optimizer_state_dict` 的调用点不在
  `is_main_process` 分支内」这类结构性质，而不是执行它。
- 这类证据能抓住「忘了传参数」「把集合通信塞进单 rank 分支」这种结构错误，
  **抓不住** 数值不一致、通信死锁、显存峰值这类只有真机才暴露的问题。

**因此第 4、5 节的改动在合入前仍需一次真实多卡 save → kill → resume 验证。**
本文不声称它们已经跑通，只声称它们在结构上成立且被测试约束住了。

其余部分（零除守卫、门禁工具、CI 接线）不依赖 GPU，证据是直接实跑的。

## 1. 改前基线

两组基线数字，后面每次改动都跟它们对比。

门禁脚本（`robo_file/_ops/gates.sh`，日志 `/tmp/w4fix/gates_before.log`，耗时 45s）：

```
RC_COUNT=10  NONZERO=0
G1_RC=0 G1P_RC=0 G2_RC=0 G3_RC=0 G4A_RC=0 G4B_RC=0 G4C_RC=0 G4D_RC=0 G5_RC=0 G6_RC=0
做功量：scanned 176 python / 8 requirement files，modules=132，MISSING-INTERNAL 0，定义 __all__ 20
```

这里特意记了「做功量」数字。RC=0 本身不能证明门禁干了活——扫了 0 个文件也会返回 0。
所以每条门禁都要同时报出它扫了多少东西，这个数字非零才算真跑过。

缺陷复现（`/tmp/w4fix/repro_zerodiv.py`）：

```
被测条件源文本 (train_pi.py:735): if step % samples_per_epoch == 0:
DEFECT PRESENT: ZeroDivisionError: integer modulo by zero   RC=1
```

这个脚本不是我手写一段等价逻辑，而是用 `ast.get_source_segment` 从真实源文件里
取出那一行的原文再执行。这样「复现的是源码」而不是「复现的是我对源码的理解」。

## 2. H 条（P0）：空 epoch 导致零除

`03f6908 train(pi): switch to FSDP2 by default, guard epoch advance against empty epoch`

`samples_per_epoch` 由数据集长度算出。数据集为空、或 batch size 大于数据集时它是 0，
而 epoch 推进那行直接对它取模，必然抛 `ZeroDivisionError`。崩的位置在训练循环里，
不在配置校验阶段，所以报错信息对使用者毫无指向性。

改动（现 `flagos_robo/train/train_pi.py:853`）：

```python
if samples_per_epoch > 0 and step % samples_per_epoch == 0:
```

同时清掉了同一函数里一段不可达的死代码。

按用户要求，这个 commit 一并带上了当时工作树里未提交的 FSDP2 默认化改造（+45/-11）。

验证：

| 检查 | 结果 |
|---|---|
| 复现脚本改后 | `RC=0`，打印「OK: samples_per_epoch=0 未崩溃 (epoch=0)」 |
| 反向对照（守卫是否过严） | 「OK: 正常情况仍推进 epoch (spe=4 step=4 -> epoch=1)」 |
| `py_compile` | `rc=0` |
| 门禁 | `RC_COUNT=10 NONZERO=0`，做功量与基线逐项一致（`gates_after_H.log`） |

反向对照这一条是刻意加的。只验证「不崩了」会漏掉一种改错方式：把条件写成恒假，
那也不崩，但 epoch 永远不前进。所以必须同时验证正常路径仍然工作。

## 3. A+E 条：pi 入口没接精度对齐工具链

`f347a1c train(pi): wire the precision-alignment toolchain into the pi loop`

`flagos_robo/utils/determinism.py` 和 `precision_dump.py` 这两个模块本来就在仓里，
gr00t 入口用了，pi 入口完全没用。后果是 pi 这条路径上：DataLoader 的 worker 没有确定性
播种、没有 per-step 的精度 dump 钩子、训练日志用裸 `print` 而不是统一的 `append_step_log`。
做精度对齐时 pi 侧拿不到可比的数据。

接入点五处，关键两处在 DataLoader 构造（`train_pi.py:717-718`，gr00t 侧对应 `523-524`）：

```python
worker_init_fn=seed_worker if _det_on else None,
generator=make_dataloader_generator(seed) if _det_on else None,
```

注意是 gated 的——默认关闭，开启确定性才生效。确定性模式会牺牲吞吐，不该无条件打开。

验证：

| 检查 | 改前 | 改后 |
|---|---|---|
| 对齐符号 grep 计数（pi / gr00t） | 0 / 16 | 16 / 16 |
| 裸 `print` 计数（pi / gr00t） | 1 / 0 | 0 / 0 |
| `test_precision_dump_hook.py` | 12 passed | 18 passed |

对称性从「我读代码觉得像」变成了机器可查的数字：测试文件把断言从 gr00t 单文件
参数化到 `ENTRY_POINTS = [gr00t, pi]`，两个入口共用同一组不变量。哪个入口漏了，
对应参数化 case 报红。

变异负对照（`/tmp/w4fix/negctl_param.sh`），确认新测试不是永绿：

```
baseline            18 passed
M1 删 worker_init_fn            → 1 failed 17 passed
M2 append_step_log 挪到 log 块后 → 1 failed 17 passed
M3 dump 改读 to_dict()           → 1 failed 17 passed
restored            18 passed
```

## 4. I 条：pi 入口不认 `checkpoint.resume_from`

`21cb299 train(pi): honor checkpoint.resume_from, and save what resume needs`

review 原本写的是「pi 侧缺 resume 分支，照 gr00t 补三行」。动手后发现这个判断不完整。

pi 的 save 侧调 `save_checkpoint()` 时**根本没传 `optimizer_state_dict`**，只传了模型权重、
lr_scheduler 和 pre/postprocessor。而 `save_training_state()` 对这个参数是
`if ... is not None` 的条件保存，所以 pi 存出来的 checkpoint 目录里
**没有 `optimizer_state.safetensors` 这个文件**。

另一边，`load_training_state_fsdp2()` 无条件调用 `load_optimizer_state_fsdp2()`。

两件事放一起：只加 resume 分支，第一次恢复就 `FileNotFoundError`。这是个假修——
代码看着对称了，功能一次也用不了。所以必须 save 和 load 两侧配套改。

改动后 pi 侧的实际消费点（`grep -n resume_from`）：

```
train_pi.py:744  resume_from = config.system.checkpoint.resume_from
train_pi.py:745  if resume_from:
train_pi.py:750      Path(resume_from), policy, optimizer, lr_scheduler, ...
train_pi.py:764      epoch = replay_dataloader_to_step(...)
train_pi.py:870      optimizer_state_dict = get_optimizer_state_dict(policy, optimizer, options=options)
```

改前这个配置项在 pi 侧是 0 处消费——配置能写，写了没人读。

新增 `tests/unit_tests/train/test_checkpoint_resume.py`，四个不变量 × 2 个入口 = 8 passed：

1. `resume_from` 有真实消费点（不是只读进变量就丢掉）
2. step 和 epoch 都恢复
3. RNG 状态在 replay 前后保存/还原（否则 replay 会把随机数流推走）
4. 集合通信调用不在 `is_main_process` 分支内

第 4 条是专门针对 FSDP2 写的。`get_optimizer_state_dict` 是 collective，
一旦被包进「只有 rank 0 执行」的分支，其他 rank 不参与，整个 job 挂住不报错。
本机没法实跑验证，但「调用点不在单 rank 分支里」这个结构性质可以用 AST 查。

变异负对照（`/tmp/w4fix/negctl_resume.sh`）五种，每种恰好 1 failed 7 passed，
恢复后回到 8 passed：删 load / 删 epoch 恢复 / 删 RNG restore / 删 optimizer_state_dict /
把 collective 塞进 `is_main_process` 块。

## 5. I 条残留：replay 期间 sampler 的 epoch 是陈旧的

`305ff97 train: advance the sampler epoch while replaying a resumed dataloader`

这一条不在原始 review 清单里，是我对自己上一个 commit 做反向检验时发现的。

当时的假设是「`for _ in range(step): next(dl_iter)` 能把数据顺序复现到断点」。
为了确认，我去读了上游 `DistributedSampler` 的实现：它的 shuffle 种子是
`seed + epoch`，而 `set_epoch()` 决定 epoch。原来的代码在 replay **之前**调一次
`set_epoch`，replay 过程中跨过 epoch 边界时 sampler 的 epoch 不会跟着变。

结果：从第 N 个 epoch 中间恢复，replay 出来的样本顺序和原始训练的顺序不一致。
不崩、不报错，只是数据顺序悄悄错了——这种缺陷比崩溃难发现得多。

修法是抽出单一来源的 helper（`flagos_robo/train/utils/train_utils.py:226`）：

```python
def replay_dataloader_to_step(dl_iter, sampler, step, samples_per_epoch) -> int:
```

两个入口都调它（`train_pi.py:764`、`train_gr00t_n1_5.py:584`），不是各抄一份。
review 里 A/E/I 这几条的共同根因就是两个入口各写一套然后慢慢长歪，
所以这里刻意走单一来源。顺带消掉了 gr00t 侧 `effective_batch_size` 的重复计算。

## 6. M+P 条：三个 review 工具进仓成为门禁

`0f69f62 tools: promote three review scripts into repo gates and wire them to CI`

review 阶段写了一批一次性脚本放在仓外（`robo_file/_ops/review_*.py`）。它们找出了
真缺陷，但跑完就没人再跑，缺陷会原样长回来。本节把其中三个固化成仓内门禁。

三个工具统一的设计（不是三套各写一遍）：

- **三层 RC 语义**：`0` 干净 / `1` 发现缺陷 / `2` 工具自身失效（扫到 0 个目标）。
  第三层是刻意加的——扫不到东西时返回 0 是最危险的失效模式，门禁变成永绿装饰品。
- **`--self-test`**：自带双向 case。阳性 case（注入缺陷）必须转红，
  阴性 case（正确写法）必须保持绿。只验红不验绿的门禁会误报到被人加 allowlist 绕过。
- **`--quiet` + argparse**：CI 里当 step 跑。

| 门禁 | 文件 | 查什么 | 本仓实测 |
|---|---|---|---|
| G7 | `tools/check_symbol_refs.py` | 符号级悬挂引用 / fallback 名字缺口 | 183 py / 342 import sites，D1=0、D2=0、D3=3 allowlisted，RC=0；self-test 11 case 全 ok |
| G8 | `tools/check_dep_drift.py` | pyproject↔requirements 漂移、未声明 import、optional 未守卫 | 132 py / 34 import roots，drift=0 conflict=0，D2=4 allowlisted，D3=0，RC=0 |
| G9 | `tools/check_test_coverage.py` | 测试覆盖 ratchet（高水位钉死） | referenced 48/132，blind_loc 22514/33045，RC=0 |

G7 的 D2（fallback 名字缺口）谓词收敛过程值得记一笔：16 → 7 → 2 → 0，
每降一次是补一个语义维度，不是加 allowlist 掩盖：

1. 作用域（名字绑定要看所在函数，不是全文件）
2. `NoReturn` 调用（handler 里调了必然抛异常的函数，等价于 bail）
3. reaching-def kill（后面重新赋值会覆盖前面的缺失）
4. try 之前已绑定（except 之后读到的是预置值，不是 `NameError`）

第 4 维补完后残留的两条（`runner_train.py:721/749` 的 `result`）消失，
且人工核实过它们确实是假阳性——`:704` 有 `result = ""` 预置，上游 FlagScale
同形代码同样安全。谓词变正确了，不是把真缺陷藏起来了。

G9 的收益可以定量。旧脚本 referenced 46/132、blind 24190；G9 是 48/132、blind 22514。
差的 1676 行是 `train_pi.py`(925) + `train_gr00t_n1_5.py`(751)——这两个文件被
AST 类测试通过路径段字面量约束（`test_checkpoint_resume.py:39`、
`test_precision_dump_hook.py:61`），旧脚本只认 import 语句，看不见这种引用方式。

CI 接线（`.github/workflows/ci.yml`）：purity job 4 步、tests-cpu job 2 步。
`tests/unit_tests/tools/test_ci_workflow.py` 新增 `TestEveryGateIsWired`
两个测试，断言「盘上每个门禁都同时出现在 ci.yml 和本地 gates 脚本里」。

## 7. 本轮没做的事

| 条目 | 处置 | 理由 |
|---|---|---|
| B / B' | 推后 | `tools/check_conf_paths.py` 在仓里但 CI 从不调用。用户定调放到后续专门的 CI 工作里一起做。 |
| N | 推后 | 仓外 `gates.sh` 应提升为仓内 `tools/run_gates.sh` 做单一来源。同上。 |
| K | **不改** | pi0 的 `shuffle: false` 原本被怀疑是遗漏。查上游 `FlagScale/examples/pi0/conf/train/pi0.yaml:21` 实测也是 `false`，另三个 config（pi0_5 / gr00t_n1_5 / qwen_gr00t）两侧都是 `true`，默认值两侧都是 `False`。差异为 0，这是上游原设计不是本仓遗漏。 |

B/B'/N 推后没有停留在口头上。它们被写进 `test_ci_workflow.py` 的两个
带理由 allowlist：`KNOWN_UNWIRED_CHECKERS`（conf_paths 未接 CI）和
`KNOWN_GATES_SCRIPT_GAPS`（compare_precision 不在 gates 脚本里）。

两个 allowlist 都配了「愈合即报红」断言：缺口一旦补上，对应表项必须删掉，
否则测试报红。这防止 allowlist 腐烂成永久豁免——它是待办清单，不是垃圾桶。

## 8. 全量回归

所有改动落地后跑一遍，和第 1 节的基线对比。

门禁（`/tmp/w4fix/gates_final2.log`，42s；退出码写进日志便于复核，
16 条 RC 行与前一轮 `gates_final.log` 逐字节相同）：

```
RC_COUNT=16  NONZERO=0  GATES_EXIT=0
G1 G1P G2 G3 G4A G4B G4C G4D G4E G4F G4G G5 G6 G7 G8 G9  全部 _RC=0
做功量：132 source modules / 33045 LOC / 38 test files / referenced 48/132 / blind 22514
```

门禁数从基线 10 增至 16（新增 G4E/G4F/G4G 与 G7/G8/G9），全部返回 0，做功量数字全部非零。
日志末行有 `SUMMARY_DONE` 标记，用来确认脚本跑完整而不是中途退出。

单元测试：

```
356 passed, 16 skipped, 14 errors
```

两个注意点：

- 必须带 `--continue-on-collection-errors`。不带的话 14 个 collection error 会
  直接 Interrupt 整个 run，输出变成 `3 skipped, 14 errors`，看起来像测试几乎没跑。
- 14 个 collection error 的成因用 `grep -oE "No module named '[a-z_.]+'"` 去重后，
  全集恰好 4 项：`torch` / `hydra` / `safetensors` / `huggingface_hub`。
  本机没装这四个三方包，与本轮改动无关。

相比改动中期的 352 passed 增加 4 个，对应新增的 `TestEveryGateIsWired` 等断言。

仓外 5 个 review 脚本 RC 全 0。其中 `review_all_dangling.py` 首跑 RC=2，
读日志确认是 argparse 缺位置参数（用法是 `review_all_dangling.py <repo_root>`），
不是门禁发现；补上参数后 RC=0。

工作树：`git status --porcelain` 0 行。注意 pytest 每次都会生成一个 `MagicMock/`
目录（mock 对象的 `log_dir` 被当成真路径创建），commit 前要清掉。

## 9. 方法上的两点

**负对照是必需的，不是加分项。** 本轮每组新测试都跑了变异负对照：
手动注入缺陷确认测试转红，恢复后确认回绿，并用 `diff -q` 验证文件字节还原。
不这么做就无法区分「测试通过」和「测试永绿」。实际抓到的问题：

- G8 的 self-test 首跑失败，诊断后发现是**我的 case 设计错了**，不是工具错——
  我把「未声明但有守卫的 import」当成了应该放过的绿 case，而守卫不等于声明，
  它本来就该进红。改用真·optional-only 模块（flash-attn / torchcodec）重建对子。
- G7 的 `_names_bound_before` 一度用 `ast.walk` 收集 try 上方所有 Store，
  导致 `if cond: result = ""` 被误判成已绑定，真 `NameError` 漏报。
  这是**假阴性**，比假阳性危险。改成只认直线语句（definite-assignment 的保守近似），
  并加了一对成对 case 把这条规则钉住。
- 一个负对照脚本自己撒过谎：注入锚点字符串写错，`replace` 成了空操作，
  于是「注入缺陷后仍然全绿」被误读成测试漏报。现在所有负对照脚本都带
  `assert old in text` —— 注入前先确认锚点存在。

**门禁的发现方式从「名字形状」换成了「自我声明」。** `test_ci_workflow.py` 原先用
`glob("check_*.py")` 找门禁、用正则 `tools/[a-z_]+\.py` 在 CI 里找调用。反向检验
注入两个探针后暴露两个方向的失真：带数字的 `check_fp8_probe.py` 正则漏掉 → 假红；
不叫 `check_*` 的 `audit_probe.py` glob 看不见 → **假绿**（门禁可以靠改名逃检）。

现在的判据是 `glob("*.py")` 配合文件内含 `--self-test`：
工具自己声明「我是门禁」，而不是靠文件名长相。依据是 `tools/` 下 9 个 py 文件
全部带 `--self-test`（`grep -l` 实测）。顺带作废了「新门禁文件名不能带数字」
这条隐含约束。

## 10. 合入前的待办

1. **真实多卡验证 resume**（第 4、5 节）。最小验证：跑 N 步 → 保存 → kill →
   从 checkpoint 恢复 → 确认 step/epoch 正确、loss 曲线接得上、
   `optimizer_state.safetensors` 存在。本机无 GPU 未做。
2. **B / B' / N**（第 7 节）。完成后记得删 `test_ci_workflow.py` 里对应的
   allowlist 表项，否则测试会报红——这是设计如此。
3. `tools/test_coverage_baseline.json` 是 ratchet 高水位。覆盖率提升时用
   `--update-baseline` 重钉，这个操作要走 review，不要随手跑。

## 附：commit 列表

```
0f69f62 tools: promote three review scripts into repo gates and wire them to CI
305ff97 train: advance the sampler epoch while replaying a resumed dataloader
21cb299 train(pi): honor checkpoint.resume_from, and save what resume needs
f347a1c train(pi): wire the precision-alignment toolchain into the pi loop
03f6908 train(pi): switch to FSDP2 by default, guard epoch advance against empty epoch
```
