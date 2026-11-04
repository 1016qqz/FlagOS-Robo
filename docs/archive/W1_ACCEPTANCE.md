# W1 验收报告（实测）

本文件记录 W1 开发计划 §五「验收」六组命令的**实跑**结果。每条都给出命令原文、
实测输出、以及 RC。凡与计划原文有出入的，单列「验收命令自身的修正」一节说明，
不在正文里悄悄改命令。

- 环境：H800 单节点，容器 `flagos_robo_w1`（`--network host`），8×H800
- 容器内仓库：`/workspace/FlagOS-Robo`（= host `/public-flash/zhangqq/flagos_robo_w1/FlagOS-Robo`）
- FlagScale 参考树：`/workspace/flagscale_ref/FlagScale`，baseline `f8e9a62c499d526a6ffa47f533bcf342330b7326`
- 复现脚本（随仓库交付）：`tools/acceptance/acc_verbatim.sh`（组 3–6）、`tools/acceptance/acc1_venv.sh`（组 1–2）

---

## 0. 结论速览

| 组 | 验收内容 | RC | 结果 |
|----|----------|----|------|
| 1 | 全新 venv + `requirements/cuda/base.txt` + `pip install -e .` | 0 | PASS，见 §1 |
| 2 | `import flagos_robo` + `get_platform()` 真卡探测 | 0 | PASS，见 §1 |
| 3 | 依赖树无 Megatron（三连） | 0 | PASS（1 条交叉检查需人工裁决，见 §3） |
| 4 | 导入闭包自洽 + 工具自检 | 0 | PASS |
| 5 | 动态导入 / 残留文本扫描 | 0 | PASS（`examples/` 不属 W1，报 SKIP 而非 PASS） |
| 6 | 单元测试 | 0 | PASS，**176 passed** |

额外：三个工具的 `--self-test` 全部 RC=0（13/13、FP control 0 误报、24/24）。

---

## 1–2. 全新 venv 安装 + 导入 + 真卡平台探测

这一组是整份验收里**唯一在消费者环境下跑的**：`python -m venv`（不带
`--system-site-packages`，因此**不继承**镜像里 conda env 的 torch 2.7.1），
从 `requirements/cuda/base.txt` 全量装依赖，再 `pip install -e .`。
复现：`bash tools/acceptance/acc1_venv.sh`。

```
=== acc1 start 2026-10-09T09:10:38Z
=== acceptance 1: pip install -r requirements/cuda/base.txt
RC_REQUIREMENTS=0
=== acceptance 1b: pip install -e .
Successfully installed flagos-robo-0.1.0.dev0
RC_EDITABLE=0
=== acceptance 2: import + version
VERSION=0.1.0.dev0
RC_IMPORT=0
=== acceptance 2b: platform in a venv-only env
PLATFORM=cuda nccl cuda 0
RC_PLATFORM=0
=== acceptance 3: purity gate inside the venv
RC_PURITY=0
OK: no megatron in venv pip list
=== torch actually installed
TORCH=2.9.0+cu128 cuda=True n=8
RC_TORCH=0
=== acc1 done 2026-10-09T09:28:39Z
```

两条与计划判定字面一致：`VERSION=0.1.0.dev0`、`PLATFORM=cuda nccl cuda 0`。

**一条原本预期会失败的检查没有失败。** 我的预测是 `RC_PLATFORM` 最可能被推翻——
`base.txt` 钉的是 `torch==2.9.0`，而平台抽象层是从 FlagScale 原样搬来、只在镜像自带的
torch 2.7.1 上验证过的；若它依赖 2.7 的某个 API，2.9 上就会炸。实测 `TORCH=2.9.0+cu128`
且 `PLATFORM` 正常，**平台层在钉死的 2.9.0 上同样成立**，不存在版本兼容缺口。
这是必须在 W1 暴露、而不是留给 W2 的结论。

耗时 ~18 分钟，长极是 `torch 2.9.0` ~900MB 加上一众 CUDA 轮子（共 ~6GB），
经代理约 1 MB/s。W2 若需重跑，建议预热 pip 缓存。

> 踩坑（已修）：产物最初放在仓库内的 `_scratch/`，而 `sync.sh` 是
> `rsync -az --delete`，本机不存在该目录 → 远端 venv 与日志被当作多余文件删除，
> pip 在已删 inode 上空转、进程还活着但毫无进展。现在脚本把产物写到
> `/tmp/acc1.log` 与 `/tmp/acc1_venv`（同步树外），脚本本身留在仓库内受同步管理。

---

## 3. 依赖树无 Megatron

```
$ python tools/check_no_megatron.py
PASS: no Megatron dependency in source, declared deps, or env
RC=0

$ pip list | grep -i megatron
（空）
RC=0

$ python tools/check_no_megatron.py --self-test
SELF-TEST PASSED: every layer detects a planted violation, and
documentation mentions of megatron do NOT trigger a false positive
RC=0
```

### 需人工裁决的交叉检查

计划 §五 里还有一条裸文本检查 `grep -rn "megatron" flagos_robo/`，实测**命中 2 处**：

```
flagos_robo/platforms/platform_base.py:62:  point was bound to ``megatron.plugin.platform`` instead). Without this
flagos_robo/__init__.py:19:  **no Megatron** in the dependency tree (see ``tools/check_no_megatron.py``)
（另有 2 个 __pycache__/*.pyc 二进制命中）
```

裁决：**两处都是 docstring 里的迁移溯源说明，不是代码引用**；`.pyc` 是构建产物噪声。
权威门禁是 `check_no_megatron.py`（AST 解析，天然跳过注释/docstring），它同时 RC=0。

规则固化：裸 grep 只作人工交叉检查，与 AST 工具冲突时**以 AST 为准并逐条裁决**；
任何验收 grep 必须带 `--exclude-dir=__pycache__ --exclude='*.pyc'`。

---

## 4. 导入闭包自洽

```
$ python tools/check_import_closure.py \
    --source-root /workspace/flagscale_ref/FlagScale --package flagscale \
    --planned-from docs/migration_scope.txt \
    --discover-seeds \
    --examples-only pi0 --examples-only pi0_5 --examples-only gr00t_n1_5
RC=0
```

与 `docs/MIGRATION_SOP.md` 声明的基线逐项一致：seeds=4、C1 reached=80 OUTSIDE=0、
C2 outside=0 opaque=1（恰为已知盲区 `pipeline.py:866`）、C3 outside=0。

```
$ python tools/check_import_closure.py --self-test
SELF-TEST PASSED (13/13)
RC=0
```

反同义反复双自证（此前已实跑，见 MIGRATION_SOP §1）：
删 `flagscale.models.pi0` → OUTSIDE 0→2 RC=1；scope 写裸包根 `flagscale` → vacuity guard RC=2 拒跑。

---

## 5. 动态导入 / 残留文本

```
$ grep -rnE '["'"'"']flagscale\.' flagos_robo/
RC=0   PASS（无字符串字面量形式的 flagscale 模块路径，且目标确实被扫到）

$ grep -rn 'flagscale' examples/
TARGET_MISSING: examples/
RC=SKIP_NOT_IN_W1_SCOPE   —— 不是通过：什么都没扫
```

`examples/` 属 W2 范围，W1 仓库中不存在。

### `docs/` 里的 flagscale 引用（计数，不禁止）

```
docs/MIGRATION_MAP.md:27
docs/MIGRATION_SOP.md:14
docs/PRECISION_BASELINE.md:1
docs/migration_scope.txt:33
```

裁决：这些是**迁移来源说明**，是文档的正当内容（MIGRATION_MAP 的命名映射表、
migration_scope.txt 的待迁移模块清单）。源码树零容忍，文档树记录计数。

---

## 6. 单元测试

```
$ python -m pytest tests/unit_tests/ -q
176 passed in 8.09s
RC=0
```

构成：platforms 27 + check_no_megatron 43 + check_import_closure 22 +
compare_precision 48 + CI 一致性 13 + raw-tensor bit-exact 23（bf16 等 5 dtype 参数化）。

`test_raw_tensor_bitexact.py` 连跑 3 次均 `23 passed`，确认无 flake —— 该文件早先版本
确实 flake 过：它断言「float32 归约必然吞掉 1-ULP 单元素扰动」，而这是**输入相关**的。
容器实测（300 seeds × 3 形状，`randn` 元素 0 加 1 ULP，`sum()` 结果**不同**的比例）：

| shape | sum() 不同 | 即被吸收 |
|-------|-----------|---------|
| (4, 8) | 37/300 (12%) | 88% |
| (32, 32) | 5/300 (2%) | 98% |
| (256, 256) | 2/300 (1%) | 99% |

两个方向现在都按**实测挑出的 seed** 钉死（`_ABSORBING_SEED=0` / `_SURVIVING_SEED=8`），
并新增 `test_elementwise_catches_it_regardless_of_input` —— 逐元素路径对所有 seed 都能
看见差异，**无输入相关性**，这才是工具真正依赖的性质。

教训：汇总统计量不仅弱，而且**它漏掉什么是不确定的**；手工构造同样不可靠
（`ones(4,8)` 与「1e8 主导元素」两个我以为必然吸收的构造，实测 `sum()` 都相同／不同与预期相反）。

---

## 验收命令自身的修正

逐字实跑暴露了**验收命令**（不是交付物）的两个缺陷，已在计划与本报告中更正：

1. **闭包检查缺 flag。** 计划 §五 只写了工具名。实跑必须带
   `--source-root / --package / --planned-from / --discover-seeds / --examples-only ×3`，
   否则 RC=2。`docs/MIGRATION_SOP.md` 与 `.github/workflows/ci.yml` 里的命令本来就是完整正确的。

2. **`grep … || echo "OK"` 会空洞通过。** 第 5 组原写法在 `examples/` 不存在时
   grep 返回 2，被 `|| echo` 吞成 RC=0，看起来是「没有残留」，实际是「根本没扫」。
   本报告的 `scan()` 把三态分开：rc=1 → PASS（确实扫了且无匹配）、
   rc=0 → FAIL（有残留）、rc≥2 → ERROR/SKIP（没扫到，不算通过）。

两条都属同一病根：**门禁必须能区分「通过」与「没跑」**。这与闭包工具的 vacuity guard、
精度工具的 VACUOUS 判定是同一条纪律。
