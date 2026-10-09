# W1 验收报告（实测）

本文件记录 W1 开发计划 §五「验收」六组命令的**实跑**结果。每条都给出命令原文、
实测输出、以及 RC。凡与计划原文有出入的，单列「验收命令自身的修正」一节说明，
不在正文里悄悄改命令。

- 环境：H800 单节点，容器 `flagos_robo_w1`（`--network host`），8×H800
- 容器内仓库：`/workspace/FlagOS-Robo`（= host `/public-flash/zhangqq/flagos_robo_w1/FlagOS-Robo`）
- FlagScale 参考树：`/workspace/flagscale_ref/FlagScale`，baseline `f8e9a62c499d526a6ffa47f533bcf342330b7326`
- 复现脚本：`_scratch/acc_verbatim.sh`（组 3–6）、`_scratch/acc1_venv.sh`（组 1–2）

---

## 0. 结论速览

| 组 | 验收内容 | RC | 结果 |
|----|----------|----|------|
| 1 | 全新 venv + `requirements/cuda/base.txt` + `pip install -e .` | 见 §1 | 见 §1 |
| 2 | `import flagos_robo` + `get_platform()` 真卡探测 | 见 §1 | 见 §1 |
| 3 | 依赖树无 Megatron（三连） | 0 | PASS（1 条交叉检查需人工裁决，见 §3） |
| 4 | 导入闭包自洽 + 工具自检 | 0 | PASS |
| 5 | 动态导入 / 残留文本扫描 | 0 | PASS（`examples/` 不属 W1，报 SKIP 而非 PASS） |
| 6 | 单元测试 | 0 | PASS，161 passed |

额外：三个工具的 `--self-test` 全部 RC=0（13/13、FP control 0 误报、24/24）。

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
$ python -m pytest tests/ -q
161 passed
RC=0
```

构成：platforms 27 + check_no_megatron 43 + check_import_closure 22 +
compare_precision 48 + CI 一致性 13（含 raw-tensor bit-exact 21，bf16 等 5 dtype 参数化）。

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
